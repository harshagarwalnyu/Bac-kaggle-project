"""The AlphaZero loop: play, learn, and only keep what wins.

    uv run python -m scripts.train_az --iterations 20 --games 200

Each iteration is four steps.

1. **Play.** The *champion* generates self-play games. Not the network being
   trained -- the champion. Generating data with a network that has not yet
   proven itself is how a bad step compounds: worse weights make worse games
   make worse weights.
2. **Learn.** Sample the replay buffer and take optimiser steps. The buffer
   spans several iterations, so the network fits a window of recent play rather
   than the last batch of it.
3. **Grade.** Play the challenger against the champion over a book of forced
   openings, both seats each. The training loss is *not* evidence here: it is
   measured against targets the network's own search produced, so it says how
   self-consistent the network is and nothing about whether it plays better.
4. **Promote, or throw the iteration away.** The challenger has to clear
   ``--gate`` to take the title. A gate above one half is the whole safety
   mechanism -- without it the loop random-walks and there is no direction in
   which it is guaranteed to travel.

**On resuming.** ``--resume`` reloads the champion and continues. The replay
buffer is not persisted: it is tens of megabytes of positions that the champion
can regenerate, and a buffer written by an older champion is exactly the stale
data step 1 exists to avoid. The first iteration after a resume is therefore
data-starved and usually fails its gate. That is correct, not a bug.

**Interrupting is safe.** Ctrl-C between iterations writes the champion out
before exiting; the worst case is losing the current iteration's games.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from connect4.az.arena import EnginePlayer, RandomPlayer, match
from connect4.az.cache import CachedEvaluator
from connect4.az.mcts import MCTSConfig
from connect4.az.net import Evaluator, NetConfig, PolicyValueNet, configure_threads
from connect4.az.player import AZPlayer
from connect4.az.selfplay import SelfPlayConfig, play_games
from connect4.az.train import ReplayBuffer, TrainConfig, train
from connect4.engine import Engine

CHAMPION = "champion.pt"
CHALLENGER = "challenger.pt"
LOG = "training_log.jsonl"

#: ``NetConfig`` uses ``slots=True``, and on a slotted dataclass the class
#: attribute is the slot descriptor, not the default value -- ``NetConfig.channels``
#: is a ``member_descriptor``. An instance is the only way to read the defaults.
NET_DEFAULTS = NetConfig()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("checkpoints/az"))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--games", type=int, default=200, help="self-play games per iteration")
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--parallel", type=int, default=128, help="games advanced in lockstep")
    parser.add_argument("--steps", type=int, default=400, help="optimiser steps per iteration")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--buffer", type=int, default=200_000, help="replay buffer capacity")
    parser.add_argument("--channels", type=int, default=NET_DEFAULTS.channels)
    parser.add_argument("--blocks", type=int, default=NET_DEFAULTS.blocks)
    parser.add_argument("--arena-games", type=int, default=40)
    parser.add_argument("--arena-simulations", type=int, default=64)
    parser.add_argument(
        "--gate",
        type=float,
        default=0.55,
        help="score the challenger must reach to be promoted (0.5 is a dead heat)",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="continue from the saved champion")
    parser.add_argument(
        "--benchmark-every",
        type=int,
        default=5,
        help="iterations between reference matches against the shipped engine (0 to skip)",
    )
    return parser.parse_args(argv)


def load_or_create(path: Path, config: NetConfig, resume: bool) -> PolicyValueNet:
    if resume and path.exists():
        net = PolicyValueNet.load(path)
        print(f"resumed champion from {path} ({net.parameter_count():,} parameters)")
        return net
    net = PolicyValueNet(config)
    print(f"new champion, {net.parameter_count():,} parameters {config}")
    return net


def check_checkpoint_round_trip(net: PolicyValueNet, path: Path) -> None:
    """Save and reload once, before anything expensive happens.

    A checkpoint that cannot be reloaded is worth discovering on iteration one
    rather than at the end of an overnight run.
    """
    net.save(path)
    PolicyValueNet.load(path)


def benchmark(player: AZPlayer, games: int, rng: np.random.Generator) -> dict[str, float]:
    """Score the champion against fixed, external opponents.

    Self-play grading only ever says "better than the previous one", which is
    true of a chain that started weak and stayed weak. These opponents do not
    move, so the numbers are comparable across the whole run.
    """
    scores = {"random": match(player, RandomPlayer(rng), games=games).score}
    for skill in (2, 4, 5):
        engine = EnginePlayer(Engine(time_limit_s=0.05), skill=skill)
        scores[f"engine{skill}"] = match(player, engine, games=games).score
    return scores


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    config = NetConfig(channels=args.channels, blocks=args.blocks)
    champion = load_or_create(args.out / CHAMPION, config, args.resume)
    check_checkpoint_round_trip(champion, args.out / CHAMPION)

    # The challenger is one module object for the whole run, reset to the
    # champion's weights at the start of each iteration rather than rebuilt.
    # That keeps its parameter *objects* stable, which is what lets a single
    # Adam optimiser -- and the moment estimates it has accumulated -- survive
    # from one iteration to the next. Building a fresh network each time would
    # throw those away and make the first steps of every iteration needlessly
    # noisy.
    challenger = PolicyValueNet(champion.config)
    optimizer = None

    # One cache spans the whole run and is cleared on promotion. A cached
    # evaluation is exact -- the same weights on the same position give the same
    # number -- but only for as long as the weights hold still, so promotion and
    # clearing have to happen together or the loop starts training on answers
    # from a network that no longer exists.
    cache = CachedEvaluator(Evaluator(champion), capacity=2_000_000)
    buffer = ReplayBuffer(capacity=args.buffer)

    selfplay_config = SelfPlayConfig(
        games_in_parallel=args.parallel, mcts=MCTSConfig(simulations=args.simulations)
    )
    train_config = TrainConfig(
        batch_size=args.batch_size, steps=args.steps, learning_rate=args.learning_rate
    )
    log_path = args.out / LOG

    for iteration in range(1, args.iterations + 1):
        started = time.perf_counter()
        cache.reset_stats()

        positions = 0
        for examples in play_games(cache, args.games, selfplay_config, rng):
            buffer.add(examples)
            positions += len(examples)
        played = time.perf_counter() - started

        challenger.load_state_dict(champion.state_dict())
        if optimizer is None:
            optimizer = torch.optim.Adam(
                challenger.parameters(),
                lr=args.learning_rate,
                weight_decay=train_config.weight_decay,
            )
        report = train(challenger, buffer, train_config, rng, optimizer)

        challenger_player = AZPlayer(
            Evaluator(challenger), simulations=args.arena_simulations, rng=rng
        )
        champion_player = AZPlayer(Evaluator(champion), simulations=args.arena_simulations, rng=rng)
        record = match(challenger_player, champion_player, games=args.arena_games)

        challenger.save(args.out / CHALLENGER)
        promoted = record.score >= args.gate
        if promoted:
            # Copy the weights across rather than rebinding the name: the
            # challenger module is reused next iteration, and aliasing the two
            # would make every subsequent gate a network playing itself.
            champion.load_state_dict(challenger.state_dict())
            champion.save(args.out / CHAMPION)
            cache = CachedEvaluator(Evaluator(champion), capacity=2_000_000)

        elapsed = time.perf_counter() - started
        entry = {
            "iteration": iteration,
            "games": args.games,
            "positions": positions,
            "buffer": len(buffer),
            "policy_loss": round(report.policy, 4),
            "value_loss": round(report.value, 4),
            "arena": str(record),
            "score": round(record.score, 3),
            "promoted": promoted,
            "cache_hit_rate": round(cache.hit_rate, 3),
            "selfplay_games_per_second": round(args.games / played, 2),
            "seconds": round(elapsed, 1),
        }

        due = args.benchmark_every and (
            iteration % args.benchmark_every == 0 or iteration == args.iterations
        )
        if due:
            entry["benchmark"] = {
                k: round(v, 3)
                for k, v in benchmark(
                    AZPlayer(Evaluator(champion), simulations=args.arena_simulations),
                    games=8,
                    rng=rng,
                ).items()
            }

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(
            f"[{iteration:3d}/{args.iterations}] {report}  arena {record}  "
            f"{'PROMOTED' if promoted else 'kept champion'}  {elapsed:.0f}s",
            flush=True,
        )
        if "benchmark" in entry:
            print(f"          vs fixed opponents: {entry['benchmark']}", flush=True)

    champion.save(args.out / CHAMPION)
    print(f"done. champion at {args.out / CHAMPION}, log at {log_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted; the last promoted champion is already on disk", file=sys.stderr)
        sys.exit(130)
