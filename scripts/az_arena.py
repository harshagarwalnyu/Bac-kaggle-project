"""Grade a trained network by playing it, against opponents that do not move.

    uv run python -m scripts.az_arena checkpoints/az/champion.pt

The training loop's own gate only ever answers "is this better than the network
before it". That is the right question to gate on and the wrong one to report:
a chain of small improvements over something weak is still weak, and the loss
curve will look beautiful throughout. So this script plays the checkpoint
against **fixed, external** opponents and reports the score.

The opponents, weakest first:

* **random** -- the floor. Anything that loses here has a sign error, not a
  training problem.
* **uniform search** -- the same MCTS at the same simulation count with a
  network that knows nothing: flat priors, zero values. This is the honest
  baseline, because it isolates what the *network* contributed from what the
  *search* contributed. A network that only matches this has learned nothing
  the search was not already doing.
* **the shipped engine, at each skill level** -- alpha-beta with the
  hand-written evaluator, which at level 6 is close to perfect play. This is
  the number that means something: it compares the learned player against the
  thing the project already ships.

**The clock is not equalised, and cannot honestly be.** MCTS at N simulations
and alpha-beta at T seconds are not commensurable, so every result here is
labelled with both budgets and is a statement about *those* settings. Raising
``--simulations`` will move the numbers.

Several checkpoints can be passed at once to see a run's progression:

    uv run python -m scripts.az_arena checkpoints/az/*.pt --games 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from connect4.az.arena import EnginePlayer, RandomPlayer, match
from connect4.az.net import Evaluator, PolicyValueNet, configure_threads, uniform_evaluator
from connect4.az.player import AZPlayer
from connect4.engine import Engine

MAX_SKILL = 6


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--games", type=int, default=20, help="games per opponent, colours split")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument(
        "--engine-time",
        type=float,
        default=0.05,
        help="seconds per move for the alpha-beta opponent",
    )
    parser.add_argument(
        "--skills",
        type=int,
        nargs="*",
        default=list(range(MAX_SKILL + 1)),
        help="engine difficulty levels to play against",
    )
    parser.add_argument("--symmetry", action="store_true", help="mirror-average every evaluation")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None, help="also write results here")
    return parser.parse_args(argv)


def grade(
    player: AZPlayer,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, dict[str, object]]:
    """Play ``player`` against every opponent and return the scoreboard."""
    results: dict[str, dict[str, object]] = {}

    opponents: list[tuple[str, object]] = [
        ("random", RandomPlayer(rng)),
        (
            "uniform-search",
            AZPlayer(uniform_evaluator, simulations=args.simulations, rng=rng),
        ),
    ]
    for skill in args.skills:
        engine = Engine(time_limit_s=args.engine_time)
        opponents.append((f"engine-skill-{skill}", EnginePlayer(engine, skill=skill)))

    for name, opponent in opponents:
        started = time.perf_counter()
        record = match(player, opponent, games=args.games)
        results[name] = {
            "wins": record.wins,
            "draws": record.draws,
            "losses": record.losses,
            "score": round(record.score, 3),
            "seconds": round(time.perf_counter() - started, 1),
        }
        print(f"    vs {name:<16} {record}  ({results[name]['seconds']}s)", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_threads(args.threads)

    missing = [p for p in args.checkpoints if not p.exists()]
    if missing:
        print(f"no such checkpoint: {', '.join(str(p) for p in missing)}", file=sys.stderr)
        return 2

    print(
        f"{args.games} games per opponent, {args.simulations} simulations per move, "
        f"engine at {args.engine_time}s per move"
    )
    report: dict[str, object] = {
        "games": args.games,
        "simulations": args.simulations,
        "engine_time": args.engine_time,
        "checkpoints": {},
    }

    for path in args.checkpoints:
        net = PolicyValueNet.load(path)
        print(f"\n{path}  ({net.parameter_count():,} parameters, {net.config})")
        # Temperature zero: this is a measurement, and a player that sometimes
        # throws a piece away would just add variance to every number below.
        player = AZPlayer(
            Evaluator(net, symmetry=args.symmetry),
            simulations=args.simulations,
            rng=np.random.default_rng(args.seed),
        )
        report["checkpoints"][str(path)] = grade(player, args, np.random.default_rng(args.seed))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
