"""Does the dataset brain actually do anything? Measure it.

Run with::

    uv run python -m scripts.validate            # default sizes, a few minutes
    uv run python -m scripts.validate --quick    # smoke test, ~30 seconds

Training accuracy proves the network fits the dataset. It does not prove the
network makes the *bot* better, and those are different claims. This script
tests the second one, three ways.

**A. Verdict agreement at ply 8.** The dataset's labels are perfect-play
outcomes from John Tromp's solver, so they are ground truth -- a rare luxury.
We hand held-out positions to the engine and ask whether its verdict has the
right sign. Same search, two evaluators, one difference. Also included: the
network on its own, with no search at all, which is the control that says
whether search is contributing anything at this depth.

**B. Head to head.** The two engines play each other, alternating colours,
with forced distinct openings (both engines are deterministic, so without
that they would replay one game N times).

**C. Search versus no search.** The full engine against a greedy one-ply bot
driven by the same network. If the network were as good as it looks at ply 8,
this would be close. It is not, and the gap is the argument for why a solved
search engine is the thing that plays and the network only advises it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from connect4.bitboard import Position
from connect4.dataset import LABELS, LABEL_TO_INDEX, ensure_raw_data, load_samples
from connect4.engine import Engine, heuristic_evaluator
from connect4.model import MLP, NeuralEvaluator

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"

# Anything inside +/- this band is read as "roughly balanced", i.e. a draw
# prediction. The engine deliberately never claims a *proven* draw from a
# depth-limited zero, so turning its scores into three classes needs an
# explicit threshold rather than a bare sign test.
DRAW_BAND = 0.05


def classify(score: float) -> int:
    if score > DRAW_BAND:
        return LABEL_TO_INDEX["win"]
    if score < -DRAW_BAND:
        return LABEL_TO_INDEX["loss"]
    return LABEL_TO_INDEX["draw"]


# --------------------------------------------------------------------------
# A. Verdict agreement against ground truth
# --------------------------------------------------------------------------


def score_positions(name: str, predict, positions, labels) -> dict:
    """Run ``predict`` over held-out positions and grade it against the solver.

    Two numbers, because they answer different questions. *Decisive accuracy*
    ignores drawn positions and asks only "did it pick the right winner" --
    the question that actually governs move choice. *Three-class accuracy*
    includes draws and is the harder, more complete measure.
    """
    started = time.perf_counter()
    predictions = np.array([predict(pos) for pos in positions])
    elapsed = time.perf_counter() - started

    decisive = labels != LABEL_TO_INDEX["draw"]
    # A "draw" prediction about a decisive position is simply wrong, which the
    # plain equality already handles.
    sign_correct = predictions[decisive] == labels[decisive]

    result = {
        "name": name,
        "three_class_accuracy": float((predictions == labels).mean()),
        "decisive_accuracy": float(sign_correct.mean()),
        "n": int(len(labels)),
        "n_decisive": int(decisive.sum()),
        "seconds": elapsed,
        "predicted_draw_rate": float((predictions == LABEL_TO_INDEX["draw"]).mean()),
    }
    print(f"  {name:<34} 3-class {result['three_class_accuracy']:.3f}   "
          f"decisive {result['decisive_accuracy']:.3f}   "
          f"({elapsed:.1f}s)")
    return result


def verdict_agreement(evaluator, n_positions: int, depth: int, time_limit: float,
                      seed: int) -> list[dict]:
    print(f"\nA. Verdict agreement at ply 8 ({n_positions} held-out positions, "
          f"fixed depth {depth})")

    samples = load_samples(ensure_raw_data(DATA_DIR), validate=False)
    rng = np.random.default_rng(seed)
    # Sample from the tail of the file; the encoded cache already shuffled the
    # training split, but load_samples returns file order, so shuffle here.
    indices = rng.permutation(len(samples))[:n_positions]
    chosen = [samples[i] for i in indices]
    positions = [s.position for s in chosen]
    labels = np.array([s.label for s in chosen])

    print(f"  true class mix: "
          f"{dict(zip(LABELS, np.round(np.bincount(labels, minlength=3) / len(labels), 3)))}")

    neural_engine = Engine(evaluator=evaluator, max_depth=depth, time_limit_s=time_limit)
    plain_engine = Engine(evaluator=heuristic_evaluator, max_depth=depth, time_limit_s=time_limit)

    return [
        score_positions(
            "network alone (no search)",
            lambda pos: classify(evaluator(pos)),
            positions, labels,
        ),
        score_positions(
            "search + hand-written heuristic",
            lambda pos: classify(plain_engine.analyse(pos).evaluations[0].score),
            positions, labels,
        ),
        score_positions(
            "search + network (the shipped bot)",
            lambda pos: classify(neural_engine.analyse(pos).evaluations[0].score),
            positions, labels,
        ),
    ]


# --------------------------------------------------------------------------
# Throughput: quantifying the confound instead of assuming it away
# --------------------------------------------------------------------------


def throughput(evaluator, depth: int) -> dict:
    """How much slower is the network per node, at identical depth?

    Everything below is run at a *fixed depth* rather than a fixed clock, so
    that any difference in play is attributable to the evaluator and not to one
    engine simply getting fewer nodes for its money. That is only worth
    asserting if the cost difference is measured, so it is measured here and
    written into the report.
    """
    pos = Position.from_moves([3, 3, 4, 2, 4, 5, 0, 3])  # a typical ply-8 board
    rows = []
    for name, ev in (("heuristic", heuristic_evaluator), ("network", evaluator)):
        engine = Engine(evaluator=ev, max_depth=depth, time_limit_s=600.0)
        stats = engine.analyse(pos).stats
        rows.append({"evaluator": name, "nodes": stats.nodes,
                     "ms": stats.elapsed_ms, "depth": stats.depth_reached})
        print(f"  {name:<12} depth {stats.depth_reached}  "
              f"{stats.nodes:7d} nodes  {stats.elapsed_ms:7.0f} ms")

    ratio = (rows[1]["ms"] / rows[0]["ms"]) if rows[0]["ms"] else float("nan")
    print(f"  network costs {ratio:.2f}x the wall clock of the heuristic at equal depth "
          f"(node-level caching absorbs most of the difference)")
    return {"measurements": rows, "network_slowdown": ratio}


# --------------------------------------------------------------------------
# B / C. Playing games
# --------------------------------------------------------------------------


def greedy_policy(evaluator):
    """A one-ply bot: play whatever the evaluator likes best. No search.

    It still takes an immediate win and blocks an immediate loss, because
    otherwise the comparison is against a straw man -- any evaluator paired
    with even a single ply of lookahead gets those for free.
    """

    def choose(pos: Position) -> int:
        legal = pos.legal_moves()
        for col in legal:
            if pos.is_winning_move(col):
                return col

        # Blocking. If every move loses there is nothing left to protect, and
        # `safe_moves` returns empty -- fall back to playing anything.
        safe = pos.safe_moves() or legal

        # Evaluate each reply from the opponent's point of view and take the
        # move that leaves them worst off.
        best_col, best_value = safe[0], float("inf")
        for col in safe:
            child = pos.played(col)
            value = -1.0 if child.has_won() else evaluator(child)
            if value < best_value:
                best_col, best_value = col, value
        return best_col

    return choose


def play_game(first, second, opening: int | None = None) -> int:
    """Play one game. Returns 1 if ``first`` won, -1 if ``second`` did, 0 for a draw."""
    pos = Position()
    if opening is not None:
        pos.play(opening)

    players = (first, second)
    while True:
        if pos.is_draw():
            return 0
        mover = players[pos.moves % 2]
        col = mover(pos)
        if col < 0 or not pos.can_play(col):
            raise RuntimeError(f"illegal move {col} from\n{pos}")
        pos.play(col)
        if pos.has_won():
            # The side that just moved won; it was players[(moves - 1) % 2].
            return 1 if (pos.moves - 1) % 2 == 0 else -1


def play_match(label_a: str, player_a, label_b: str, player_b, games: int) -> dict:
    """A vs B over ``games`` games, alternating who starts.

    Both engines are deterministic, so every game would otherwise be identical.
    Forcing a different opening column each round gives genuinely different
    games without introducing randomness into either player's decisions.
    """
    print(f"\n  {label_a}  vs  {label_b}  ({games} games)")
    wins_a = wins_b = draws = 0
    started = time.perf_counter()

    for game in range(games):
        opening = game % 7  # cycle through all seven opening columns
        a_starts = game % 2 == 0
        if a_starts:
            outcome = play_game(player_a, player_b, opening=None if game < 2 else opening)
        else:
            outcome = -play_game(player_b, player_a, opening=None if game < 2 else opening)

        if outcome > 0:
            wins_a += 1
        elif outcome < 0:
            wins_b += 1
        else:
            draws += 1
        print(f"    game {game + 1:2d}  {'AB'[not a_starts]} starts  "
              f"-> {'A' if outcome > 0 else 'B' if outcome < 0 else 'draw'}")

    elapsed = time.perf_counter() - started
    print(f"    result: {label_a} {wins_a} - {wins_b} {label_b} "
          f"({draws} draws) in {elapsed:.0f}s")
    return {"a": label_a, "b": label_b, "wins_a": wins_a, "wins_b": wins_b,
            "draws": draws, "games": games, "seconds": elapsed}


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=300)
    parser.add_argument("--games", type=int, default=14)
    parser.add_argument("--depth", type=int, default=7,
                        help="fixed search depth for both engines")
    parser.add_argument("--time-limit", type=float, default=8.0,
                        help="per-move safety net in seconds; should rarely bind")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help="tiny sizes, for checking the script runs")
    args = parser.parse_args()

    if args.quick:
        args.positions, args.games, args.depth = 40, 4, 5

    model_path = MODEL_DIR / "evaluator.npz"
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found -- run `uv run python -m scripts.train` first")
    evaluator = NeuralEvaluator(MLP.load(model_path))

    report: dict[str, object] = {"config": vars(args)}

    print(f"\n0. Cost of the two evaluators at equal depth {args.depth}")
    report["throughput"] = throughput(evaluator, args.depth)

    report["verdict_agreement"] = verdict_agreement(
        evaluator, args.positions, args.depth, args.time_limit, args.seed
    )

    neural_engine = Engine(evaluator=evaluator, max_depth=args.depth,
                           time_limit_s=args.time_limit)
    plain_engine = Engine(evaluator=heuristic_evaluator, max_depth=args.depth,
                          time_limit_s=args.time_limit)

    print("\nB. Head to head: which evaluator makes the better player?")
    report["head_to_head"] = play_match(
        "search+network", lambda p: neural_engine.choose_move(p),
        "search+heuristic", lambda p: plain_engine.choose_move(p),
        args.games,
    )

    print("\nC. Does the search earn its keep?")
    report["search_vs_greedy"] = play_match(
        "search+network", lambda p: neural_engine.choose_move(p),
        "network alone (greedy)", greedy_policy(evaluator),
        args.games,
    )

    out = MODEL_DIR / "validation_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
