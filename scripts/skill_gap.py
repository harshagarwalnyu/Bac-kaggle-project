"""Where does difficulty 6 actually differ from difficulty 5, and why?

Levels 0-5 and level 6 are not different algorithms. They run the same
negamax, over the same bitboard, with the same heuristic evaluator, and both
play the top-ranked move -- `choose_move` returns `ranked[0].column` outright
once `skill >= 5`. Two things separate them, and only two:

  * level 6 gets `SOLVER_TIME_LIMIT_S / TIME_LIMIT_S` times the clock (12x as
    shipped), so iterative deepening gets further before the deadline;
  * level 6 keeps its transposition table between its own moves, so each
    search continues the previous one instead of restarting.

Both differences buy exactly one thing: **depth**. So the interesting question
is not "is 6 stronger" -- it must be, weakly -- but *where the extra depth
changes the move*, because everywhere else the two levels are the same player
and the extra eleven seconds are spent confirming a decision already made.

This script samples positions across the whole game, asks both engines, and
buckets the disagreements by ply. It reports, per bucket:

  agree        both engines chose the same column
  disagree     they chose differently -- the only games where 6 can beat 5
  proof gap    6 came back with an exact result and 5 did not; the deep engine
               *knows* and the shallow one is estimating
  depth        plies of search each reached

Run it the way the app is configured, or scaled down to finish sooner --
what defines the two levels is the *ratio*, not the absolute clock:

    uv run python -m scripts.skill_gap                       # scaled, ~minutes
    uv run python -m scripts.skill_gap --time 1.0 --samples 60   # as shipped
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from connect4.bitboard import Position
from connect4.engine import MAX_PLIES, Engine, heuristic_evaluator

SOLVER_MULTIPLE = 12.0


@dataclass(slots=True)
class Probe:
    """Both engines' answer to one position."""

    ply: int
    moves: tuple[int, ...]
    shallow_move: int
    deep_move: int
    shallow_depth: int
    deep_depth: int
    shallow_exact: bool
    deep_exact: bool
    shallow_nodes: int
    deep_nodes: int
    # The deep engine's own score for the move the shallow engine wanted,
    # minus its score for the move it played. Zero means the disagreement was
    # cosmetic; large and negative means level 5 was about to give something up.
    cost_of_shallow_move: float | None

    @property
    def agree(self) -> bool:
        return self.shallow_move == self.deep_move


def sample_positions(count: int, seed: int, min_ply: int, max_ply: int) -> list[Position]:
    """Random legal positions spread across the length of a game.

    Random play, not engine play: positions from an engine-vs-engine game are
    all on one narrow trajectory, and the question here is about the space of
    positions a human can steer the bot into.
    """
    rng = random.Random(seed)
    out: list[Position] = []
    attempts = 0
    while len(out) < count and attempts < count * 100:
        attempts += 1
        target = rng.randint(min_ply, max_ply)
        pos = Position()
        for _ in range(target):
            legal = pos.legal_moves()
            if not legal:
                break
            pos.play(rng.choice(legal))
            if pos.has_won():
                break
        # A finished position has nothing to choose between.
        if pos.has_won() or pos.is_draw() or pos.moves != target:
            continue
        out.append(pos)
    out.sort(key=lambda p: p.moves)
    return out


def score_of(analysis, column: int) -> float | None:
    for evaluation in analysis.evaluations:
        if evaluation.column == column:
            return evaluation.score
    return None


def probe(pos: Position, base_time_s: float, solver_multiple: float) -> Probe:
    # Fresh engines per position: a persistent table carried between unrelated
    # positions would hand the deep engine work it did not do here.
    shallow = Engine(
        evaluator=heuristic_evaluator, max_depth=MAX_PLIES, time_limit_s=base_time_s
    )
    deep = Engine(
        evaluator=heuristic_evaluator,
        max_depth=MAX_PLIES,
        time_limit_s=base_time_s * solver_multiple,
        persist_table=True,
    )

    shallow_analysis = shallow.analyse(pos)
    deep_analysis = deep.analyse(pos)

    shallow_move = shallow.choose_move(pos, skill=5)
    deep_move = deep.choose_move(pos, skill=6)

    cost = None
    if shallow_move != deep_move:
        theirs = score_of(deep_analysis, shallow_move)
        mine = score_of(deep_analysis, deep_move)
        if theirs is not None and mine is not None:
            cost = theirs - mine

    return Probe(
        ply=pos.moves,
        moves=(),
        shallow_move=shallow_move,
        deep_move=deep_move,
        shallow_depth=shallow_analysis.stats.depth_reached,
        deep_depth=deep_analysis.stats.depth_reached,
        shallow_exact=shallow_analysis.stats.exact,
        deep_exact=deep_analysis.stats.exact,
        shallow_nodes=shallow_analysis.stats.nodes,
        deep_nodes=deep_analysis.stats.nodes,
        cost_of_shallow_move=cost,
    )


BUCKETS = ((1, 7, "opening"), (8, 16, "early middlegame"), (17, 26, "late middlegame"), (27, 42, "endgame"))


def bucket_of(ply: int) -> str:
    for low, high, name in BUCKETS:
        if low <= ply <= high:
            return name
    return "endgame"


def report(probes: list[Probe]) -> str:
    lines = [
        "",
        "Where level 6 changes the move that level 5 would have played",
        "",
        (
            f"  {'phase':<18}{'n':>4}{'agree':>8}{'differ':>8}"
            f"{'depth 5':>9}{'depth 6':>9}{'proofs 5':>10}{'proofs 6':>10}"
        ),
        "  " + "-" * 76,
    ]

    for _low, _high, name in BUCKETS:
        group = [p for p in probes if bucket_of(p.ply) == name]
        if not group:
            continue
        agree = sum(1 for p in group if p.agree)
        differ = len(group) - agree
        lines.append(
            f"  {name:<18}{len(group):>4}"
            f"{agree / len(group):>7.0%} {differ / len(group):>7.0%} "
            f"{sum(p.shallow_depth for p in group) / len(group):>8.1f} "
            f"{sum(p.deep_depth for p in group) / len(group):>8.1f} "
            f"{sum(p.shallow_exact for p in group) / len(group):>9.0%} "
            f"{sum(p.deep_exact for p in group) / len(group):>9.0%}"
        )

    total = len(probes)
    differ = [p for p in probes if not p.agree]
    proof_gap = [p for p in probes if p.deep_exact and not p.shallow_exact]
    costly = [
        p for p in differ if p.cost_of_shallow_move is not None and p.cost_of_shallow_move < -0.5
    ]

    lines += [
        "",
        f"  {len(differ)} of {total} positions ({len(differ) / max(1, total):.0%}) got a different move.",
        (
            f"  {len(proof_gap)} of {total} ({len(proof_gap) / max(1, total):.0%}) are positions "
            f"level 6 *solved* and level 5 only estimated."
        ),
        (
            f"  {len(costly)} of {total} ({len(costly) / max(1, total):.0%}) are ones where level 6 "
            f"scores level 5's move materially worse -- the games 5 actually loses."
        ),
    ]

    if differ:
        deep_nodes = sum(p.deep_nodes for p in probes)
        shallow_nodes = sum(p.shallow_nodes for p in probes)
        lines.append(
            f"  Level 6 searched {deep_nodes / max(1, shallow_nodes):.1f}x the nodes for "
            f"{sum(p.deep_depth for p in probes) / total - sum(p.shallow_depth for p in probes) / total:+.1f} "
            f"plies of average depth."
        )

    worst = sorted(
        (p for p in differ if p.cost_of_shallow_move is not None),
        key=lambda p: p.cost_of_shallow_move,
    )[:5]
    if worst:
        lines += ["", "  Worst disagreements (level 6's score for level 5's move):", ""]
        for p in worst:
            lines.append(
                f"    ply {p.ply:>2}  L5 plays column {p.shallow_move}, L6 plays {p.deep_move}  "
                f"-> {p.cost_of_shallow_move:+.2f}"
                + ("   [L6 had a proof, L5 did not]" if p.deep_exact and not p.shallow_exact else "")
            )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--time", type=float, default=0.1, dest="base_time")
    parser.add_argument("--solver-multiple", type=float, default=SOLVER_MULTIPLE)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-ply", type=int, default=1)
    parser.add_argument("--max-ply", type=int, default=34)
    args = parser.parse_args()

    positions = sample_positions(args.samples, args.seed, args.min_ply, args.max_ply)
    print(
        f"{len(positions)} positions, plies {args.min_ply}-{args.max_ply}\n"
        f"level 5: {args.base_time:g}s/move   "
        f"level 6: {args.base_time * args.solver_multiple:g}s/move\n",
        flush=True,
    )

    started = time.perf_counter()
    probes = []
    for i, pos in enumerate(positions, start=1):
        probes.append(probe(pos, args.base_time, args.solver_multiple))
        if i % 10 == 0:
            print(f"  {i}/{len(positions)}...", flush=True)

    print(report(probes))
    print(f"\n{len(probes)} positions in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
