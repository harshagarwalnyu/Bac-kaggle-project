"""A search-speed baseline, so "faster" can be a measurement instead of a hope.

Every optimisation in the engine is currently justified by an argument. Some of
those arguments are good ones and one of them is even backed by a timing note
in a comment, but the project has no number for how fast the search actually
is -- which means no change can be shown to have helped, and a regression can
land without anyone noticing until a game feels sluggish.

This fixes the search rather than the clock. Timing "one second of search" only
measures the clock; timing a *fixed depth* over a *fixed set of positions*
measures the engine. Node counts are then identical between runs, so the only
thing that varies is how long the machine took -- and nodes per second becomes
a number two commits can be compared on.

The positions are drawn from a fixed seed and span the game, because the search
is not uniformly hard: the opening branches widest, and the endgame resolves
early through the transposition table. A benchmark that only measured empty
boards would report a number nothing in a real game ever sees.

    uv run python -m scripts.bench                 # baseline
    uv run python -m scripts.bench --json out.json # for comparing two commits
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from connect4.bitboard import Position
from connect4.engine import Engine, heuristic_evaluator


@dataclass(slots=True)
class Result:
    ply: int
    depth: int
    nodes: int
    seconds: float

    @property
    def nps(self) -> float:
        return self.nodes / self.seconds if self.seconds else 0.0


def positions(count: int, seed: int, max_ply: int) -> list[Position]:
    """Random legal, undecided positions spread across the length of a game."""
    rng = random.Random(seed)
    out: list[Position] = []
    attempts = 0
    while len(out) < count and attempts < count * 100:
        attempts += 1
        target = rng.randint(0, max_ply)
        pos = Position()
        for _ in range(target):
            legal = pos.legal_moves()
            if not legal:
                break
            pos.play(rng.choice(legal))
            if pos.has_won():
                break
        if pos.has_won() or pos.is_draw() or pos.moves != target:
            continue
        out.append(pos)
    out.sort(key=lambda p: p.moves)
    return out


def measure(pos: Position, depth: int) -> Result:
    # No time limit and no persistent table: a deadline would make the node
    # count depend on the machine, and a shared table would make each position
    # cheaper than the last, so the benchmark would measure the order of the
    # list rather than the speed of the search.
    engine = Engine(
        evaluator=heuristic_evaluator,
        max_depth=depth,
        time_limit_s=float("inf"),
    )
    started = time.perf_counter()
    analysis = engine.analyse(pos)
    elapsed = time.perf_counter() - started
    return Result(
        ply=pos.moves,
        depth=analysis.stats.depth_reached,
        nodes=analysis.stats.nodes,
        seconds=elapsed,
    )


def report(results: list[Result], depth: int) -> str:
    nodes = sum(r.nodes for r in results)
    seconds = sum(r.seconds for r in results)
    per_position = sorted(r.nps for r in results)

    lines = [
        "",
        f"Fixed-depth search speed (depth {depth}, {len(results)} positions)",
        "",
        f"  nodes            {nodes:>14,}",
        f"  seconds          {seconds:>14.2f}",
        f"  nodes/second     {nodes / seconds if seconds else 0:>14,.0f}",
        "",
        f"  median position  {statistics.median(per_position):>14,.0f} nodes/s",
        f"  slowest          {per_position[0]:>14,.0f} nodes/s",
        f"  fastest          {per_position[-1]:>14,.0f} nodes/s",
        "",
        "  The node total is deterministic. If it moves between two commits, the",
        "  search changed shape -- which may be the point, but it means the",
        "  nodes/second figures are no longer measuring the same work.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--positions", type=int, default=40)
    parser.add_argument("--max-ply", type=int, default=28)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    board = positions(args.positions, args.seed, args.max_ply)
    print(f"{len(board)} positions, plies 0-{args.max_ply}, depth {args.depth}\n", flush=True)

    results = [measure(pos, args.depth) for pos in board]
    print(report(results, args.depth))

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "depth": args.depth,
                    "seed": args.seed,
                    "nodes": sum(r.nodes for r in results),
                    "seconds": round(sum(r.seconds for r in results), 4),
                    "results": [
                        {"ply": r.ply, "nodes": r.nodes, "seconds": round(r.seconds, 5)}
                        for r in results
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nraw timings -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
