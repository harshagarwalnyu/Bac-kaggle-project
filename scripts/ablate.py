"""Which half of solver mode is doing the work: the clock, or the table?

``api.py`` builds difficulty 6 with two changes over difficulty 5 -- twelve
times the time limit, and a transposition table that survives between moves.
A comment beside it used to assert the table was "the bigger of the two".
That is a plausible claim: consecutive searches in one game really do overlap
enormously, so a kept table really should turn each move into a continuation
of the last rather than a fresh start. But it was never measured, and an
unmeasured claim in shipped code is a guess wearing a lab coat, so the comment
now points here instead. This script is what gets to answer it.

So: cross the two knobs and play the four resulting configurations against
each other.

    fast + fresh   the difficulty 5 configuration
    slow + fresh   the clock alone
    fast + kept    the table alone
    slow + kept    the difficulty 6 configuration

The reading that matters is not who wins overall -- ``slow + kept`` will --
but which single knob recovers more of the gap from ``fast + fresh``. If the
comment is right, ``fast + kept`` outscores ``slow + fresh``.

Same discipline as the tournament, for the same reasons: the engine is
deterministic, so variation comes from a fixed-seed set of openings rather
than from replaying one game; every pairing is played from both seats,
because Connect 4 is a first-player win with perfect play; and every game gets
a fresh engine, so no result depends on the order the matches ran in.

    uv run python -m scripts.ablate                      # scaled, ~minutes
    uv run python -m scripts.ablate --time 1.0 --max-seconds 5400   # as shipped
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from connect4.bitboard import Position
from connect4.engine import MAX_PLIES, Engine, heuristic_evaluator
from scripts.tournament import Outcome, Record, make_openings


@dataclass(frozen=True, slots=True)
class Config:
    """One corner of the 2x2. ``name`` is what shows up in the table."""

    name: str
    multiple: float
    persist: bool

    @property
    def label(self) -> str:
        clock = "slow" if self.multiple > 1 else "fast"
        table = "kept" if self.persist else "fresh"
        return f"{clock}+{table}"


def configs(solver_multiple: float) -> tuple[Config, ...]:
    return (
        Config("L5", 1.0, persist=False),
        Config("clock", solver_multiple, persist=False),
        Config("table", 1.0, persist=True),
        Config("L6", solver_multiple, persist=True),
    )


def build_engine(config: Config, base_time_s: float) -> Engine:
    return Engine(
        evaluator=heuristic_evaluator,
        max_depth=MAX_PLIES,
        time_limit_s=base_time_s * config.multiple,
        persist_table=config.persist,
    )


@dataclass
class Report:
    base_time_s: float
    solver_multiple: float
    openings: list[tuple[int, ...]]
    outcomes: list[tuple[Config, Config, Outcome]] = field(default_factory=list)


def play_game(
    first: Config,
    second: Config,
    opening: tuple[int, ...],
    base_time_s: float,
) -> Outcome:
    """One game. ``first`` moves first, which is worth more than either knob."""
    engines = {1: build_engine(first, base_time_s), 2: build_engine(second, base_time_s)}

    pos = Position.from_moves(opening)
    started = time.perf_counter()

    while not pos.is_draw():
        player = pos.current_player()
        # Skill 5 and skill 6 take the same branch in `choose_move` -- the top
        # ranked move, no softening -- so the skill number passed here cannot
        # be what separates the configurations. Only the engine can be.
        pos.play(engines[player].choose_move(pos, skill=5))
        if pos.has_won():
            return Outcome(
                first=0,
                second=0,
                opening=opening,
                winner=player,
                plies=pos.moves,
                seconds=time.perf_counter() - started,
            )

    return Outcome(
        first=0,
        second=0,
        opening=opening,
        winner=0,
        plies=pos.moves,
        seconds=time.perf_counter() - started,
    )


def run(report: Report, verbose: bool = True, max_seconds: float | None = None) -> Report:
    """Play the round robin, opening by opening.

    The loop is opening-major rather than pairing-major so that ``max_seconds``
    can cut the run without tilting it. Every opening plays all twelve ordered
    pairings, so a run stopped between openings is still a balanced design --
    fewer openings, but every configuration has met every other one the same
    number of times from each seat. Stopping mid-opening would hand whichever
    configurations happened to be scheduled early a few extra games, and the
    share-of-the-gap arithmetic downstream would silently divide by that.

    ``--time 1.0`` is a run of hours, which is exactly when an unattended bound
    is worth having.
    """
    if max_seconds is not None and not (math.isfinite(max_seconds) and max_seconds >= 0):
        # argparse takes `nan` and `inf` as floats without complaint, and both
        # quietly disable the bound: every comparison against nan is false, and
        # nothing is ever >= inf. A negative budget is the opposite failure --
        # it stops after one opening no matter what was asked for. All three
        # produce a run that does not match its command line, which is worse
        # than a run that refuses to start.
        raise ValueError(f"max_seconds must be a finite, non-negative number, got {max_seconds!r}")

    pairs = list(itertools.permutations(configs(report.solver_multiple), 2))
    total = len(pairs) * len(report.openings)
    played = 0
    started = time.perf_counter()

    for index, opening in enumerate(report.openings, start=1):
        for first, second in pairs:
            outcome = play_game(first, second, opening, report.base_time_s)
            report.outcomes.append((first, second, outcome))
            played += 1
            if verbose:
                result = "draw" if not outcome.winner else (
                    first.name if outcome.winner == 1 else second.name
                )
                print(
                    f"  [{played:>3}/{total}] {first.label} vs {second.label}: "
                    f"{result:>6} in {outcome.plies} plies, {outcome.seconds:5.1f}s",
                    flush=True,
                )

        elapsed = time.perf_counter() - started
        if max_seconds is not None and elapsed >= max_seconds and index < len(report.openings):
            report.openings = report.openings[:index]
            print(
                f"\n  stopping after {index} of the requested openings: "
                f"{elapsed:.0f}s past the {max_seconds:g}s budget.",
                flush=True,
            )
            break
    return report


def records(report: Report) -> dict[str, Record]:
    out = {config.name: Record() for config in configs(report.solver_multiple)}
    for first, second, outcome in report.outcomes:
        winner, loser = (first, second) if outcome.winner == 1 else (second, first)
        if outcome.winner:
            out[winner.name].wins += 1
            out[loser.name].losses += 1
        else:
            out[first.name].draws += 1
            out[second.name].draws += 1
    return out


def format_report(report: Report) -> str:
    table = records(report)
    order = configs(report.solver_multiple)
    lines = [
        "",
        "Which knob buys the strength",
        "",
        f"  {'config':<10}{'clock':>8}{'table':>8}{'w-d-l':>10}{'points':>9}",
        "  " + "-" * 45,
    ]
    for config in order:
        record = table[config.name]
        lines.append(
            f"  {config.name:<10}"
            f"{config.multiple:>7.0f}x"
            f"{'kept' if config.persist else 'fresh':>8}"
            f"{record!s:>10}"
            f"{record.points:>6.1f}/{record.played}"
        )

    floor = table["L5"].points
    ceiling = table["L6"].points
    span = ceiling - floor
    lines += ["", "  Share of the level-5-to-level-6 gap each knob recovers on its own:", ""]
    if span <= 0:
        lines.append("    The two endpoints did not separate -- run more openings or a longer clock.")
        return "\n".join(lines)

    for name in ("clock", "table"):
        share = (table[name].points - floor) / span
        lines.append(f"    {name:<8} {share:>6.0%}")

    clock_share = (table["clock"].points - floor) / span
    table_share = (table["table"].points - floor) / span
    games = table["L5"].played
    lines += ["", *verdict(clock_share, table_share, games, report.base_time_s)]
    return "\n".join(lines)


def standard_error(games: int) -> float:
    """Points of standard error on a config's score, in units of the L5-L6 span.

    A score out of ``games`` is a sum of Bernoulli-ish results, so its standard
    error is about ``sqrt(games / 4)`` points. Reported because a share computed
    from 36 games looks exactly as precise as one from 3,600 and is not.
    """
    return (games / 4.0) ** 0.5


def verdict(
    clock_share: float, table_share: float, games: int, base_time_s: float
) -> list[str]:
    """What this run does and does not license anyone to say.

    The temptation is to read a separation and declare the api.py comment
    settled. Two things forbid it. The sample is small, so the shares carry a
    standard error worth naming. And more importantly the *clock is the
    independent variable of the table's own advantage*: what a persistent table
    buys is the overlap between one search and the next, and at a short base
    clock the searches are shallow and overlap little. A scaled-down run is a
    fair test of the clock and a hostile one for the table.
    """
    lines = [
        (
            f"  +/- {standard_error(games):.1f} points of standard error on each "
            f"score ({games} games per config)."
        ),
    ]

    margin = 0.1
    if table_share > clock_share + margin:
        lines.append("  The persistent table is the bigger of the two.")
    elif clock_share > table_share + margin:
        lines.append("  The clock is the bigger of the two here.")
    else:
        lines.append("  Neither knob dominates within the margin of this sample.")

    if base_time_s < 1.0:
        lines += [
            "",
            (
                f"  Read that with the clock in mind: at {base_time_s:g}s a move this "
                f"is a scaled-down"
            ),
            "  run, and search overlap between consecutive moves -- the entire thing a",
            "  persistent table sells -- grows with the budget. Re-run at --time 1.0",
            "  before treating any result about the table as settled.",
        ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--time", type=float, default=0.1, dest="base_time")
    parser.add_argument("--solver-multiple", type=float, default=12.0)
    parser.add_argument("--openings", type=int, default=4)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="stop after the first whole opening that finishes past this budget",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    openings = make_openings(args.openings, args.opening_plies, args.seed)
    report = Report(
        base_time_s=args.base_time,
        solver_multiple=args.solver_multiple,
        openings=openings,
    )

    print(
        f"{len(openings)} openings, both seats, 4 configurations\n"
        f"base clock {args.base_time:g}s, solver multiple {args.solver_multiple:g}x\n",
        flush=True,
    )

    started = time.perf_counter()
    run(report, verbose=not args.quiet, max_seconds=args.max_seconds)
    print(format_report(report))
    print(f"\n{len(report.outcomes)} games in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
