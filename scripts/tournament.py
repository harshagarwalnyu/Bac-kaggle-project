"""Round-robin self-play: every difficulty against every other difficulty.

Why this is not simply "play them 100 times and count"
------------------------------------------------------
The engine is deterministic. `choose_move` picks the n-th best move from an
honest search with no randomness anywhere -- that is a deliberate design
decision (a bot that plays well and then randomly throws a piece away feels
broken). The consequence is that difficulty 5 against difficulty 6 from the
empty board is *one game*, and playing it a hundred times gives you that same
game a hundred times.

So two things vary instead, and nothing else:

1. **The seat.** Connect 4 is solved and the first player wins with perfect
   play, so who moves first is the single largest term in any result here.
   Every pair is therefore played in both seats, and the report keeps the two
   apart. A one-number answer to "does 5 beat 6" would be hiding this.

2. **The opening.** Each match starts from a shared book of random-but-legal
   openings, generated once from a fixed seed and reused for every pairing, so
   that all levels face exactly the same set of positions. Games are the unit
   of variation; the engines stay deterministic within a game.

What the numbers do and do not mean
-----------------------------------
Strength here is a function of the clock. Levels 0-5 differ only in *which*
ranked move they play, off one search at the base budget; level 6 is the same
search given `--solver-multiple` times the budget and a transposition table
that survives between its own moves. Shrink the clock and level 6 loses the
advantage that defines it. The defaults below are small enough to finish in
minutes, which makes this a comparison at a short time control -- not a claim
about the engine at the settings the app ships.

    uv run python -m scripts.tournament                    # quick, ~minutes
    uv run python -m scripts.tournament --time 1.0 --openings 8   # slow, honest
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from connect4.bitboard import Position
from connect4.engine import MAX_PLIES, Engine, heuristic_evaluator

# Mirrors api.py: skill 6 is solver mode, everything below it shares one engine.
SOLVER_SKILL = 6
SKILLS = tuple(range(SOLVER_SKILL + 1))


@dataclass(slots=True)
class Outcome:
    """One finished game, from the point of view of nobody in particular."""

    first: int
    second: int
    opening: tuple[int, ...]
    winner: int  # 1 = the side that moved first, 2 = the other, 0 = draw
    plies: int
    seconds: float
    #: CPU time the game actually consumed. Wall clock alone cannot tell a slow
    #: game from a sleeping laptop: Windows modern standby keeps
    #: `perf_counter` running while the process is suspended, which is how one
    #: overnight ablation game came back reading 24,032 seconds for a normal
    #: 38-ply game. The two readings diverging is the signature, so both are
    #: kept. Defaulted, because the tournament does not measure it.
    cpu_seconds: float = 0.0


@dataclass(slots=True)
class Record:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def played(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def points(self) -> float:
        """Chess scoring: a draw is half a win. Connect 4 draws are rare enough
        that ignoring them would still rank correctly, but not honestly."""
        return self.wins + 0.5 * self.draws

    def __str__(self) -> str:
        return f"{self.wins}-{self.draws}-{self.losses}"


@dataclass
class Report:
    base_time_s: float
    solver_multiple: float
    openings: list[tuple[int, ...]]
    outcomes: list[Outcome] = field(default_factory=list)


def build_engine(skill: int, base_time_s: float, solver_multiple: float) -> Engine:
    """A fresh engine per game, configured the way ``api.py`` configures it.

    Fresh, not shared: solver mode keeps its transposition table between moves,
    and a table carried from a previous game would make a match's result depend
    on the order the matches happened to run in. Within one game the table is
    exactly the advantage solver mode is supposed to have, so it stays.
    """
    if skill >= SOLVER_SKILL:
        return Engine(
            evaluator=heuristic_evaluator,
            max_depth=MAX_PLIES,
            time_limit_s=base_time_s * solver_multiple,
            persist_table=True,
        )
    return Engine(
        evaluator=heuristic_evaluator,
        max_depth=MAX_PLIES,
        time_limit_s=base_time_s,
    )


def make_openings(count: int, plies: int, seed: int) -> list[tuple[int, ...]]:
    """A shared book of short random openings, identical for every pairing.

    Rejects any opening that is already decided, and any duplicate: two matches
    starting from the same position would be the same game twice and would
    weight that position double.
    """
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    book: list[tuple[int, ...]] = []

    # The empty board is always in the book: it is the position the app itself
    # starts from, so leaving it out would measure everything except the game
    # people actually play.
    if plies == 0 or count > 0:
        book.append(())
        seen.add(())

    attempts = 0
    while len(book) < count and attempts < count * 200:
        attempts += 1
        pos = Position()
        moves: list[int] = []
        for _ in range(plies):
            legal = pos.legal_moves()
            if not legal:
                break
            col = rng.choice(legal)
            pos.play(col)
            moves.append(col)
            if pos.has_won():
                break
        if pos.has_won() or pos.is_draw() or len(moves) < plies:
            continue
        key = tuple(moves)
        if key in seen:
            continue
        seen.add(key)
        book.append(key)

    return book


def play_game(
    first_skill: int,
    second_skill: int,
    opening: tuple[int, ...],
    base_time_s: float,
    solver_multiple: float,
) -> Outcome:
    """One game. ``first_skill`` moves first, which is the whole ballgame."""
    engines = {
        1: build_engine(first_skill, base_time_s, solver_multiple),
        2: build_engine(second_skill, base_time_s, solver_multiple),
    }
    skills = {1: first_skill, 2: second_skill}

    pos = Position.from_moves(opening)
    started = time.perf_counter()

    while not pos.is_draw():
        player = pos.current_player()
        column = engines[player].choose_move(pos, skill=skills[player])
        pos.play(column)
        if pos.has_won():
            return Outcome(
                first=first_skill,
                second=second_skill,
                opening=opening,
                winner=player,
                plies=pos.moves,
                seconds=time.perf_counter() - started,
            )

    return Outcome(
        first=first_skill,
        second=second_skill,
        opening=opening,
        winner=0,
        plies=pos.moves,
        seconds=time.perf_counter() - started,
    )


def run(report: Report, skills, verbose: bool = True) -> Report:
    pairs = [(a, b) for a, b in itertools.permutations(skills, 2)]
    total = len(pairs) * len(report.openings)
    done = 0

    for first, second in pairs:
        for opening in report.openings:
            outcome = play_game(
                first, second, opening, report.base_time_s, report.solver_multiple
            )
            report.outcomes.append(outcome)
            done += 1
            if verbose:
                verdict = (
                    "draw"
                    if outcome.winner == 0
                    else f"{first if outcome.winner == 1 else second} wins"
                )
                print(
                    f"  [{done:>4}/{total}] {first} (first) vs {second}: "
                    f"{verdict:>8} in {outcome.plies:>2} plies, {outcome.seconds:5.1f}s",
                    flush=True,
                )
    return report


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def head_to_head(report: Report) -> dict[tuple[int, int], Record]:
    """``table[(a, b)]`` is a's record against b, across both seats."""
    table: dict[tuple[int, int], Record] = {}
    for outcome in report.outcomes:
        a, b = outcome.first, outcome.second
        for me, them, i_moved_first in ((a, b, True), (b, a, False)):
            record = table.setdefault((me, them), Record())
            if outcome.winner == 0:
                record.draws += 1
            elif (outcome.winner == 1) == i_moved_first:
                record.wins += 1
            else:
                record.losses += 1
    return table


def by_seat(report: Report) -> dict[tuple[int, int], tuple[Record, Record]]:
    """``table[(a, b)]`` is (a's record moving first, a's record moving second)."""
    table: dict[tuple[int, int], tuple[Record, Record]] = {}
    for outcome in report.outcomes:
        a, b = outcome.first, outcome.second
        first_rec, _ = table.setdefault((a, b), (Record(), Record()))
        if outcome.winner == 0:
            first_rec.draws += 1
        elif outcome.winner == 1:
            first_rec.wins += 1
        else:
            first_rec.losses += 1

        _, other_second = table.setdefault((b, a), (Record(), Record()))
        if outcome.winner == 0:
            other_second.draws += 1
        elif outcome.winner == 2:
            other_second.wins += 1
        else:
            other_second.losses += 1
    return table


def format_matrix(report: Report, skills) -> str:
    table = head_to_head(report)
    width = 9
    lines = ["Head to head (win-draw-loss for the row, both seats)", ""]
    header = "      " + "".join(f"{('L' + str(s)):>{width}}" for s in skills) + "     points"
    lines.append(header)
    lines.append("      " + "-" * (width * len(skills) + 11))

    for a in skills:
        cells = []
        total = Record()
        for b in skills:
            if a == b:
                cells.append(f"{'--':>{width}}")
                continue
            record = table.get((a, b), Record())
            cells.append(f"{record!s:>{width}}")
            total.wins += record.wins
            total.draws += record.draws
            total.losses += record.losses
        share = f"{total.points:5.1f}/{total.played}"
        lines.append(f"  L{a} |" + "".join(cells) + f"   {share:>10}")
    return "\n".join(lines)


def format_seats(report: Report, skills) -> str:
    table = by_seat(report)
    lines = ["", "The seat, isolated (Connect 4 is a first-player win with perfect play)", ""]
    moving_first = Record()
    for outcome in report.outcomes:
        if outcome.winner == 0:
            moving_first.draws += 1
        elif outcome.winner == 1:
            moving_first.wins += 1
        else:
            moving_first.losses += 1
    lines.append(
        f"  Across every game, the side that moved first went {moving_first} "
        f"({moving_first.points / max(1, moving_first.played):.0%} of the points)."
    )

    lines.append("")
    lines.append("  Same pairing, both seats:")
    for a, b in itertools.combinations(skills, 2):
        first_rec, _ = table.get((a, b), (Record(), Record()))
        other_first, _ = table.get((b, a), (Record(), Record()))
        lines.append(
            f"    L{a} first vs L{b}: {first_rec!s:>9}    "
            f"L{b} first vs L{a}: {other_first!s:>9}"
        )
    return "\n".join(lines)


def format_answer(report: Report, low: int, high: int) -> str:
    table = head_to_head(report)
    seats = by_seat(report)
    record = table.get((low, high), Record())
    low_first, _ = seats.get((low, high), (Record(), Record()))
    high_first, _ = seats.get((high, low), (Record(), Record()))
    return "\n".join(
        [
            "",
            f"Level {low} against level {high}",
            "",
            (
                f"  Overall            {record}  "
                f"({record.wins} wins in {record.played} games)"
            ),
            f"  L{low} moving first    {low_first}",
            f"  L{high} moving first    {high_first}",
        ]
    )


def elo(report: Report, skills, prior: float = 1.0, rounds: int = 500) -> dict[int, float]:
    """Fit a Bradley-Terry rating to the results and express it in Elo points.

    The points column is ordinal. It ranks the levels correctly and says
    nothing about the *size* of the steps -- whether 4 to 5 is the same jump as
    5 to 6 is exactly the question it cannot answer, because a level that wins
    every game caps out at the same score no matter how far ahead it is.

    Bradley-Terry answers it: fit each level a strength such that the predicted
    win probabilities best match what happened, then map to the Elo scale. It
    is fitted by minorization-maximization, which for this model is a handful
    of lines and converges monotonically -- no gradient, no step size.

    The catch is that an undefeated level has no finite maximum: nothing in the
    data bounds it from above. Level 6 in a small run is exactly that. So every
    pair is credited with ``prior`` virtual draws against each other, which
    bounds every rating and shrinks the extremes toward the field. It is a
    thumb on the scale, and it is deliberate: an honest 700 with a stated prior
    beats an infinity.
    """
    table = head_to_head(report)
    strength = dict.fromkeys(skills, 1.0)

    for _ in range(rounds):
        updated = {}
        for a in skills:
            wins = expected = 0.0
            for b in skills:
                if a == b:
                    continue
                record = table.get((a, b), Record())
                # A draw is half a win to each side, the same convention the
                # points column uses, plus the virtual pair.
                games = record.played + 2 * prior
                if not games:
                    continue
                wins += record.wins + 0.5 * record.draws + prior
                expected += games / (strength[a] + strength[b])
            updated[a] = wins / expected if expected else strength[a]
        strength = updated

    # Elo is a log scale with 400 points per factor of ten in odds. The anchor
    # is arbitrary; the weakest level gets 0 so every number reads as "how far
    # above the floor".
    scale = 400.0 / math.log10(10.0)
    raw = {s: scale * math.log10(v) for s, v in strength.items()}
    floor = min(raw.values())
    return {s: v - floor for s, v in raw.items()}


def format_elo(report: Report, skills) -> str:
    ratings = elo(report, skills)
    order = sorted(skills, key=lambda s: ratings[s], reverse=True)
    lines = [
        "",
        "Fitted strength (Bradley-Terry, Elo scale, weakest level anchored at 0)",
        "",
        f"  {'level':<8}{'elo':>7}{'step':>8}   expected score vs the level below",
        "  " + "-" * 62,
    ]
    for i, skill in enumerate(order):
        below = order[i + 1] if i + 1 < len(order) else None
        if below is None:
            lines.append(f"  L{skill:<7}{ratings[skill]:>7.0f}{'--':>8}")
            continue
        step = ratings[skill] - ratings[below]
        share = 1.0 / (1.0 + 10.0 ** (-step / 400.0))
        lines.append(f"  L{skill:<7}{ratings[skill]:>7.0f}{step:>8.0f}   {share:>29.0%}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--time",
        type=float,
        default=0.05,
        dest="base_time",
        help="seconds per move for levels 0-5 (the app ships 1.0)",
    )
    parser.add_argument(
        "--solver-multiple",
        type=float,
        default=12.0,
        help="level 6's clock as a multiple of --time (the app ships 12.0)",
    )
    parser.add_argument("--openings", type=int, default=4, help="openings per pairing")
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None, help="write the raw outcomes here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    book = make_openings(args.openings, args.opening_plies, args.seed)
    report = Report(
        base_time_s=args.base_time,
        solver_multiple=args.solver_multiple,
        openings=book,
    )

    pairings = len(SKILLS) * (len(SKILLS) - 1)
    print(
        f"{pairings} ordered pairings x {len(book)} openings = "
        f"{pairings * len(book)} games\n"
        f"levels 0-5: {args.base_time:g}s/move   "
        f"level {SOLVER_SKILL}: {args.base_time * args.solver_multiple:g}s/move\n",
        flush=True,
    )

    started = time.perf_counter()
    run(report, SKILLS, verbose=not args.quiet)
    elapsed = time.perf_counter() - started

    print()
    print(format_matrix(report, SKILLS))
    print(format_elo(report, SKILLS))
    print(format_seats(report, SKILLS))
    print(format_answer(report, SOLVER_SKILL - 1, SOLVER_SKILL))
    print(f"\n{len(report.outcomes)} games in {elapsed:.1f}s")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "base_time_s": report.base_time_s,
                    "solver_multiple": report.solver_multiple,
                    "openings": [list(o) for o in report.openings],
                    "outcomes": [
                        {
                            "first": o.first,
                            "second": o.second,
                            "opening": list(o.opening),
                            "winner": o.winner,
                            "plies": o.plies,
                            "seconds": round(o.seconds, 3),
                        }
                        for o in report.outcomes
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"raw outcomes -> {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
