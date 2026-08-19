"""Negamax search with alpha-beta pruning over the bitboard position.

Design note: exact vs. heuristic
--------------------------------
Connect 4 is solved -- perfect play from the empty board is a first-player win
on move 41. But actually *proving* that takes billions of nodes; Pascal Pons'
heavily optimised C solver needs minutes. CPython is roughly two orders of
magnitude slower, so a full solve from an empty board is not something a web UI
can wait for.

So this engine runs in two regimes, and it is explicit about which one produced
any given answer:

*Exact.*      When the search reaches a real terminal position (a win, or a full
              board) on every line, the score it returns is ground truth: "I win
              in 7 plies" is a proof, not an estimate. Endgames and forced
              sequences resolve exactly, quickly.

*Heuristic.*  When the depth limit is hit first, the leaf is scored by an
              evaluator instead. That evaluator is the pluggable seam of this
              project -- it can be the hand-written threat heuristic, or the
              neural network trained on the UCI dataset. Same search, two
              different brains, directly comparable.

Every score carries a flag saying which regime produced it, and the UI shows
that flag. Presenting a guess as a proof would be the real failure here.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .bitboard import (
    HEIGHT,
    MOVE_ORDER,
    WIDTH,
    Position,
    popcount,
)

# Terminal scores sit far above anything an evaluator can return, so a proven
# win always outranks a merely promising-looking position. The board holds 42
# stones, so subtracting the ply count can never cross into heuristic range.
WIN_SCORE = 10_000
MAX_PLIES = WIDTH * HEIGHT

# Transposition table entry kinds. Alpha-beta does not always compute a node's
# true score -- when a cutoff happens it only learns a bound. Storing which of
# the three we have is what makes table reuse sound rather than subtly wrong.
EXACT, LOWER_BOUND, UPPER_BOUND = 0, 1, 2

Evaluator = Callable[[Position], float]


def heuristic_evaluator(pos: Position) -> float:
    """Hand-written leaf evaluation, from the point of view of the side to move.

    Deliberately simple, because its job is to be a *baseline* that the neural
    evaluator gets measured against. Two signals, both cheap:

    1. Immediate threats -- empty cells where a player would complete a four.
       Odd/even threat theory is the real depth here, but plain threat count
       already plays a respectable game.
    2. Centre control -- a stone in column 3 belongs to more possible fours
       (13) than one in column 0 (3), so centre occupancy is a decent proxy
       for future mobility.

    Returns a value in roughly [-1, 1]; the search treats anything in that band
    as "not proven, just an opinion".
    """
    mover_threats = popcount(pos.winning_spots())
    opponent_threats = popcount(pos.opponent_winning_spots())

    centre_col = WIDTH // 2
    centre_bits = ((1 << HEIGHT) - 1) << (centre_col * (HEIGHT + 1))
    mover_centre = popcount(pos.position & centre_bits)
    opponent_centre = popcount((pos.position ^ pos.mask) & centre_bits)

    threat_term = (mover_threats - opponent_threats) * 0.18
    centre_term = (mover_centre - opponent_centre) * 0.06
    return max(-0.95, min(0.95, threat_term + centre_term))


@dataclass(slots=True)
class SearchStats:
    """Instrumentation. The UI renders these, which is half the point."""

    nodes: int = 0
    depth_reached: int = 0
    table_hits: int = 0
    elapsed_ms: float = 0.0
    exact: bool = False
    aborted: bool = False


@dataclass(slots=True)
class MoveEvaluation:
    """The engine's verdict on one candidate column."""

    column: int
    score: float
    exact: bool
    # Plies until the game ends under best play, when the score is a proof.
    mate_in: int | None = None

    def label(self) -> str:
        """Short human-readable verdict, for the analysis panel."""
        if not self.exact:
            return f"{self.score:+.2f}"
        if self.mate_in is None or self.score == 0:
            return "draw"
        who = "win" if self.score > 0 else "loss"
        return f"{who} in {self.mate_in}"


@dataclass
class Analysis:
    """Everything the engine is willing to say about a position."""

    best_move: int
    evaluations: list[MoveEvaluation]
    principal_variation: list[int] = field(default_factory=list)
    stats: SearchStats = field(default_factory=SearchStats)


class SearchAborted(Exception):
    """Raised internally when the time budget runs out mid-search."""


class Engine:
    """Alpha-beta searcher, parameterised by its leaf evaluator.

    Swapping ``evaluator`` is the only difference between the "solver brain"
    and the "dataset brain". Everything else -- ordering, pruning, the table --
    is shared, so any difference in play is attributable to the evaluator
    alone. That is what makes the side-by-side comparison meaningful.
    """

    def __init__(
        self,
        evaluator: Evaluator | None = None,
        max_depth: int = MAX_PLIES,
        time_limit_s: float = 2.0,
        persist_table: bool = False,
        table_capacity: int = 1_200_000,
    ) -> None:
        # The default cap is the length of the whole game, which means the
        # *clock* is the real governor and the cap only exists to bound the
        # recursion. An arbitrary cap like 12 looks harmless and is not: past
        # the midgame the tree narrows enough that depth 12 finishes in ~25ms
        # of a 2000ms budget, so the engine would sit on 99% of its thinking
        # time and play a blind endgame -- precisely the phase where a human
        # sets up the double threat that wins the game. Uncapped, the same
        # position reaches depth 27.
        #
        # ``persist_table`` keeps the transposition table between calls to
        # ``analyse``. Entries are keyed by position, not by search, so reuse is
        # sound -- and in a real game the next search re-visits an enormous
        # fraction of the previous one's tree, two plies deeper. It is the
        # single cheapest way to make the solver mode actually reach proofs.
        # The cost is memory, so the table is capped and dropped wholesale when
        # it overflows (simpler than an eviction policy, and a fresh table is
        # correct, merely slower).
        self.evaluator = evaluator or heuristic_evaluator
        self.max_depth = max_depth
        self.time_limit_s = time_limit_s
        self.persist_table = persist_table
        self.table_capacity = table_capacity
        self._table: dict[int, tuple[int, float, int, bool]] = {}
        self._stats = SearchStats()
        self._deadline = 0.0

    # ------------------------------------------------------------------ public

    def analyse(self, pos: Position) -> Analysis:
        """Score every legal move, best first.

        Each child is searched with a *full* window rather than the usual
        null-window re-search. That is slower, but it yields a real score for
        every column instead of just "worse than the best one" -- and showing
        every column is the entire premise of the glass-box UI.
        """
        start = time.perf_counter()
        if not self.persist_table or len(self._table) > self.table_capacity:
            self._table.clear()
        self._stats = SearchStats()
        self._deadline = start + self.time_limit_s

        legal = pos.legal_moves()
        if not legal:
            return Analysis(best_move=-1, evaluations=[], stats=self._stats)

        # Iterative deepening: solve shallow, then deeper, reusing the table.
        # Cheap insurance -- if time runs out we still hold a complete,
        # coherent answer from the last finished depth.
        best: list[MoveEvaluation] = []
        for depth in range(1, self.max_depth + 1):
            try:
                current = self._analyse_at_depth(pos, depth)
            except SearchAborted:
                self._stats.aborted = True
                break
            best = current
            self._stats.depth_reached = depth
            # Every line ended in a proven result, so deeper search cannot
            # change anything. Stop.
            if all(e.exact for e in best):
                self._stats.exact = True
                break
            if time.perf_counter() >= self._deadline:
                break

        best.sort(key=lambda e: e.score, reverse=True)
        best_move = best[0].column if best else legal[0]
        self._stats.elapsed_ms = (time.perf_counter() - start) * 1000.0

        return Analysis(
            best_move=best_move,
            evaluations=best,
            principal_variation=self._extract_pv(pos, best_move),
            stats=self._stats,
        )

    def choose_move(self, pos: Position, skill: int = 5) -> int:
        """Pick a move at a given strength, 0 (weakest) to 5 (full strength).

        Weakening is done by playing the *n-th best* move according to the same
        honest evaluation, never by injecting randomness. A bot that plays
        well and then randomly throws a piece away feels broken; a bot that
        consistently picks the third-best move feels like a weaker opponent.

        One hard floor regardless of skill: it always takes an immediate win
        and always blocks an immediate loss. Missing a four-in-a-row that is
        sitting on the board reads as a bug, not as "easy mode".
        """
        analysis = self.analyse(pos)
        if not analysis.evaluations:
            return -1

        ranked = analysis.evaluations
        top = ranked[0]

        # Hard floor 1: an immediate win on the board is always taken.
        if skill >= 5 or (top.exact and top.score > 0 and (top.mate_in or 99) <= 1):
            return top.column

        # Hard floor 2: never allow the opponent a four-in-a-row next ply while
        # a move that prevents it exists. `mate_in <= 2` is exactly "I move,
        # they win" -- the blunder a human instantly notices.
        safe = [e for e in ranked if not (e.exact and e.score < 0 and (e.mate_in or 99) <= 2)]
        pool = safe or ranked

        slack = 5 - skill  # skill 4 -> 2nd best, skill 0 -> 6th best
        index = min(slack, len(pool) - 1)
        return pool[index].column

    # ----------------------------------------------------------------- internal

    def _analyse_at_depth(self, pos: Position, depth: int) -> list[MoveEvaluation]:
        results: list[MoveEvaluation] = []
        for col in pos.legal_moves():
            child = pos.played(col)

            if child.has_won():
                # We just won. mate_in 1 = "this very move ends it".
                results.append(MoveEvaluation(col, WIN_SCORE - child.moves, True, 1))
                continue
            if child.is_draw():
                results.append(MoveEvaluation(col, 0.0, True, 0))
                continue

            # Negamax: the child's score is from the opponent's point of view,
            # so ours is its negation. Same for the window bounds.
            score = -self._negamax(child, depth - 1, -WIN_SCORE, WIN_SCORE)

            # A score in terminal range is a proof; anything an evaluator could
            # have produced lives in [-1, 1] and is only an opinion. A plain
            # zero is deliberately *not* claimed as a proven draw -- it is the
            # value a depth-limited search returns when it simply ran out of
            # depth, and overclaiming it would be the one dishonest thing this
            # engine could do.
            exact = abs(score) > WIN_SCORE / 2
            mate_in = int(WIN_SCORE - abs(score)) - pos.moves if exact else None
            results.append(MoveEvaluation(col, score, exact, mate_in))
        return results

    def _negamax(self, pos: Position, depth: int, alpha: float, beta: float) -> float:
        """Core recursion. Returns the score from ``pos``'s mover's point of view."""
        self._stats.nodes += 1

        # Checking the clock on every node would cost more than it saves.
        if self._stats.nodes % 2048 == 0 and time.perf_counter() >= self._deadline:
            raise SearchAborted

        if pos.is_draw():
            return 0.0

        # If we can win right now, no need to look further -- nothing can beat
        # an immediate win, and this prunes an enormous number of nodes.
        winning = pos.winning_spots() & pos.possible_moves()
        if winning:
            return WIN_SCORE - (pos.moves + 1)

        # Every reply loses. Detecting it here rather than one ply deeper is
        # what turns the threat analysis into real pruning.
        playable = pos.non_losing_moves()
        if playable == 0:
            return -(WIN_SCORE - (pos.moves + 2))

        if depth <= 0:
            return self.evaluator(pos)

        alpha_original = alpha
        key = pos.key()

        cached = self._table.get(key)
        if cached is not None:
            cached_depth, cached_score, kind, _ = cached
            if cached_depth >= depth:
                self._stats.table_hits += 1
                # A bound is only usable if it actually resolves the window.
                if kind == EXACT:
                    return cached_score
                if kind == LOWER_BOUND and cached_score >= beta:
                    return cached_score
                if kind == UPPER_BOUND and cached_score <= alpha:
                    return cached_score

        best_score = -float("inf")
        best_col = -1

        for col in self._ordered_moves(pos, playable, cached):
            score = -self._negamax(pos.played(col), depth - 1, -beta, -alpha)
            if score > best_score:
                best_score, best_col = score, col
            if best_score > alpha:
                alpha = best_score
            if alpha >= beta:
                # The opponent would never allow this line, so its exact value
                # is irrelevant -- stop searching siblings.
                break

        if best_score <= alpha_original:
            kind = UPPER_BOUND
        elif best_score >= beta:
            kind = LOWER_BOUND
        else:
            kind = EXACT
        self._table[key] = (depth, best_score, kind, best_col)

        return best_score

    def _ordered_moves(self, pos: Position, playable: int, cached) -> list[int]:
        """Order candidate moves so the best is tried first.

        Alpha-beta's saving depends almost entirely on ordering: with perfect
        ordering it examines the square root of the nodes that plain minimax
        would. Two cheap signals, in priority order:

        1. The best move found for this position at a shallower depth (from the
           transposition table). Shallow search is a good predictor of deep
           search, which is what makes iterative deepening pay for itself.
        2. Otherwise centre-out, plus a bonus for moves that create threats.
        """
        columns = [c for c in MOVE_ORDER if playable & pos._landing_bit(c)]

        table_move = cached[3] if cached else -1
        opponent_wins = pos.opponent_winning_spots()

        def priority(col: int) -> tuple[int, int]:
            if col == table_move:
                return (-1, 0)
            child = pos.played(col)
            # More threats created is better; giving the opponent threats is worse.
            created = popcount(child.opponent_winning_spots() & ~opponent_wins)
            return (0, -created)

        columns.sort(key=priority)
        return columns

    def _extract_pv(self, pos: Position, first_move: int, limit: int = 8) -> list[int]:
        """Walk the transposition table to recover the expected line of play.

        This is what the UI draws as ghost pieces: "I play here, you play there,
        I play here". It is a best-effort reconstruction -- table entries can be
        overwritten -- so it stops at the first gap rather than guessing.
        """
        if first_move < 0:
            return []

        line = [first_move]
        cursor = pos.played(first_move)

        for _ in range(limit - 1):
            if cursor.has_won() or cursor.is_draw():
                break
            entry = self._table.get(cursor.key())
            if entry is None or entry[3] < 0 or not cursor.can_play(entry[3]):
                break
            line.append(entry[3])
            cursor = cursor.played(entry[3])

        return line
