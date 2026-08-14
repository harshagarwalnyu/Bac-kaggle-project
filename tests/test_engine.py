"""Tests for the search engine.

The engine has two properties worth testing, and they need different tools.

*Tactics* -- "does it take the win, does it block the loss" -- are testable with
hand-built positions where only one move is defensible.

*Correctness of the search itself* is harder. Alpha-beta, a transposition table
and threat-based pruning are each an opportunity to return a wrong score in a
way that no single hand-written position would catch. So again: a reference
implementation. ``brute_force_solve`` is plain negamax with no pruning, no
table, no ordering -- slow but transparently correct. On positions with few
empty cells it runs fast enough to check thousands of nodes, and the optimised
engine must agree with it on every one.
"""

from __future__ import annotations

import random

import pytest

from connect4.bitboard import WIDTH, Position
from connect4.engine import WIN_SCORE, Engine, heuristic_evaluator

# --------------------------------------------------------------------------
# Reference solver -- no pruning, no table, no cleverness.
# --------------------------------------------------------------------------


def brute_force_solve(pos: Position) -> float:
    """Exact score for ``pos`` from the mover's point of view.

    Scoring convention matches the engine: a win at ply ``n`` scores
    ``WIN_SCORE - n``, so faster wins score higher and slower losses score
    less badly. A drawn game scores 0.
    """
    if pos.is_draw():
        return 0.0

    best = -float("inf")
    for col in range(WIDTH):
        if not pos.can_play(col):
            continue
        if pos.is_winning_move(col):
            score = WIN_SCORE - (pos.moves + 1)
        else:
            score = -brute_force_solve(pos.played(col))
        best = max(best, score)
    return best


def _random_position_with_empties(rng: random.Random, empties: int) -> Position | None:
    """A random non-terminal position with exactly ``empties`` free cells.

    Purely random play almost never survives 36 plies -- someone stumbles into
    a four long before the board fills, so the naive version discards well over
    90% of its attempts. Instead we prefer moves that do *not* win outright,
    which lets games run to the endgame reliably.

    This biases the sample toward positions where both players kept missing
    wins, which is fine for the purpose: we are testing that the search
    computes the correct value of whatever position it is handed, not that the
    position is one strong players would reach.
    """
    target_plies = WIDTH * 6 - empties
    pos = Position()
    for _ in range(target_plies):
        legal = [c for c in range(WIDTH) if pos.can_play(c)]
        if not legal:
            return None
        quiet = [c for c in legal if not pos.is_winning_move(c)]
        pos.play(rng.choice(quiet or legal))
        if pos.has_won():
            return None  # forced to end the game early; discard and retry
    return pos


# --------------------------------------------------------------------------
# Tactics
# --------------------------------------------------------------------------


def test_takes_an_immediate_vertical_win():
    pos = Position.from_moves([3, 0, 3, 1, 3])  # X has three in column 3
    # It is O's move; give the turn back to X by having O play elsewhere.
    pos.play(6)
    engine = Engine(max_depth=6, time_limit_s=2.0)
    assert engine.choose_move(pos) == 3


def test_takes_an_immediate_horizontal_win():
    # X on (1,0), (2,0), (3,0); O parked on top. X to move, wins at (4,0) or (0,0).
    pos = Position.from_moves([1, 1, 2, 2, 3, 3])
    assert pos.current_player() == 1
    engine = Engine(max_depth=6, time_limit_s=2.0)
    assert engine.choose_move(pos) in (0, 4)


def test_blocks_an_immediate_loss():
    # X threatens a vertical four in column 3; O must block there.
    pos = Position.from_moves([3, 0, 3, 1, 3, 2])
    pos.play(6)  # X plays elsewhere... actually give O the move with X threatening
    pos = Position.from_moves([3, 0, 3, 1, 3])  # X: c3 rows 0,1,2. O to move.
    assert pos.current_player() == 2
    engine = Engine(max_depth=8, time_limit_s=3.0)
    assert engine.choose_move(pos) == 3, "O failed to block a three-in-a-row"


def test_blocks_even_at_the_lowest_skill_level():
    """Difficulty must not mean "misses a four sitting on the board"."""
    pos = Position.from_moves([3, 0, 3, 1, 3])
    engine = Engine(max_depth=8, time_limit_s=3.0)
    for skill in range(6):
        assert engine.choose_move(pos, skill=skill) == 3, f"skill {skill} ignored the threat"


def test_prefers_the_faster_of_two_wins():
    """Given a mate in 1 and a slower win, take the mate in 1."""
    pos = Position.from_moves([1, 1, 2, 2, 3, 3])  # X to move, immediate win available
    engine = Engine(max_depth=8, time_limit_s=3.0)
    analysis = engine.analyse(pos)
    best = analysis.evaluations[0]
    assert best.exact and best.score > 0
    assert best.mate_in == 1


# --------------------------------------------------------------------------
# Differential correctness against the brute-force reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("empties", [4, 5, 6, 7])
def test_exact_scores_match_brute_force(empties):
    """On near-full boards the engine must reproduce the true game value.

    Depth is set above the number of empty cells so the search always bottoms
    out in terminal nodes -- no evaluator involved, every score a proof.
    """
    rng = random.Random(4242 + empties)
    engine = Engine(max_depth=empties + 2, time_limit_s=20.0)

    checked = 0
    attempts = 0
    while checked < 12 and attempts < 400:
        attempts += 1
        pos = _random_position_with_empties(rng, empties)
        if pos is None or pos.is_draw():
            continue

        expected = brute_force_solve(pos)
        analysis = engine.analyse(pos)
        assert analysis.evaluations, f"no moves returned for\n{pos}"
        actual = analysis.evaluations[0].score

        assert actual == pytest.approx(expected), (
            f"engine says {actual}, brute force says {expected}, for\n{pos}"
        )
        checked += 1

    assert checked >= 12, f"only generated {checked} usable positions"


@pytest.mark.parametrize("empties", [5, 6])
def test_best_move_is_actually_optimal(empties):
    """The chosen column must achieve the position's true value.

    Weaker than "matches the reference's choice" on purpose: several moves can
    share the optimal score, and picking a different one of them is fine.
    """
    rng = random.Random(9000 + empties)
    engine = Engine(max_depth=empties + 2, time_limit_s=20.0)

    checked = 0
    attempts = 0
    while checked < 10 and attempts < 400:
        attempts += 1
        pos = _random_position_with_empties(rng, empties)
        if pos is None or pos.is_draw():
            continue

        true_value = brute_force_solve(pos)
        chosen = engine.choose_move(pos)
        assert chosen >= 0 and pos.can_play(chosen)

        if pos.is_winning_move(chosen):
            achieved = WIN_SCORE - (pos.moves + 1)
        else:
            achieved = -brute_force_solve(pos.played(chosen))

        assert achieved == pytest.approx(true_value), (
            f"chose column {chosen} worth {achieved}, but {true_value} was available:\n{pos}"
        )
        checked += 1

    assert checked >= 10


# --------------------------------------------------------------------------
# Search bookkeeping and the analysis payload
# --------------------------------------------------------------------------


def test_analyse_scores_every_legal_column():
    pos = Position.from_moves([3, 3, 4])
    analysis = Engine(max_depth=5, time_limit_s=3.0).analyse(pos)
    assert {e.column for e in analysis.evaluations} == set(pos.legal_moves())


def test_analyse_returns_evaluations_sorted_best_first():
    pos = Position.from_moves([3, 3, 4, 4])
    analysis = Engine(max_depth=6, time_limit_s=3.0).analyse(pos)
    scores = [e.score for e in analysis.evaluations]
    assert scores == sorted(scores, reverse=True)
    assert analysis.best_move == analysis.evaluations[0].column


def test_full_column_is_never_offered():
    pos = Position.from_moves([0, 0, 0, 0, 0, 0])  # column 0 is now full
    analysis = Engine(max_depth=4, time_limit_s=3.0).analyse(pos)
    assert 0 not in {e.column for e in analysis.evaluations}


def test_principal_variation_is_a_legal_sequence():
    """Every move in the PV must be playable when it is reached."""
    pos = Position.from_moves([3, 3, 4])
    analysis = Engine(max_depth=7, time_limit_s=3.0).analyse(pos)
    assert analysis.principal_variation
    assert analysis.principal_variation[0] == analysis.best_move

    cursor = pos.copy()
    for col in analysis.principal_variation:
        assert cursor.can_play(col), f"PV contains illegal move {col}"
        cursor.play(col)


def test_stats_are_populated():
    analysis = Engine(max_depth=6, time_limit_s=3.0).analyse(Position.from_moves([3, 3]))
    assert analysis.stats.nodes > 0
    assert analysis.stats.depth_reached > 0
    assert analysis.stats.elapsed_ms >= 0.0


def test_time_limit_is_respected():
    """A tight budget on a wide-open position must still return promptly."""
    engine = Engine(max_depth=42, time_limit_s=0.5)
    analysis = engine.analyse(Position())
    assert analysis.stats.elapsed_ms < 2500, "search blew through its time budget"
    assert analysis.best_move in range(WIDTH)


def test_a_per_call_budget_overrides_the_engine_default():
    """The advisory search borrows the solver's warm table, not its clock."""
    engine = Engine(max_depth=42, time_limit_s=30.0)
    analysis = engine.analyse(Position(), time_limit_s=0.3)
    assert analysis.stats.elapsed_ms < 2500, "the per-call budget was ignored"


def test_an_aborted_search_names_a_move_without_inventing_a_variation():
    """A budget too small to finish depth 1 leaves nothing scored.

    Callers still need something playable, but a principal variation through a
    position that was never searched is invention -- and the UI would draw it
    with exactly the same confidence as a real one.
    """
    engine = Engine(max_depth=42, time_limit_s=0.0)
    analysis = engine.analyse(Position())

    assert analysis.best_move in range(WIDTH)
    if not analysis.evaluations:
        assert analysis.principal_variation == []


def test_concurrent_searches_do_not_corrupt_each_other():
    """FastAPI runs synchronous handlers in a threadpool and the app shares one
    engine, so two requests really do land in ``analyse`` at once. Without the
    lock, one search resets the other's deadline and stats mid-flight."""
    import threading

    engine = Engine(max_depth=8, time_limit_s=0.4)
    results: list = []
    errors: list = []

    def search(moves):
        try:
            results.append(engine.analyse(Position.from_moves(moves)))
        except Exception as error:  # pragma: no cover - the failure being tested
            errors.append(error)

    threads = [
        threading.Thread(target=search, args=(moves,))
        for moves in ([3], [3, 3], [2, 4], [3, 2, 4], [0], [1])
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"concurrent searches raised: {errors}"
    assert len(results) == len(threads)
    for analysis in results:
        assert analysis.best_move in range(WIDTH)
        assert analysis.stats.nodes > 0, "a search was robbed of its own statistics"


def test_no_legal_moves_on_a_full_board():
    pos = Position()
    order = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    for _ in range(3):
        for col in order:
            pos.play(col)
    analysis = Engine(max_depth=4, time_limit_s=1.0).analyse(pos)
    assert analysis.best_move == -1
    assert analysis.evaluations == []


# --------------------------------------------------------------------------
# Evaluator seam
# --------------------------------------------------------------------------


def test_heuristic_evaluator_stays_inside_the_opinion_band():
    """Evaluator output must never stray into proven-score territory.

    If it did, a guess would outrank a proof and the ``exact`` flag would start
    lying -- which is the one thing the whole design promises not to do.
    """
    rng = random.Random(7)
    for _ in range(500):
        pos = Position()
        for _ in range(rng.randint(0, 30)):
            legal = [c for c in range(WIDTH) if pos.can_play(c)]
            if not legal:
                break
            pos.play(rng.choice(legal))
            if pos.has_won():
                break
        value = heuristic_evaluator(pos)
        assert -1.0 <= value <= 1.0, f"evaluator returned {value}"


def test_evaluator_is_swappable():
    """The engine must accept any callable -- this is the dataset brain's socket."""
    calls: list[Position] = []

    def always_zero(pos: Position) -> float:
        calls.append(pos)
        return 0.0

    engine = Engine(evaluator=always_zero, max_depth=3, time_limit_s=2.0)
    engine.analyse(Position.from_moves([3, 3]))
    assert calls, "custom evaluator was never consulted"


def test_evaluator_sees_positions_from_the_movers_point_of_view():
    """A symmetric position must evaluate to zero for whoever is to move."""
    mirrored = Position.from_moves([0, 6, 1, 5])  # symmetric about the centre column
    assert heuristic_evaluator(mirrored) == pytest.approx(0.0)
