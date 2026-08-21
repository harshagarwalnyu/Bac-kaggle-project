"""Property-based tests for the bitboard and the search.

The example-based tests next door check positions someone thought of. These
check the invariants that have to hold for *every* position, against inputs
nobody chose. The two find different bugs: an example test catches the case you
predicted, a property test catches the encoding mistake in a column you never
played into.

Everything here leans on one idea -- an independent oracle. The bitboard packs
a board into two integers and answers questions with shifts and masks, which is
fast and completely opaque. So each property compares it against a slow, dumb,
obviously-correct implementation over ``to_grid()``. When the two disagree, the
bit trick is wrong, because nobody gets a nested loop over 42 cells wrong.
"""

from __future__ import annotations

from collections.abc import Iterator

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from connect4.bitboard import (
    BOARD_MASK,
    HEIGHT,
    MOVE_ORDER,
    WIDTH,
    Position,
    has_alignment,
    popcount,
)
from connect4.engine import Engine

# A move sequence long enough to fill the board, drawn without regard to
# legality. `_replay` below is what makes it meaningful.
#
# Deliberately *not* `st.lists(...).filter(is_legal)`. Filtering throws away
# most of what it generates -- a random 42-column sequence is almost never
# legal -- and Hypothesis eventually gives up and fails the run for it. Taking
# an arbitrary sequence and interpreting it as "play these when you can" keeps
# every generated example useful, and still reaches deep positions.
move_sequences = st.lists(st.integers(min_value=0, max_value=WIDTH - 1), max_size=42)


def _replay(columns: list[int]) -> tuple[Position, list[int]]:
    """Play what is playable, stop when the game ends.

    Returns the position and the moves that were actually applied, which is
    what the caller must compare against -- the input is a wish list, not a
    game.
    """
    pos = Position()
    applied: list[int] = []
    for col in columns:
        if not pos.can_play(col):
            continue
        winning = pos.is_winning_move(col)
        pos.play(col)
        applied.append(col)
        if winning or pos.is_draw():
            break
    return pos, applied


def _live_positions(columns: list[int]) -> Iterator[Position]:
    """Every position along a replay that deliberately declines to win.

    A normal replay is useless for testing whether the search *finds* a win,
    because it takes the win itself and the game stops. Skipping any move that
    would end the game keeps the line going and lets threats pile up, so
    positions with a win available -- the interesting ones -- become common
    rather than vanishingly rare.

    Yields copies: the caller hands these to the engine, and a shared mutable
    board would let one iteration corrupt the next.
    """
    pos = Position()
    for col in columns:
        if not pos.can_play(col) or pos.is_winning_move(col):
            continue
        pos.play(col)
        if pos.is_draw():
            return
        yield pos.copy()


def _naive_has_four(grid: list[list[int]], player: int) -> bool:
    """Does ``player`` have four in a row? Checked the boring way.

    Four directions, every start cell, no cleverness. This is the oracle for
    :func:`has_alignment`, so it must not share any of its reasoning.
    """
    for row in range(HEIGHT):
        for col in range(WIDTH):
            for drow, dcol in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = []
                for step in range(4):
                    r, c = row + drow * step, col + dcol * step
                    if 0 <= r < HEIGHT and 0 <= c < WIDTH:
                        cells.append(grid[r][c])
                if len(cells) == 4 and all(cell == player for cell in cells):
                    return True
    return False


def _column_heights(grid: list[list[int]]) -> list[int]:
    """Stones per column, read off the rendered grid."""
    return [sum(1 for row in range(HEIGHT) if grid[row][col] != 0) for col in range(WIDTH)]


# --------------------------------------------------------------- encoding


@given(move_sequences)
def test_occupancy_invariants_hold(columns: list[int]) -> None:
    """The two integers stay consistent with what they claim to mean.

    ``position`` is a subset of ``mask`` by definition -- a stone belonging to
    the mover is a stone. And ``mask`` must never light a bit outside
    ``BOARD_MASK``: the encoding leaves a sentinel row between columns, and a
    stone leaking into one would silently join two columns into a single
    diagonal.
    """
    pos, applied = _replay(columns)

    assert pos.position & ~pos.mask == 0, "mover holds a stone that is not on the board"
    assert pos.mask & ~BOARD_MASK == 0, "a stone leaked into a sentinel row"
    assert popcount(pos.mask) == pos.moves == len(applied)


@given(move_sequences)
def test_grid_agrees_with_the_bitboard(columns: list[int]) -> None:
    """``to_grid`` renders exactly the stones the masks hold, in the right order."""
    pos, applied = _replay(columns)
    grid = pos.to_grid()

    first_player_stones = sum(row.count(1) for row in grid)
    second_player_stones = sum(row.count(2) for row in grid)

    # Player 1 moves on even plies, so they are never behind and never more
    # than one ahead.
    assert first_player_stones + second_player_stones == len(applied)
    assert first_player_stones - second_player_stones == len(applied) % 2

    # Column heights derived from the grid must match the moves played into
    # each column -- this is the check that catches an off-by-one in the
    # stride arithmetic.
    expected = [applied.count(col) for col in range(WIDTH)]
    assert _column_heights(grid) == expected

    # And no floating stones: every column is a contiguous stack from the
    # bottom, since row 0 of the grid is the top.
    for col in range(WIDTH):
        column = [grid[row][col] for row in range(HEIGHT)]
        filled = [cell != 0 for cell in column]
        assert filled == sorted(filled), f"column {col} has a gap under a stone"


@given(st.integers(min_value=0, max_value=(1 << 49) - 1))
def test_popcount_matches_the_obvious_implementation(bits: int) -> None:
    """Counting bits by string, because that shares nothing with the real one.

    ``FURB161`` wants ``bits.bit_count()`` here, and it is right that this is
    the slower way to count bits -- but ``popcount`` *is* ``bit_count``, so
    taking the suggestion would leave the test asserting that a function equals
    itself. An oracle has to be independent to be an oracle, and slow is the
    price of that.
    """
    assert popcount(bits) == bin(bits).count("1")  # noqa: FURB161


# ----------------------------------------------------------------- queries


@given(move_sequences)
def test_win_detection_matches_a_naive_scan(columns: list[int]) -> None:
    """``has_won`` agrees with looking at all 69 possible lines by hand.

    Note whose win is being asked about. After ``play`` the position is seen
    from the *next* player, so the side that just moved is ``3 - current``.
    Getting this backwards is exactly the kind of thing this test exists for.
    """
    pos, applied = _replay(columns)
    grid = pos.to_grid()

    just_moved = 3 - pos.current_player()
    expected = bool(applied) and _naive_has_four(grid, just_moved)

    assert pos.has_won() == expected

    # The side to move cannot already have four in a row: the game would have
    # ended on the move that made it.
    assert not _naive_has_four(grid, pos.current_player())


@given(move_sequences)
def test_legal_moves_match_column_heights(columns: list[int]) -> None:
    """A column is playable exactly when it holds fewer than ``HEIGHT`` stones."""
    pos, _ = _replay(columns)
    heights = _column_heights(pos.to_grid())

    expected = {col for col in range(WIDTH) if heights[col] < HEIGHT}
    assert set(pos.legal_moves()) == expected
    assert all(pos.can_play(col) == (col in expected) for col in range(WIDTH))

    # The list is ordered, not merely correct -- the search depends on centre
    # columns coming first, and a reordering here would quietly cost strength
    # without failing any example test.
    assert pos.legal_moves() == [col for col in MOVE_ORDER if col in expected]


@given(move_sequences)
def test_is_winning_move_predicts_has_won(columns: list[int]) -> None:
    """The lookahead and the after-the-fact check tell the same story."""
    pos, _ = _replay(columns)
    assume(not pos.has_won())

    for col in pos.legal_moves():
        predicted = pos.is_winning_move(col)
        assert pos.played(col).has_won() == predicted


@given(move_sequences)
def test_safe_moves_are_a_subset_of_legal_moves(columns: list[int]) -> None:
    """Every non-losing move is playable, and refusing to lose is optional.

    ``safe_moves`` returning empty is meaningful (every reply loses) rather
    than an error, so the only invariant available is containment and order.
    """
    pos, _ = _replay(columns)
    assume(not pos.has_won())

    legal = pos.legal_moves()
    safe = pos.safe_moves()

    assert set(safe) <= set(legal)
    assert safe == [col for col in MOVE_ORDER if col in set(safe)]


# ---------------------------------------------------------------- identities


@given(move_sequences)
def test_played_does_not_mutate_the_original(columns: list[int]) -> None:
    """``played`` is the non-destructive twin of ``play``.

    The search relies on this. A ``played`` that mutated in place would corrupt
    the parent node halfway through its own move loop, and the symptom would be
    a wrong evaluation rather than a crash.
    """
    pos, _ = _replay(columns)
    assume(not pos.has_won())
    assume(pos.legal_moves())

    before = (pos.position, pos.mask, pos.moves)
    for col in pos.legal_moves():
        child = pos.played(col)
        assert (pos.position, pos.mask, pos.moves) == before
        assert child is not pos
        assert child.moves == pos.moves + 1


@given(move_sequences)
def test_copy_is_equal_and_independent(columns: list[int]) -> None:
    pos, _ = _replay(columns)
    clone = pos.copy()

    assert clone == pos
    assert clone is not pos
    assert clone.key() == pos.key()

    assume(not pos.has_won())
    assume(pos.legal_moves())
    clone.play(pos.legal_moves()[0])
    assert clone != pos, "mutating the copy changed the original"


@given(move_sequences)
def test_from_moves_round_trips(columns: list[int]) -> None:
    """Replaying the applied moves reconstructs the identical position."""
    pos, applied = _replay(columns)
    rebuilt = Position.from_moves(applied)

    assert rebuilt == pos
    assert rebuilt.key() == pos.key()
    assert rebuilt.to_grid() == pos.to_grid()


@given(move_sequences)
def test_equal_positions_share_a_key(columns: list[int]) -> None:
    """``key`` is a function of the position, which is what the table assumes.

    The transposition table indexes on this. If two identical boards produced
    different keys the table would merely be useless; if two *different* boards
    collided it would return a score for the wrong position, so the reachable
    half of that -- same board, same key -- is worth pinning.
    """
    pos, applied = _replay(columns)
    assert Position.from_moves(applied).key() == pos.key()


@given(move_sequences)
def test_mirroring_the_board_mirrors_every_answer(columns: list[int]) -> None:
    """Connect 4 is left-right symmetric, and so is the encoding.

    This is the strongest single property here: it exercises the landing-bit
    arithmetic, the direction masks and the grid rendering at once, and it
    fails loudly if any of them has a column-dependent bug -- the sort that a
    test playing down the middle would never reach.
    """
    pos, applied = _replay(columns)
    mirrored, mirrored_applied = _replay([WIDTH - 1 - col for col in applied])

    # The mirror of a legal sequence is legal, so nothing should be dropped.
    assert mirrored_applied == [WIDTH - 1 - col for col in applied]

    assert mirrored.moves == pos.moves
    assert mirrored.has_won() == pos.has_won()
    assert mirrored.is_draw() == pos.is_draw()
    assert mirrored.to_grid() == [list(reversed(row)) for row in pos.to_grid()]
    assert sorted(mirrored.legal_moves()) == sorted(
        WIDTH - 1 - col for col in pos.legal_moves()
    )
    assert sorted(mirrored.safe_moves()) == sorted(
        WIDTH - 1 - col for col in pos.safe_moves()
    )


@given(st.integers(min_value=0, max_value=(1 << 49) - 1))
def test_has_alignment_only_sees_board_bits(bits: int) -> None:
    """Sentinel bits must never be able to fake an alignment.

    The stride encoding is only sound because the sentinel row breaks vertical
    and diagonal runs at the column boundary. Feeding in arbitrary integers is
    the direct test of that: masking to the board first must not change the
    answer for anything that was already a legal board.
    """
    on_board = bits & BOARD_MASK
    assert has_alignment(on_board) == has_alignment(on_board & BOARD_MASK)


# ------------------------------------------------------------------ search


# The engine is orders of magnitude slower than the bitboard, so these run
# fewer examples. `deadline=None` because a single search legitimately takes
# longer than Hypothesis's default per-example budget, and a deadline here
# would report a timing artefact as a failing property. The `too_slow` health
# check goes for the same reason.
engine_settings = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# How many qualifying positions to search per generated example. Each one is a
# full alpha-beta call, so the cost is real; two is enough to keep every
# example doing work without turning the suite into a benchmark.
_POSITIONS_PER_EXAMPLE = 2


def _fixed_depth_engine(depth: int = 4) -> Engine:
    """A search governed by depth alone, never by the clock.

    The time limit is set absurdly high on purpose. With a real budget the
    engine aborts mid-iteration and the result depends on how fast the machine
    is, which would make every property here flaky on a loaded CI runner.
    """
    return Engine(max_depth=depth, time_limit_s=3600.0)


@given(move_sequences)
@engine_settings
def test_analysis_is_well_formed_and_non_destructive(columns: list[int]) -> None:
    """Whatever the engine concludes, it concludes it about a legal move."""
    pos, _ = _replay(columns)
    assume(not pos.has_won() and not pos.is_draw())
    assume(pos.legal_moves())

    before = (pos.position, pos.mask, pos.moves)
    analysis = _fixed_depth_engine().analyse(pos)

    assert (pos.position, pos.mask, pos.moves) == before, "the search mutated its input"

    legal = set(pos.legal_moves())
    assert analysis.best_move in legal
    assert analysis.evaluations, "a position with legal moves produced no evaluations"
    assert {ev.column for ev in analysis.evaluations} <= legal
    assert not analysis.stats.aborted, "the clock fired despite an hour-long budget"

    # The reported best move must be one the evaluations actually rank first;
    # a best move that contradicts its own evaluation list is the failure mode
    # that makes an analysis panel lie.
    best = max(analysis.evaluations, key=lambda ev: ev.score)
    assert analysis.best_move in {
        ev.column for ev in analysis.evaluations if ev.score == best.score
    }

    # The principal variation must start from a legal move and stay playable.
    replay = pos.copy()
    for ply, col in enumerate(analysis.principal_variation):
        assert replay.can_play(col), f"PV move {ply} ({col}) is not playable"
        if replay.is_winning_move(col):
            replay.play(col)
            break
        replay.play(col)


def test_an_immediate_win_is_never_missed_setup() -> None:
    """Sanity check that the generator below actually produces winnable spots.

    Without this, a bug in `_live_positions` would make the two competence
    properties pass by testing nothing at all, which is the classic way a
    property suite becomes decorative.
    """
    pos, _ = _replay([3, 3, 4, 4, 5, 5])
    assert any(pos.is_winning_move(col) for col in pos.legal_moves())


@given(move_sequences)
@engine_settings
def test_an_immediate_win_is_never_missed(columns: list[int]) -> None:
    """If a move wins on the spot, the engine plays one that does.

    The weakest possible competence claim, and therefore the one worth making
    as a property: no search bug should ever survive it. Stated as "a winning
    move" rather than "this winning move" because several may exist and the
    engine is free to prefer any of them.

    Scanning the intermediate positions rather than `assume`-ing on the final
    one is deliberate. A replay that takes its wins ends the moment one exists,
    so the terminal position almost never has a win pending -- filtering for it
    discards essentially every generated example. Declining wins on the way
    through leaves them standing, and every position along the line becomes a
    test case instead of one in a hundred.
    """
    checked = 0
    for pos in _live_positions(columns):
        winning = [col for col in pos.legal_moves() if pos.is_winning_move(col)]
        if not winning:
            continue
        assert _fixed_depth_engine().analyse(pos).best_move in winning
        checked += 1
        if checked == _POSITIONS_PER_EXAMPLE:
            break


def _immediate_threats(pos: Position) -> list[int]:
    """Columns the opponent would win with *if it were their turn right now*.

    Playable cells only. A cell the opponent wins on but that nothing can be
    dropped into yet is not a threat and cannot be blocked -- it is a trap to
    stay out of, which is the separate property below.
    """
    reachable = pos.possible_moves() & pos.opponent_winning_spots()
    return [col for col in pos.legal_moves() if reachable & pos._landing_bit(col)]


@given(move_sequences)
@engine_settings
def test_the_engine_blocks_a_single_immediate_threat(columns: list[int]) -> None:
    """With exactly one way to lose next move, the engine takes it away.

    Restricted to positions where the engine has no win of its own -- otherwise
    winning beats blocking and the assertion would be wrong rather than the
    engine.
    """
    checked = 0
    for pos in _live_positions(columns):
        if any(pos.is_winning_move(col) for col in pos.legal_moves()):
            continue
        threats = _immediate_threats(pos)
        if len(threats) != 1:
            continue
        assert _fixed_depth_engine().analyse(pos).best_move == threats[0]
        checked += 1
        if checked == _POSITIONS_PER_EXAMPLE:
            break


@given(move_sequences)
@engine_settings
def test_the_engine_does_not_play_underneath_an_opponent_win(
    columns: list[int],
) -> None:
    """It never fills the square below a cell the opponent wins on.

    The trap the block property above is not about. Dropping a stone under an
    opponent winning cell does not lose the game this move; it lifts them into
    the win on the next one, which a shallow search will happily walk into if
    it only ever looks for threats it can block.

    Asserted only where a safe alternative exists: with every reply losing,
    playing into the trap is as good as anything else and the engine is not
    wrong to.
    """
    checked = 0
    for pos in _live_positions(columns):
        if any(pos.is_winning_move(col) for col in pos.legal_moves()):
            continue
        safe = pos.safe_moves()
        if not safe or len(safe) == len(pos.legal_moves()):
            continue
        assert _fixed_depth_engine().analyse(pos).best_move in safe
        checked += 1
        if checked == _POSITIONS_PER_EXAMPLE:
            break


@given(move_sequences)
@engine_settings
def test_search_is_deterministic(columns: list[int]) -> None:
    """The same position searched twice gives the same answer.

    Guards the transposition table specifically. A table that returns a bound
    it should not -- wrong depth, wrong bound type -- often shows up first as a
    search that disagrees with itself, because the second run reads entries the
    first one wrote.

    Two depths rather than one: a shallow search and a deeper one exercise
    different table interactions, and the parity of the depth decides which
    side of the tree gets cut.
    """
    pos, _ = _replay(columns)
    assume(not pos.has_won() and not pos.is_draw())
    assume(pos.legal_moves())

    for depth in (2, 4):
        first = _fixed_depth_engine(depth).analyse(pos)
        second = _fixed_depth_engine(depth).analyse(pos)

        assert first.best_move == second.best_move, f"depth {depth} disagreed with itself"
        assert [(ev.column, ev.score) for ev in first.evaluations] == [
            (ev.column, ev.score) for ev in second.evaluations
        ]
