"""Tests for the bitboard representation.

The bit manipulation in ``bitboard.py`` is fast but not self-evidently correct
-- a wrong shift constant produces a board that works for a hundred positions
and then silently miscounts a diagonal. So the core strategy here is
*differential testing*: write a slow, obviously-correct reference
implementation using nested loops, then assert the fast version agrees with it
over many thousands of random positions. If the two disagree even once, the
clever version is wrong.
"""

from __future__ import annotations

import random

import pytest

from connect4.bitboard import (
    HEIGHT,
    MOVE_ORDER,
    WIDTH,
    Position,
    has_alignment,
    popcount,
)

# --------------------------------------------------------------------------
# Slow reference implementation -- deliberately dumb, easy to eyeball.
# --------------------------------------------------------------------------


def _grid_from_bits(bits: int) -> list[list[bool]]:
    """Unpack a bitboard into ``cells[col][row]``, row 0 at the bottom."""
    return [[bool(bits & (1 << (c * (HEIGHT + 1) + r))) for r in range(HEIGHT)] for c in range(WIDTH)]


def reference_has_alignment(bits: int) -> bool:
    """Four-in-a-row detection by brute force over every cell and direction."""
    cells = _grid_from_bits(bits)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # up, right, up-right, down-right
    for col in range(WIDTH):
        for row in range(HEIGHT):
            if not cells[col][row]:
                continue
            for dc, dr in directions:
                run = 0
                c, r = col, row
                while 0 <= c < WIDTH and 0 <= r < HEIGHT and cells[c][r]:
                    run += 1
                    if run >= 4:
                        return True
                    c += dc
                    r += dr
    return False


def reference_winning_columns(pos: Position) -> set[int]:
    """Columns where dropping a stone wins immediately, found by simulation."""
    winners = set()
    for col in range(WIDTH):
        if not pos.can_play(col):
            continue
        child = pos.played(col)
        # After play(), the stones of the player who just moved are position ^ mask.
        if reference_has_alignment(child.position ^ child.mask):
            winners.add(col)
    return winners


def _random_position(rng: random.Random, max_plies: int = 42) -> Position:
    """Play random legal moves, stopping at the first terminal position."""
    pos = Position()
    for _ in range(rng.randint(0, max_plies)):
        legal = [c for c in range(WIDTH) if pos.can_play(c)]
        if not legal:
            break
        pos.play(rng.choice(legal))
        if pos.has_won():
            break
    return pos


# --------------------------------------------------------------------------
# Win detection
# --------------------------------------------------------------------------


def _bits_for(cells: list[tuple[int, int]]) -> int:
    """Build a bitboard from a list of (col, row) pairs, row 0 at the bottom."""
    return sum(1 << (c * (HEIGHT + 1) + r) for c, r in cells)


def test_vertical_win_detected():
    assert has_alignment(_bits_for([(2, 0), (2, 1), (2, 2), (2, 3)]))


def test_horizontal_win_detected():
    assert has_alignment(_bits_for([(1, 0), (2, 0), (3, 0), (4, 0)]))


def test_diagonal_up_right_detected():
    assert has_alignment(_bits_for([(0, 0), (1, 1), (2, 2), (3, 3)]))


def test_diagonal_down_right_detected():
    assert has_alignment(_bits_for([(0, 3), (1, 2), (2, 1), (3, 0)]))


def test_three_in_a_row_is_not_a_win():
    assert not has_alignment(_bits_for([(2, 0), (2, 1), (2, 2)]))


def test_no_wraparound_between_columns():
    """The sentinel row's whole reason for existing.

    Column 0 rows 4 and 5 are bits 4 and 5; column 1 rows 0 and 1 are bits 7
    and 8. Without a sentinel, bits 4,5,6,7 would look like a vertical run.
    Bit 6 is the sentinel and can never be set, so this must not register.
    """
    assert not has_alignment(_bits_for([(0, 4), (0, 5), (1, 0), (1, 1)]))


def test_no_wraparound_horizontally_across_board_edge():
    """A run from the right edge must not continue onto the next row's left."""
    assert not has_alignment(_bits_for([(5, 0), (6, 0), (0, 1), (1, 1)]))


@pytest.mark.parametrize("seed", range(40))
def test_alignment_matches_reference_on_random_positions(seed):
    """Differential test: fast shift-based detection vs. brute-force scan."""
    rng = random.Random(seed)
    for _ in range(120):
        pos = _random_position(rng)
        for stones in (pos.position, pos.position ^ pos.mask):
            assert has_alignment(stones) == reference_has_alignment(stones), (
                f"disagreement on bits {stones:#x}\n{pos}"
            )


# --------------------------------------------------------------------------
# Threat detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_winning_moves_match_reference(seed):
    """``is_winning_move`` must agree with actually playing the move."""
    rng = random.Random(1000 + seed)
    for _ in range(80):
        pos = _random_position(rng)
        if pos.has_won() or pos.is_draw():
            continue
        fast = {c for c in range(WIDTH) if pos.can_play(c) and pos.is_winning_move(c)}
        assert fast == reference_winning_columns(pos), f"\n{pos}"


def test_winning_spots_includes_unreachable_cells():
    """Threats above the current stack still count -- they drive zugzwang.

    X has stones at column 0 rows 0,1,2. The winning cell is (0, 3), which is
    playable. Now stack column 1 so that X also threatens at (1, 3) -- reached
    only later. Both must be reported.
    """
    pos = Position()
    # X: c0r0, c0r1, c0r2 -- O filler in column 6 to pass the turn.
    for col in (0, 6, 0, 6, 0, 6):
        pos.play(col)
    # It is X's turn again; X threatens at column 0 row 3.
    spots = pos.winning_spots()
    assert spots & (1 << (0 * (HEIGHT + 1) + 3))


def test_opponent_winning_spots_is_the_mirror():
    """After a move, my threats become the opponent's, seen from the other side."""
    pos = Position()
    for col in (3, 0, 3, 0, 3):  # X has three in column 3, threatening row 3
        pos.play(col)
    # Turn is now O's. X's threat must show up as an *opponent* threat.
    assert pos.opponent_winning_spots() & (1 << (3 * (HEIGHT + 1) + 3))


def test_non_losing_moves_blocks_a_single_threat():
    """With exactly one opponent threat, the only legal choice is to block it."""
    pos = Position()
    for col in (3, 0, 3, 1, 3):  # X threatens column 3 row 3; it is O's move
        pos.play(col)
    allowed = pos.non_losing_moves()
    block_bit = 1 << (3 * (HEIGHT + 1) + 3)
    assert allowed == block_bit, "O must be forced to block column 3"


def test_non_losing_moves_is_empty_when_two_threats_exist():
    """Two separate immediate threats cannot both be blocked -- position is lost."""
    pos = Position()
    # X builds a horizontal three with open ends at columns 1 and 5:
    #   X at c2r0, c3r0, c4r0; O parked harmlessly on top of c2/c3.
    for col in (2, 2, 3, 3, 4):
        pos.play(col)
    # Turn is O's; X threatens at both (1,0) and (5,0).
    assert pos.opponent_winning_spots() & (1 << (1 * (HEIGHT + 1) + 0))
    assert pos.opponent_winning_spots() & (1 << (5 * (HEIGHT + 1) + 0))
    assert pos.non_losing_moves() == 0


def test_never_plays_under_an_opponent_threat():
    """Filling the cell below an opponent's winning cell just hands it to them.

    Construct X with a horizontal three along row 1 (columns 1, 2, 3). That
    threatens row 1 of columns 0 and 4, both of which are *empty* columns --
    so the cell currently playable there is row 0, sitting directly underneath
    the threat. Dropping a stone into either column lifts X straight into the
    win, so both must be excluded even though neither is an immediate loss.
    """
    pos = Position()
    # X takes row 1 of columns 1-3 while O fills the row beneath it.
    #      X     O     X     O     X     O     X
    for col in (6,    1,    1,    2,    2,    3,    3):
        pos.play(col)

    assert pos.current_player() == 2, "O should be to move"

    threat_left = 1 << (0 * (HEIGHT + 1) + 1)  # (col 0, row 1)
    threat_right = 1 << (4 * (HEIGHT + 1) + 1)  # (col 4, row 1)
    opponent_wins = pos.opponent_winning_spots()
    assert opponent_wins & threat_left
    assert opponent_wins & threat_right

    # Neither threat cell is playable yet, so O is not *forced* anywhere...
    assert pos.possible_moves() & opponent_wins == 0
    # ...but O must still avoid the two cells directly below them.
    allowed = pos.non_losing_moves()
    assert allowed & (1 << (0 * (HEIGHT + 1) + 0)) == 0, "must not play under the left threat"
    assert allowed & (1 << (4 * (HEIGHT + 1) + 0)) == 0, "must not play under the right threat"
    assert allowed != 0, "safe moves still exist here"


# --------------------------------------------------------------------------
# safe_moves: the column-list view of non_losing_moves
# --------------------------------------------------------------------------


def test_safe_moves_agrees_with_non_losing_moves_over_random_play():
    """The two views must never disagree; one is only a re-encoding of the other.

    Random play rather than a hand-picked board, because the interesting cases
    here are the ones nobody thinks to write down.
    """
    rng = random.Random(11)
    for _ in range(400):
        pos = Position()
        for _ in range(rng.randrange(0, 25)):
            legal = pos.legal_moves()
            if not legal or pos.has_won():
                break
            pos.play(rng.choice(legal))
        if pos.has_won() or pos.is_draw():
            continue

        allowed = pos.non_losing_moves()
        expected = {col for col in pos.legal_moves() if allowed & pos._landing_bit(col)}
        assert set(pos.safe_moves()) == expected
        # Every safe move is legal; no move outside the mask sneaks in.
        assert set(pos.safe_moves()) <= set(pos.legal_moves())


def test_safe_moves_is_centre_first():
    """Order is part of the contract -- callers use it as a tie-break."""
    columns = Position().safe_moves()
    assert columns == list(MOVE_ORDER)


def test_safe_moves_returns_only_the_block_when_forced():
    pos = Position()
    for col in (3, 0, 3, 1, 3):  # X threatens (3, 3); O must block
        pos.play(col)
    assert pos.safe_moves() == [3]


def test_safe_moves_is_empty_when_every_reply_loses():
    """Empty is information, not an error -- the caller decides what to do."""
    pos = Position()
    for col in (2, 2, 3, 3, 4):  # X threatens both (1, 0) and (5, 0)
        pos.play(col)
    assert pos.safe_moves() == []
    assert pos.legal_moves(), "the position is lost, but moves still exist"


# --------------------------------------------------------------------------
# Mechanics: play, keys, rendering
# --------------------------------------------------------------------------


def test_stones_stack_from_the_bottom():
    pos = Position()
    pos.play(3)
    grid = pos.to_grid()
    assert grid[HEIGHT - 1][3] == 1, "first stone must land on the bottom row"
    assert all(grid[r][3] == 0 for r in range(HEIGHT - 1))


def test_column_fills_and_then_rejects():
    pos = Position()
    for i in range(HEIGHT):
        assert pos.can_play(0)
        pos.play(0)
        # Alternate a filler column so nobody accidentally wins vertically.
        if i < HEIGHT - 1:
            pass
    assert not pos.can_play(0)
    assert popcount(pos.mask) == HEIGHT


def test_turn_alternates():
    pos = Position()
    assert pos.current_player() == 1
    pos.play(0)
    assert pos.current_player() == 2
    pos.play(1)
    assert pos.current_player() == 1


@pytest.mark.parametrize("seed", range(25))
def test_position_key_is_unique_per_position(seed):
    """Two different positions must never collide on ``key()``.

    A collision would make the transposition table return another position's
    score, which is the kind of bug that produces occasional inexplicable
    blunders rather than a clean failure.
    """
    rng = random.Random(2000 + seed)
    seen: dict[int, tuple[int, int]] = {}
    for _ in range(400):
        pos = _random_position(rng)
        identity = (pos.position, pos.mask)
        key = pos.key()
        if key in seen:
            assert seen[key] == identity, f"key collision: {seen[key]} vs {identity}"
        seen[key] = identity


def test_from_moves_rejects_out_of_range_column():
    with pytest.raises(ValueError, match="out of range"):
        Position.from_moves([0, 1, 9])


def test_from_moves_rejects_a_full_column():
    with pytest.raises(ValueError, match="full"):
        Position.from_moves([0] * 7)


def test_from_moves_rejects_moves_after_the_game_ended():
    # X wins with the 7th ply (columns 0,0,0,0 for X interleaved with 1s).
    with pytest.raises(ValueError, match="ended the game"):
        Position.from_moves([0, 1, 0, 1, 0, 1, 0, 2])


def test_from_moves_roundtrips_through_play():
    moves = [3, 3, 4, 4, 5, 2]
    a = Position.from_moves(moves)
    b = Position()
    for m in moves:
        b.play(m)
    assert (a.position, a.mask, a.moves) == (b.position, b.mask, b.moves)


def test_to_grid_uses_stable_absolute_colours():
    """Player identity must not flip as the turn changes.

    ``position`` means "the mover's stones", so a naive renderer would swap
    both players' colours every ply. The stone played first must stay player 1
    no matter whose turn it currently is.
    """
    pos = Position()
    pos.play(0)  # player 1
    first = pos.to_grid()[HEIGHT - 1][0]
    pos.play(1)  # player 2
    still_first = pos.to_grid()[HEIGHT - 1][0]
    assert first == 1
    assert still_first == 1, "player 1's stone changed colour when the turn flipped"
    assert pos.to_grid()[HEIGHT - 1][1] == 2


def test_has_won_reports_the_player_who_just_moved():
    pos = Position()
    for col in (0, 1, 0, 1, 0, 1):
        pos.play(col)
        assert not pos.has_won()
    pos.play(0)  # player 1 completes a vertical four
    assert pos.has_won()


def test_draw_detection_on_a_full_board():
    """Fill the board in a pattern that produces no four-in-a-row.

    The standard trick is column-pair ordering: 1122 repeated per column gives
    every column the pattern XXOO, and shifting the pattern between adjacent
    columns avoids horizontal and diagonal fours.
    """
    pos = Position()
    # Column order chosen so each column receives XXOOXX or OOXXOO alternately.
    order = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    for _ in range(3):
        for col in order:
            pos.play(col)
    assert pos.moves == WIDTH * HEIGHT
    assert pos.is_draw()


# --------------------------------------------------------------------------
# Remembered threat maps
# --------------------------------------------------------------------------


def _fresh(pos: Position) -> Position:
    """The same board, with nothing remembered about it."""
    return Position(pos.position, pos.mask, pos.moves)


def test_playing_a_stone_forgets_the_remembered_threat_maps():
    """The classic failure mode of a cache on a mutable object: a stale answer.

    ``play`` mutates in place, so both maps have to be dropped -- and the way to
    prove it is to ask *before* the move, which is what fills them.
    """
    pos = Position.from_moves([3, 2, 3, 4])
    pos.winning_spots()
    pos.opponent_winning_spots()

    pos.play(3)

    clean = _fresh(pos)
    assert pos.winning_spots() == clean.winning_spots()
    assert pos.opponent_winning_spots() == clean.opponent_winning_spots()


def test_a_remembered_map_does_not_change_what_a_position_is():
    """Two identical boards stay equal even if only one has been asked.

    The maps are a consequence of the board rather than part of it, so they are
    excluded from equality. Were they not, a position would stop being equal to
    itself halfway through a search.
    """
    asked = Position.from_moves([3, 3, 4])
    untouched = Position.from_moves([3, 3, 4])
    asked.winning_spots()
    asked.opponent_winning_spots()
    assert asked == untouched
    assert repr(asked) == repr(untouched)


@pytest.mark.parametrize("seed", range(20))
def test_remembered_maps_agree_with_a_fresh_position(seed):
    """Differential test for the cache: same board, same answer, always.

    Every position reached along a random game is asked for both maps twice --
    once as it stands, having already answered other questions during play, and
    once as a position that has never been asked anything.
    """
    rng = random.Random(5000 + seed)
    for _ in range(60):
        pos = _random_position(rng)
        clean = _fresh(pos)
        assert pos.winning_spots() == clean.winning_spots(), f"\n{pos}"
        assert pos.opponent_winning_spots() == clean.opponent_winning_spots(), f"\n{pos}"


@pytest.mark.parametrize("seed", range(20))
def test_played_agrees_with_copy_then_play(seed):
    """``played`` spells the move arithmetic out for speed; it must not drift.

    It is the one place in this file where the same rule is written twice, so it
    is the one place that needs a test whose only job is to compare the two.
    """
    rng = random.Random(9000 + seed)
    for _ in range(60):
        pos = _random_position(rng)
        if pos.has_won():
            continue
        for col in pos.legal_moves():
            stepwise = pos.copy()
            stepwise.play(col)
            direct = pos.played(col)
            assert (direct.position, direct.mask, direct.moves) == (
                stepwise.position,
                stepwise.mask,
                stepwise.moves,
            ), f"column {col}\n{pos}"
