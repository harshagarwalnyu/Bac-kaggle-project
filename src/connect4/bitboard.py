"""Bitboard representation of a Connect 4 position.

Why bitboards
-------------
The obvious representation is a 6x7 list of lists. Checking for a win then means
scanning rows, columns and both diagonals -- dozens of comparisons per node. A
search engine visits millions of nodes, so that cost dominates everything.

Instead we pack the board into a single Python integer and detect a win with
four shift-and-mask steps, no loops at all. That is the difference between a bot
that searches a few thousand positions per second and one that searches
hundreds of thousands.

Bit layout
----------
We use the classic 7-bits-per-column layout (Fhourstones / Pascal Pons). Each
column gets 7 bits: 6 playable cells plus one always-empty "sentinel" bit on
top. 7 columns x 7 bits = 49 bits, comfortably inside a machine word.

Bit index of (column c, row r) is ``c * 7 + r``, with row 0 at the *bottom*::

     column:      0   1   2   3   4   5   6
                +---+---+---+---+---+---+---+
    sentinel  6 |  6| 13| 20| 27| 34| 41| 48|   <- never occupied
       row    5 |  5| 12| 19| 26| 33| 40| 47|
       row    4 |  4| 11| 18| 25| 32| 39| 46|
       row    3 |  3| 10| 17| 24| 31| 38| 45|
       row    2 |  2|  9| 16| 23| 30| 37| 44|
       row    1 |  1|  8| 15| 22| 29| 36| 43|
       row    0 |  0|  7| 14| 21| 28| 35| 42|   <- bottom of the board
                +---+---+---+---+---+---+---+

The sentinel row is the trick that makes the whole scheme work. Column stride
is 7, so a vertical run is 4 bits that are adjacent (stride 1). Without the
sentinel, the top cell of column 0 (bit 5) and the bottom of column 1 (bit 6)
would be adjacent too, and a vertical win check would happily match four bits
that wrap from one column into the next. The permanently-empty sentinel bit
sits between them and breaks every such wrap.

Two-integer encoding
--------------------
We store the position as two integers rather than one board of "colours":

``position``  bits set for the stones of the player *whose turn it is*
``mask``      bits set for *every* stone on the board, either colour

The opponent's stones are therefore ``position ^ mask``. This encoding makes
playing a move almost free and, crucially, makes the code colour-agnostic:
search never has to ask "am I red or yellow", it only ever reasons from the
point of view of the side to move. That is what lets us write negamax instead
of a separate minimax with two mirrored branches.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

WIDTH = 7
HEIGHT = 6

# Bits per column: the 6 playable cells plus the sentinel.
_COL_STRIDE = HEIGHT + 1  # 7

# The four directions a line of four can run in, expressed as a bit-shift
# distance. Read them against the diagram above:
#   1  -> vertical      (next cell up in the same column)
#   7  -> horizontal    (same row, next column)
#   6  -> diagonal "\"  (one column right, one row down)
#   8  -> diagonal "/"  (one column right, one row up)
_DIRECTIONS = (1, _COL_STRIDE, _COL_STRIDE - 1, _COL_STRIDE + 1)

# The same four directions, with their doubled and tripled shifts worked out
# once instead of on every visit to the hottest function in the program.
_DIRECTION_STEPS = tuple((d, 2 * d, 3 * d) for d in _DIRECTIONS)

# A bit set at the bottom cell of every column: 0b1000000100000010000001000000100000010000001
BOTTOM_MASK_ALL = sum(1 << (col * _COL_STRIDE) for col in range(WIDTH))

# Every playable cell (all 42 of them), sentinel row excluded.
BOARD_MASK = BOTTOM_MASK_ALL * ((1 << HEIGHT) - 1)

# 49 bits -- used to clip left-shifts, since Python ints are arbitrary
# precision and would otherwise grow past the board instead of overflowing.
FULL_MASK = (1 << (WIDTH * _COL_STRIDE)) - 1

MIN_SCORE = -(WIDTH * HEIGHT) // 2 + 3
MAX_SCORE = (WIDTH * HEIGHT + 1) // 2 - 3

# Column exploration order for the search: centre first. A centre stone
# participates in more possible lines of four than an edge stone, so centre
# moves are more often best, and trying them first makes alpha-beta cut off
# sooner. Pure move-ordering heuristic; it cannot change the result, only the
# time taken to reach it.
MOVE_ORDER = (3, 2, 4, 1, 5, 0, 6)


_BOTTOM_MASKS = tuple(1 << (col * _COL_STRIDE) for col in range(WIDTH))
_TOP_MASKS = tuple(1 << (col * _COL_STRIDE + HEIGHT - 1) for col in range(WIDTH))
_COLUMN_MASKS = tuple(((1 << HEIGHT) - 1) << (col * _COL_STRIDE) for col in range(WIDTH))


def bottom_mask(col: int) -> int:
    """Bit for the bottom cell of ``col``."""
    return _BOTTOM_MASKS[col]


def top_mask(col: int) -> int:
    """Bit for the topmost *playable* cell of ``col`` (row 5, not the sentinel)."""
    return _TOP_MASKS[col]


def column_mask(col: int) -> int:
    """All six playable bits of ``col``."""
    return _COLUMN_MASKS[col]


def popcount(bits: int) -> int:
    """Number of set bits. ``int.bit_count`` is a single CPU instruction."""
    return bits.bit_count()


def has_alignment(pos: int) -> bool:
    """True if ``pos`` contains four in a row, in any of the four directions.

    The idea, for a direction with shift ``d``:

    ``pos & (pos >> d)`` keeps a bit only where that cell *and* the cell ``d``
    further along are both occupied -- i.e. it marks the start of every run of
    two. Shifting that result by ``2 * d`` and ANDing again asks whether a run
    of two is followed by another run of two starting two steps later, which is
    exactly a run of four.

    Only right-shifts are used, so no bit can escape past the top of the board
    and no clipping is needed.
    """
    for direction in _DIRECTIONS:
        pairs = pos & (pos >> direction)
        if pairs & (pairs >> (2 * direction)):
            return True
    return False


def _winning_spots(position: int, mask: int) -> int:
    """Every empty cell where the *current player* would immediately have four.

    Unlike :func:`has_alignment`, which answers "is there a win on the board",
    this answers "where are the wins one stone away" -- including cells that are
    not yet reachable because the column is not filled up to them. Those
    unreachable threats still matter: they are what makes a position a zugzwang
    trap, and the search uses them for both pruning and move ordering.

    For each direction we look for the three patterns that a single stone can
    complete: the gap at either end of a run of three, and the gap in the middle
    of a split run. Every left-shift is clipped to ``FULL_MASK`` so the value
    cannot grow beyond the 49-bit board.
    """
    result = 0
    for d1, d2, d3 in _DIRECTION_STEPS:

        # Two stones extending upward from a candidate cell...
        pair_up = (position << d1) & (position << d2) & FULL_MASK
        result |= pair_up & ((position << d3) & FULL_MASK)  # _XXX  -> fill the low end
        result |= pair_up & (position >> d1)  # X_XX  -> fill an inner gap

        # ...and the mirror image, extending downward.
        pair_down = (position >> d1) & (position >> d2)
        result |= pair_down & ((position << d1) & FULL_MASK)  # XX_X -> the other inner gap
        result |= pair_down & (position >> d3)  # XXX_ -> fill the high end

    # A "winning spot" is only useful if the cell is actually empty.
    return result & BOARD_MASK & ~mask


@dataclass(slots=True)
class Position:
    """A Connect 4 position, always seen from the point of view of the mover.

    ``slots=True`` because the search allocates and copies these constantly and
    we do not want a per-instance ``__dict__``.
    """

    position: int = 0  # stones belonging to the player to move
    mask: int = 0  # stones belonging to either player
    moves: int = 0  # plies played so far; parity tells us whose turn it is

    # Both threat maps, computed on first use and remembered. Every node of
    # the search asks for them two or three times over -- the move filter, the
    # ordering, the leaf evaluator -- and the answer cannot change while the
    # board does not. -1 means not computed yet, which no real map can be.
    # Excluded from equality and repr: they are a consequence of the board,
    # not part of it, so two equal boards stay equal whether or not either has
    # been asked about its threats.
    _wins: int = field(default=-1, compare=False, repr=False)
    _opponent_wins: int = field(default=-1, compare=False, repr=False)

    # ---------------------------------------------------------------- queries

    def can_play(self, col: int) -> bool:
        """Is ``col`` not yet full?

        We only need to test the *top* playable cell. If it is empty the column
        has room; if it is occupied the column is full. One AND, no counting.
        """
        return (self.mask & _TOP_MASKS[col]) == 0

    def legal_moves(self) -> list[int]:
        """Playable columns, in centre-first search order."""
        return [col for col in MOVE_ORDER if self.can_play(col)]

    def is_winning_move(self, col: int) -> bool:
        """Would playing ``col`` right now complete four in a row?"""
        return bool(self.winning_spots() & self._landing_bit(col))

    def winning_spots(self) -> int:
        """Empty cells that would immediately win *for the player to move*."""
        spots = self._wins
        if spots < 0:
            spots = self._wins = _winning_spots(self.position, self.mask)
        return spots

    def opponent_winning_spots(self) -> int:
        """Empty cells that would immediately win *for the opponent*."""
        spots = self._opponent_wins
        if spots < 0:
            spots = self._opponent_wins = _winning_spots(
                self.position ^ self.mask, self.mask
            )
        return spots

    def possible_moves(self) -> int:
        """Bitmap of the cells that are actually playable right now.

        ``mask + BOTTOM_MASK_ALL`` is the carry trick: adding the bottom bit of
        every column to the occupancy makes each column's stack of 1s roll over
        into the single cell just above it. Clipping to ``BOARD_MASK`` discards
        columns that overflowed into their sentinel, i.e. full columns.
        """
        return (self.mask + BOTTOM_MASK_ALL) & BOARD_MASK

    def non_losing_moves(self) -> int:
        """Playable cells that do not hand the opponent an immediate win.

        Three separate things get filtered out here, and the order matters:

        1. If the opponent has two or more distinct winning spots we cannot
           block them all, so the position is lost whatever we do -- return 0.
        2. If the opponent has exactly one winning spot we are forced to play
           it (either it wins for us too, or it blocks them).
        3. Never play directly *underneath* one of the opponent's winning
           cells, because that stone would lift them straight into the win.
        """
        possible = self.possible_moves()
        opponent_wins = self.opponent_winning_spots()
        forced = possible & opponent_wins

        if forced:
            if forced & (forced - 1):  # more than one bit set -> unstoppable
                return 0
            possible = forced  # exactly one reply, we must play it

        return possible & ~(opponent_wins >> 1)

    def safe_moves(self) -> list[int]:
        """:meth:`non_losing_moves` as a column list, in centre-first order.

        The bitmask form is what the search wants; callers outside the search
        want columns. Returns an empty list when every reply loses -- which is
        information, not an error, so it is left to the caller to interpret.
        """
        allowed = self.non_losing_moves()
        return [col for col in MOVE_ORDER if allowed & self._landing_bit(col)]

    def is_draw(self) -> bool:
        return self.moves == WIDTH * HEIGHT

    def key(self) -> int:
        """A value that uniquely identifies this position, for the table.

        Neither ``position`` nor ``mask`` identifies a position alone, and
        hashing the pair costs an allocation. ``position + mask`` is not unique
        either -- but ``position + mask + BOTTOM_MASK_ALL`` is. Adding the
        bottom bits pushes each column's occupancy up by one, which encodes the
        column heights into the sum, and the whole thing collapses to one int.
        """
        return self.position + self.mask + BOTTOM_MASK_ALL

    # ---------------------------------------------------------------- updates

    def _landing_bit(self, col: int) -> int:
        """Which cell a stone dropped into ``col`` comes to rest in."""
        return (self.mask + _BOTTOM_MASKS[col]) & _COLUMN_MASKS[col]

    def play(self, col: int) -> None:
        """Drop a stone into ``col`` and hand the turn over. Mutates in place.

        The XOR is the whole reason for the two-integer encoding. Before the
        move ``position`` holds *our* stones; ``position ^ mask`` cancels them
        out and leaves the opponent's, which is precisely what ``position``
        should mean once it is their turn. The new stone is added to ``mask``
        only -- it belongs to the side that just moved, i.e. no longer the side
        ``position`` describes.
        """
        self.position ^= self.mask
        self.mask |= self.mask + _BOTTOM_MASKS[col]
        self.moves += 1
        # The board moved, so anything remembered about it is now a lie.
        self._wins = self._opponent_wins = -1

    def copy(self) -> Position:
        # The threat maps come along: a copy is the same board, and the search
        # copies far more often than it plays.
        return Position(
            self.position, self.mask, self.moves, self._wins, self._opponent_wins
        )

    def played(self, col: int) -> Position:
        """Non-mutating :meth:`play` -- returns the resulting position.

        Spelled out rather than ``copy`` then ``play`` because the search builds
        one of these for every move it considers, and the two extra calls plus
        the throwaway threat maps of the intermediate copy are pure overhead.
        The arithmetic is exactly :meth:`play`'s, and the differential tests hold
        both to the same slow reference implementation.
        """
        mask = self.mask
        return Position(
            self.position ^ mask, mask | (mask + _BOTTOM_MASKS[col]), self.moves + 1
        )

    # ------------------------------------------------------- interop / display

    @classmethod
    def from_moves(cls, columns: Sequence[int]) -> Position:
        """Build a position by replaying a sequence of column indices.

        Rejects illegal input rather than silently producing a corrupt board:
        a move into a full column, an out-of-range column, or a move played
        after the game is already over.
        """
        pos = cls()
        for ply, col in enumerate(columns):
            col = int(col)
            if not 0 <= col < WIDTH:
                raise ValueError(f"move {ply}: column {col} out of range 0..{WIDTH - 1}")
            if not pos.can_play(col):
                raise ValueError(f"move {ply}: column {col} is full")
            if pos.is_winning_move(col):
                pos.play(col)
                if ply != len(columns) - 1:
                    raise ValueError(f"move {ply} ended the game; later moves are illegal")
            else:
                pos.play(col)
        return pos

    def current_player(self) -> int:
        """1 for the player who moved first, 2 for the other."""
        return 1 if self.moves % 2 == 0 else 2

    def to_grid(self) -> list[list[int]]:
        """Render as ``grid[row][col]``, row 0 at the *top* for display.

        Values are 0 empty, 1 first player, 2 second player -- absolute
        identities, not "mover / opponent", because the UI needs stable colours.
        """
        mover = self.current_player()
        other = 3 - mover
        opponent_stones = self.position ^ self.mask

        grid = [[0] * WIDTH for _ in range(HEIGHT)]
        for col in range(WIDTH):
            for row in range(HEIGHT):
                bit = 1 << (col * _COL_STRIDE + row)
                display_row = HEIGHT - 1 - row  # flip so row 0 prints at the top
                if self.position & bit:
                    grid[display_row][col] = mover
                elif opponent_stones & bit:
                    grid[display_row][col] = other
        return grid

    def has_won(self) -> bool:
        """Did the player who *just* moved make four in a row?

        Note whose stones we test: after :meth:`play`, ``position`` describes
        the side to move, so the side that just moved is ``position ^ mask``.
        """
        return has_alignment(self.position ^ self.mask)

    def __str__(self) -> str:
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = ["".join(symbols[cell] for cell in row) for row in self.to_grid()]
        return "\n".join(rows) + "\n" + "0123456"
