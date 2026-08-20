"""Board -> tensor, for a convolutional network.

This is not the encoding :mod:`connect4.dataset` uses. That one flattens the
board into a feature vector with hand-derived extras (threat counts, column
heights) because it feeds a dense MLP. A convolution wants the board's *shape*
back: four-in-a-row is a translation-invariant pattern, which is precisely the
thing a conv layer gets for free and a dense layer has to learn seven times over.

So the encoding here is deliberately plain -- two occupancy planes and nothing
else. No threat counts, no heights. Anything hand-derived that helps is a
feature the network could have learned, and leaving it out is how we find out
whether it did.

Orientation: ``planes[p][row][col]`` with **row 0 at the bottom**, matching the
bitboard's numbering rather than the display's. Nothing in a convolution cares
which way up the board is, but a mismatch between training and inference would
be silent and fatal, so there is exactly one convention and it is this one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from connect4.bitboard import HEIGHT, WIDTH

if TYPE_CHECKING:
    from collections.abc import Sequence

    from connect4.bitboard import Position

#: Occupancy planes: the mover's stones, then the opponent's.
PLANES = 2

_COL_STRIDE = HEIGHT + 1

# Precomputed bit index for every cell, so encoding is a couple of array ops
# rather than 42 shifts in a Python loop. Shape (HEIGHT, WIDTH), row 0 bottom.
_BIT_INDEX = np.array(
    [[col * _COL_STRIDE + row for col in range(WIDTH)] for row in range(HEIGHT)],
    dtype=np.uint64,
)


def _unpack(bits: int) -> np.ndarray:
    """Spread the set bits of ``bits`` over a ``(HEIGHT, WIDTH)`` float plane."""
    # Python ints are arbitrary precision and numpy's are not; 49 bits fits in
    # uint64, which is why the board uses a 7-bit column stride at all.
    return ((np.uint64(bits) >> _BIT_INDEX) & np.uint64(1)).astype(np.float32)


def planes(position: Position) -> np.ndarray:
    """Encode one position as ``(PLANES, HEIGHT, WIDTH)``, mover's point of view.

    Point of view is the whole trick. ``Position.position`` always holds the
    stones of the side to move, so plane 0 is "mine" and plane 1 is "theirs" no
    matter whose turn it actually is. One network serves both players, and the
    value head's output means the same thing at every node of a search.
    """
    mine = _unpack(position.position)
    theirs = _unpack(position.position ^ position.mask)
    return np.stack((mine, theirs))


def planes_batch(positions: Sequence[Position]) -> np.ndarray:
    """Encode many positions into one ``(N, PLANES, HEIGHT, WIDTH)`` array."""
    if not positions:
        return np.zeros((0, PLANES, HEIGHT, WIDTH), dtype=np.float32)
    return np.stack([planes(p) for p in positions])


def mirror(batch: np.ndarray) -> np.ndarray:
    """Flip a batch of plane stacks left-to-right.

    Connect 4 is symmetric about the centre column: a position and its mirror
    have the same value, and their best moves are mirrors of each other. That
    doubles the training set for free, which matters a great deal when the data
    comes from self-play at a few games per second.
    """
    return batch[..., ::-1].copy()


def mirror_policy(policy: np.ndarray) -> np.ndarray:
    """The matching flip for a policy vector or batch of them."""
    return policy[..., ::-1].copy()
