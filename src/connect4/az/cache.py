"""Memoisation for network evaluations.

Measured on a 64-game self-play run at 64 simulations a move: of 100,460
positions handed to the network, only 62,433 were distinct. **38% of the work
was a position the network had already scored.** Two sources, and both are
inherent rather than accidental:

* *Transpositions.* Connect 4 reaches the same board by many move orders, and
  MCTS revisits the same subtree thousands of times within one search.
* *Shared openings.* Sixty-four games running in parallel all start from the
  empty board and stay near each other for the first few plies.

A cache is sound here for a reason worth stating precisely: **the network is
frozen for the duration of a self-play run.** Same position, same weights, same
answer -- so a hit is not an approximation, it is the identical number. That
stops being true the moment the weights change, which is why :meth:`clear`
exists and why the training loop calls it after every promotion.

Deliberately not an LRU. An LRU costs a reordering on every hit -- on this
workload, tens of millions of them -- to protect against an eviction pattern
this workload does not have. When the cap is reached the cache is simply
emptied. The following searches repopulate it quickly, since what they ask for
next is mostly what they asked for last.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from connect4.bitboard import WIDTH

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from connect4.bitboard import Position


class CachedEvaluator:
    """Wraps a batch evaluator, answering repeats from memory.

    Does two distinct jobs, both of which reduce the batch actually sent on:

    1. **Deduplicates within the batch.** The same position often arrives twice
       in one round from two different games.
    2. **Remembers across batches**, which is where the bulk of the saving is.

    The wrapped callable sees only the positions that are genuinely new, and
    sees them once each.
    """

    __slots__ = ("_priors", "_values", "capacity", "evaluate", "hits", "misses")

    def __init__(
        self,
        evaluate: Callable[[list[Position]], tuple[np.ndarray, np.ndarray]],
        capacity: int = 2_000_000,
    ) -> None:
        self.evaluate = evaluate
        self.capacity = capacity
        self._priors: dict[int, np.ndarray] = {}
        self._values: dict[int, float] = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, positions: Sequence[Position]) -> tuple[np.ndarray, np.ndarray]:
        n = len(positions)
        priors = np.empty((n, WIDTH), dtype=np.float32)
        values = np.empty(n, dtype=np.float32)

        # Positions still needing an answer, keyed so that two arrivals of the
        # same board share one slot in the outgoing batch.
        wanted: dict[int, list[int]] = {}
        to_evaluate: list[Position] = []
        keys: list[int] = []

        for i, position in enumerate(positions):
            key = position.key()
            cached = self._priors.get(key)
            if cached is not None:
                priors[i] = cached
                values[i] = self._values[key]
                self.hits += 1
                continue
            self.misses += 1
            slot = wanted.get(key)
            if slot is None:
                wanted[key] = [i]
                to_evaluate.append(position)
                keys.append(key)
            else:
                slot.append(i)

        if not to_evaluate:
            return priors, values

        fresh_priors, fresh_values = self.evaluate(to_evaluate)
        if len(self._priors) + len(keys) > self.capacity:
            self.clear()
        for j, key in enumerate(keys):
            prior = np.asarray(fresh_priors[j], dtype=np.float32)
            value = float(fresh_values[j])
            self._priors[key] = prior
            self._values[key] = value
            for i in wanted[key]:
                priors[i] = prior
                values[i] = value

        return priors, values

    def clear(self) -> None:
        """Forget everything. Mandatory whenever the weights change."""
        self._priors.clear()
        self._values.clear()

    def reset_stats(self) -> None:
        self.hits = self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __len__(self) -> int:
        return len(self._priors)
