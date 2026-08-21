"""Two ways to let the network actually play.

Until now the network in this project has never chosen a move. It publishes an
opinion next to the search's and the UI draws both, which is honest but leaves
the obvious question unanswered. These are the two answers:

:class:`AZPlayer`
    The network plays *through MCTS*, AlphaZero style. Prior from the policy
    head, leaf value from the value head, move from the visit counts.

:func:`value_evaluator`
    The network plays *through the existing alpha-beta search*, as a drop-in
    replacement for :func:`connect4.engine.heuristic_evaluator`. Same tree, same
    move ordering, same everything -- only the leaf scoring changes, so any
    difference in strength is attributable to the evaluator and nothing else.

Having both is the point. They share a value head, so grading one against the
other separates "the network learned something" from "MCTS is a better search
for this than alpha-beta", and those are very different claims.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from connect4.az.mcts import MCTSConfig, Search, run_batch

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from connect4.bitboard import Position

#: Matches the band :func:`connect4.engine.heuristic_evaluator` returns. The
#: search treats anything inside it as an opinion rather than a proof, and a
#: value head saturating at exactly 1.0 would sit awkwardly against that.
_OPINION_LIMIT = 0.95


class AZPlayer:
    """Chooses moves by running MCTS with a network evaluator.

    ``temperature`` defaults to 0 -- the most-visited move, deterministically.
    That is what you want for a *player*; sampling belongs in self-play, where
    variety is the goal, not in a game someone is trying to win. It stays a
    parameter because a tournament of a deterministic player against another
    deterministic player replays one game per pairing, and the only cures are
    forced openings or a little temperature.
    """

    __slots__ = ("config", "evaluate", "rng", "temperature")

    def __init__(
        self,
        evaluate: Callable[[list[Position]], tuple[np.ndarray, np.ndarray]],
        simulations: int = 200,
        temperature: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        # Noise off: it exists to diversify self-play openings, and a player
        # that adds random mass to its own prior is simply playing worse.
        self.config = MCTSConfig(simulations=simulations, add_noise=False)
        self.evaluate = evaluate
        self.temperature = temperature
        self.rng = rng if rng is not None else np.random.default_rng()

    def choose_move(self, position: Position) -> int:
        search = self.search(position)
        if self.temperature <= 0:
            return search.best_move()
        return int(self.rng.choice(len(search.root.prior), p=search.policy(self.temperature)))

    def search(self, position: Position) -> Search:
        """The whole search object, for callers that want the visit counts too."""
        search = Search(position, self.config, self.rng)
        run_batch([search], self.evaluate)
        return search

    def choose_moves(self, positions: Sequence[Position]) -> list[int]:
        """Pick a move for several independent positions, batching the network calls.

        This is what makes an arena run bearable: a tournament advances many
        games at once, and evaluating one board at a time would spend nearly all
        of its wall clock in per-call overhead rather than in arithmetic.
        """
        searches = [Search(p, self.config, self.rng) for p in positions]
        run_batch(searches, self.evaluate)
        return [s.best_move() for s in searches]


def value_evaluator(
    evaluate: Callable[[list[Position]], tuple[np.ndarray, np.ndarray]],
    cache: dict[int, float] | None = None,
) -> Callable[[Position], float]:
    """Wrap a network as a leaf evaluator for :class:`connect4.engine.Engine`.

    The policy head is ignored here -- alpha-beta has its own move ordering and
    the seam it exposes is scoring a leaf, nothing else.

    **This is slow, and unavoidably so.** Alpha-beta visits leaves one at a
    time, in an order that depends on what the previous leaf returned, so there
    is nothing to batch; every leaf is a separate forward pass. The cache takes
    the edge off, because transpositions are common and a position's value does
    not change, but a network evaluator will always cost far more per node than
    a dozen bit operations. Whether it buys enough strength to pay for the nodes
    it loses is exactly the question ``scripts/validate.py`` exists to settle --
    and last time it was asked, about a hand-written MLP, the answer was no.
    """
    memo = cache if cache is not None else {}

    def evaluator(position: Position) -> float:
        key = position.key()
        hit = memo.get(key)
        if hit is not None:
            return hit
        _, values = evaluate([position])
        value = max(-_OPINION_LIMIT, min(_OPINION_LIMIT, float(values[0])))
        memo[key] = value
        return value

    return evaluator
