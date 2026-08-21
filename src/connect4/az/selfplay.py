"""Self-play: where the training data comes from.

This is the part of AlphaZero that has no analogue in the rest of the project.
Everything else here learns from a file someone else produced. This learns from
games the current network plays against itself, and the only external input is
the rules.

Each move is chosen by a full MCTS search, and two things are written down: the
search's **visit distribution** (a better move-ranking than the network's raw
policy, because search improved it) and, once the game ends, **who won** from
that position's point of view. Training on those two targets is the whole loop
-- the network learns to predict what the search concluded, the next search
starts from a better prior, and its conclusions improve in turn.

**Games run in batches, not one at a time.** A single game is a strictly
sequential thing, and a batch of one wastes almost all of a forward pass. So
``games_in_parallel`` games advance in lockstep, one move each per round, and
every search in the round gets its leaves evaluated together. Finished games are
replaced immediately so the batch stays full.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from connect4.az.mcts import MCTSConfig, Search, run_batch
from connect4.bitboard import WIDTH, Position

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    """How games are generated.

    ``opening_plies`` is how long moves are *sampled* from the visit
    distribution rather than taken greedily. Without it every game from the
    empty board is the same game, and a training set of identical games teaches
    a network that the opening has exactly one move. Eight plies is roughly
    where Connect 4 openings stop being interchangeable -- and, not by accident,
    the depth of the UCI dataset.

    ``temperature`` applies during those opening plies; afterwards the move is
    the most-visited one, because past the opening the point of a self-play game
    is to be *played well*, so that the result labelling the positions means
    something.

    ``games_in_parallel`` is a throughput knob and nothing else -- the games are
    independent, so it cannot change what gets played. It matters because a
    forward pass over 128 boards costs barely more than one over 40, and the
    round-robin structure means the batch is roughly this wide. 128 was chosen
    from a sweep of 64 / 128 / 256 / 384, though honestly: repeated runs on this
    machine varied by a factor of two, so the sweep says "at least 128" and not
    much more than that.
    """

    games_in_parallel: int = 128
    opening_plies: int = 8
    temperature: float = 1.0
    mcts: MCTSConfig = field(default_factory=MCTSConfig)


@dataclass(frozen=True, slots=True)
class Example:
    """One training row: a position, what the search wanted, and how it ended.

    ``policy`` is the search's visit distribution over columns. ``value`` is the
    final result **from the point of view of the side to move in this
    position** -- +1 if they went on to win, -1 if they lost, 0 for a draw. The
    point of view is the part worth double-checking: it flips every ply, and
    getting it wrong trains a network to prefer losing.
    """

    position: Position
    policy: np.ndarray
    value: float


class _Game:
    """One in-flight game, accumulating positions until it ends."""

    __slots__ = ("policies", "position", "positions")

    def __init__(self) -> None:
        self.position = Position()
        self.positions: list[Position] = []
        self.policies: list[np.ndarray] = []

    @property
    def finished(self) -> bool:
        return self.position.has_won() or self.position.is_draw()

    def record(self, policy: np.ndarray) -> None:
        self.positions.append(self.position)
        self.policies.append(policy)

    def examples(self) -> list[Example]:
        """Label every recorded position with the result, sign-flipped per ply.

        The game ends when someone plays a winning move, so the *last* recorded
        position is the one its mover won from: value +1. One ply earlier it was
        the loser's turn: -1. And so on back to the start. A draw makes every
        value 0 and the alternation moot.
        """
        if self.position.is_draw() and not self.position.has_won():
            outcome = 0.0
        else:
            outcome = 1.0
        values = [outcome * (-1.0) ** (len(self.positions) - 1 - i) for i in range(len(self.positions))]
        return [
            Example(position=p, policy=pol, value=v)
            for p, pol, v in zip(self.positions, self.policies, values, strict=True)
        ]


def play_games(
    evaluate: Callable[[list[Position]], tuple[np.ndarray, np.ndarray]],
    games: int,
    config: SelfPlayConfig | None = None,
    rng: np.random.Generator | None = None,
) -> Iterator[list[Example]]:
    """Play ``games`` self-play games, yielding each game's examples as it ends.

    Yielding per game rather than returning a list keeps memory flat and lets a
    caller checkpoint or report progress partway through a long run -- which
    matters when a run is measured in hours.
    """
    cfg = config or SelfPlayConfig()
    generator = rng if rng is not None else np.random.default_rng()

    remaining = games
    active: list[_Game] = []
    while remaining > 0 and len(active) < cfg.games_in_parallel:
        active.append(_Game())
        remaining -= 1

    while active:
        searches = [Search(g.position, cfg.mcts, generator) for g in active]
        run_batch(searches, evaluate)

        finished: list[_Game] = []
        for game, search in zip(active, searches, strict=True):
            ply = game.position.moves
            temperature = cfg.temperature if ply < cfg.opening_plies else 0.0
            # Always record the full-temperature distribution, whatever
            # temperature the *move* was chosen at. The training target is what
            # the search believed, not how greedily we acted on it; sharpening
            # the target to a one-hot would throw away everything the search
            # learned about the runners-up.
            game.record(search.policy(1.0))
            policy = search.policy(temperature)
            column = int(generator.choice(WIDTH, p=policy))
            game.position = game.position.played(column)
            if game.finished:
                finished.append(game)

        for game in finished:
            yield game.examples()
            active.remove(game)
            if remaining > 0:
                active.append(_Game())
                remaining -= 1
