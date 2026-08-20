"""The training half of the loop: turn self-play games into better weights.

Two targets, one shared trunk:

* **Policy.** Cross-entropy against the search's visit distribution. Note that
  the target is a *distribution*, not the move that was played -- the search's
  second and third choices carry real information about the position, and a
  one-hot target throws all of it away.
* **Value.** Mean squared error against the game's eventual result, from the
  point of view of the side to move.

Their sum is the whole objective. AlphaZero's paper weights them equally, and
this does too by default, but ``value_weight`` is exposed because the two heads
converge at very different speeds: the value head is fitting a noisy scalar
(one number per *game*, shared by every position in it) while the policy head
gets a rich target per *position*.

**The replay buffer is not an optimisation.** Training on only the newest games
makes the network chase its own most recent quirks, and the loop oscillates
instead of improving. Keeping a window of recent games is what makes it a
stable fixed-point iteration rather than a feedback squeal.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from connect4.az.features import mirror, mirror_policy, planes_batch

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from connect4.az.net import PolicyValueNet
    from connect4.az.selfplay import Example


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Optimiser settings.

    ``learning_rate`` at 2e-3 with Adam is on the brisk side, which suits a
    network this small and a data stream this noisy: the target distribution
    itself moves every iteration, so converging precisely on the current one is
    wasted effort.

    ``weight_decay`` is AlphaZero's L2 term. It matters more than usual here
    because self-play data is heavily autocorrelated -- consecutive positions in
    a game are nearly identical -- and an unregularised network will happily
    memorise the games rather than learn the game.

    ``augment`` mirrors each sample left-to-right with probability one half.
    Connect 4 is symmetric about the centre column, so this is free data that is
    exactly as true as the original, and it stops the network learning that
    column 2 and column 4 are different kinds of place.
    """

    batch_size: int = 256
    steps: int = 400
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    augment: bool = True
    grad_clip: float = 5.0


@dataclass(frozen=True, slots=True)
class LossReport:
    """What one call to :func:`train` cost, averaged over its steps."""

    policy: float
    value: float
    total: float
    steps: int

    def __str__(self) -> str:
        return (
            f"policy {self.policy:.4f}  value {self.value:.4f}  "
            f"total {self.total:.4f}  ({self.steps} steps)"
        )


class ReplayBuffer:
    """A bounded window over recent self-play positions.

    Stores the encoded planes rather than the ``Position`` objects. Encoding is
    cheap but not free, and every position is sampled several times over the
    life of the buffer; doing it once on the way in is the obvious trade. The
    cost is memory: 2 x 6 x 7 floats is 336 bytes a position, so a 200,000
    position buffer is about 67 MB. That fits, and it is worth checking against
    the machine before anyone raises the capacity.
    """

    __slots__ = ("boards", "policies", "values")

    def __init__(self, capacity: int = 200_000) -> None:
        self.boards: deque[np.ndarray] = deque(maxlen=capacity)
        self.policies: deque[np.ndarray] = deque(maxlen=capacity)
        self.values: deque[float] = deque(maxlen=capacity)

    def add(self, examples: Iterable[Example]) -> None:
        batch = list(examples)
        if not batch:
            return
        boards = planes_batch([e.position for e in batch])
        for i, example in enumerate(batch):
            self.boards.append(boards[i])
            self.policies.append(np.asarray(example.policy, dtype=np.float32))
            self.values.append(float(example.value))

    def __len__(self) -> int:
        return len(self.boards)

    def sample(
        self, size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw ``size`` positions *with* replacement.

        With replacement because the alternative -- shuffling and walking the
        buffer in epochs -- would have the network see the oldest games exactly
        as often as the newest ones, and the newest ones are the better games.
        """
        index = rng.integers(0, len(self.boards), size=size)
        boards = np.stack([self.boards[i] for i in index])
        policies = np.stack([self.policies[i] for i in index])
        values = np.array([self.values[i] for i in index], dtype=np.float32)
        return boards, policies, values


def augment_batch(
    boards: np.ndarray, policies: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror a random half of the batch. The board and its policy must flip together."""
    flip = rng.random(len(boards)) < 0.5
    if not flip.any():
        return boards, policies
    boards = boards.copy()
    policies = policies.copy()
    boards[flip] = mirror(boards[flip])
    policies[flip] = mirror_policy(policies[flip])
    return boards, policies


def losses(
    logits: torch.Tensor,
    value_out: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    value_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Policy cross-entropy, value MSE, and their weighted sum.

    Cross-entropy is written out rather than taken from ``nn.CrossEntropyLoss``
    because that one wants hard class indices, and the target here is a full
    distribution. ``log_softmax`` then a dot product is the same quantity,
    computed in the numerically stable order.
    """
    policy_loss = -(policy_target * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    value_loss = torch.mean((value_out - value_target) ** 2)
    return policy_loss, value_loss, policy_loss + value_weight * value_loss


def train(
    net: PolicyValueNet,
    buffer: ReplayBuffer,
    config: TrainConfig | None = None,
    rng: np.random.Generator | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> LossReport:
    """Run ``config.steps`` optimiser steps against samples from ``buffer``.

    The optimiser may be passed in so that a training *loop* can keep Adam's
    moment estimates across iterations. Rebuilding the optimiser every iteration
    throws them away and makes the first steps after each self-play round much
    noisier than they need to be.
    """
    cfg = config or TrainConfig()
    generator = rng if rng is not None else np.random.default_rng()
    if len(buffer) == 0:
        return LossReport(0.0, 0.0, 0.0, 0)

    opt = optimizer or torch.optim.Adam(
        net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    net.train()

    policy_total = value_total = combined_total = 0.0
    for _ in range(cfg.steps):
        boards, policies, values = buffer.sample(cfg.batch_size, generator)
        if cfg.augment:
            boards, policies = augment_batch(boards, policies, generator)

        logits, value_out = net(torch.from_numpy(boards))
        policy_loss, value_loss, total = losses(
            logits,
            value_out,
            torch.from_numpy(policies),
            torch.from_numpy(values),
            cfg.value_weight,
        )

        opt.zero_grad(set_to_none=True)
        total.backward()
        # Self-play data contains the occasional position whose value target is
        # wildly at odds with what the network currently believes -- a game that
        # turned on one move. Clipping keeps one such sample from undoing an
        # iteration's worth of progress.
        nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()

        policy_total += policy_loss.item()
        value_total += value_loss.item()
        combined_total += total.item()

    net.eval()
    steps = max(cfg.steps, 1)
    return LossReport(
        policy=policy_total / steps,
        value=value_total / steps,
        total=combined_total / steps,
        steps=cfg.steps,
    )


def supervised_batches(
    boards: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Shuffle a fixed dataset into epoch-order batches.

    Used by the pretraining script rather than the self-play loop: a labelled
    file on disk is a fixed population, and sampling it with replacement would
    just be a worse shuffle.
    """
    order = rng.permutation(len(boards))
    return [
        (boards[chunk], policies[chunk], values[chunk])
        for chunk in (order[i : i + batch_size] for i in range(0, len(order), batch_size))
        if len(chunk) > 1  # a batch of one breaks BatchNorm in training mode
    ]
