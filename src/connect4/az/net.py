"""The policy-value network, in PyTorch.

This is the one module in the project that imports a framework, and it is worth
being explicit about why the rule changed. The existing :mod:`connect4.model` is
a hand-written NumPy MLP, and that was the right call for what it does: a small
dense classifier is genuinely legible when you write the backward pass yourself,
and legibility is this project's whole premise. A residual convolutional tower
trained by self-play is not that. Hand-rolling conv backprop here would not
teach anyone anything -- it would just be slower and, far more likely than not,
subtly wrong.

So: torch, as an optional extra. ``uv sync --extra az``. Playing a game against
the shipped bot still needs neither torch nor this file.

**Architecture**, and why it is small. AlphaGo Zero used 20 residual blocks of
256 filters. Connect 4 has 4,531,985,219,092 positions to Go's ~10^170, a
6x7 board, and seven legal moves. The bottleneck here is not capacity, it is how
many self-play games an 8-core CPU can generate -- so the tower is sized to be
fast enough that the data keeps coming, and every parameter added has to earn
its place against the games it costs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from connect4.az.features import PLANES, mirror, mirror_policy, planes_batch
from connect4.bitboard import HEIGHT, WIDTH

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from connect4.bitboard import Position


@dataclass(frozen=True, slots=True)
class NetConfig:
    """Shape of the tower, stored inside every checkpoint.

    Saving the shape alongside the weights is not fussiness. A checkpoint whose
    architecture lives only in whatever the training script's defaults happened
    to be that week is a checkpoint you cannot load six months later, and the
    failure mode is a shape-mismatch traceback rather than anything informative.

    **The defaults are measured, not guessed.** Forward pass over a batch of
    128 boards, best of 15, four torch threads, on the 8-core CPU this was
    developed on:

    ==========  ==========  ============
    shape       parameters  boards/second
    ==========  ==========  ============
    64 x 4      300,826     5,479
    48 x 3      129,514     8,431
    **32 x 3**  **59,834**  **17,943**
    32 x 2      41,274      27,229
    ==========  ==========  ============

    32 x 3 is where the curve bends. Going up to 64 x 4 costs 3.3x the time for
    5x the parameters, and on a CPU that time is self-play games not played --
    which is the resource this whole method is actually short of. Dropping to
    two blocks is faster again, but two residual blocks give a receptive field
    that barely spans a four-in-a-row, and there is no point being quick about
    a board you cannot see all of.
    """

    channels: int = 32
    blocks: int = 3
    value_hidden: int = 64


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions and a skip connection.

    The skip is what makes depth free-ish: the block starts life computing the
    identity, so adding blocks cannot make an untrained network worse, and
    gradients reach the early layers without passing through every weight in
    between.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        # Add before the activation, not after: the identity path has to stay
        # linear all the way through for the "cannot make it worse" argument to
        # hold.
        return torch.relu(x + y)


class PolicyValueNet(nn.Module):
    """Two heads on a shared trunk.

    Sharing the trunk is not just a saving. "Which move is good here" and "who
    is winning here" are the same question asked twice, and the features that
    answer one answer the other; training them together regularises both. It
    also means the value head gets gradient from every self-play position rather
    than only from the ones whose game ended informatively.

    The value head ends in ``tanh``: a value is a game result in ``[-1, 1]``,
    always from the point of view of the side to move. The policy head returns
    **logits**, not probabilities -- the loss wants logits, and the search
    softmaxes them itself.
    """

    def __init__(self, config: NetConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetConfig()
        c = self.config.channels

        self.stem = nn.Sequential(
            nn.Conv2d(PLANES, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*(ResidualBlock(c) for _ in range(self.config.blocks)))

        # Both heads squeeze to very few channels before flattening. A 1x1
        # convolution down to 2 channels is 2*6*7 = 84 numbers into the linear
        # layer instead of 64*6*7 = 2688, which is where most of the parameters
        # would otherwise sit -- in the least interesting part of the network.
        self.policy_head = nn.Sequential(
            nn.Conv2d(c, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * HEIGHT * WIDTH, WIDTH),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(c, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(HEIGHT * WIDTH, self.config.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trunk = self.tower(self.stem(x))
        return self.policy_head(trunk), self.value_head(trunk).squeeze(-1)

    # ------------------------------------------------------------ persistence

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, map_location: str = "cpu") -> PolicyValueNet:
        # weights_only=True: a checkpoint is data, and torch.load's default of
        # unpickling arbitrary objects turns "load a model someone sent me" into
        # "run code someone sent me".
        blob = torch.load(path, map_location=map_location, weights_only=True)
        net = cls(NetConfig(**blob["config"]))
        net.load_state_dict(blob["state"])
        net.eval()
        return net

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Evaluator:
    """Adapts a network to the callable :mod:`connect4.az.mcts` expects.

    Holds the network in eval mode under ``no_grad``, converts boards to tensors
    and answers back to NumPy. Two options worth knowing about:

    ``symmetry`` averages each prediction with its left-right mirror. Connect 4
    is symmetric about the centre column, so the two answers *should* agree, and
    averaging them costs one extra board per evaluation while removing a chunk
    of the network's variance. It also makes a useful diagnostic: if mirrored
    predictions disagree wildly, the network has memorised orientation rather
    than learning shape.

    ``threads`` exists because torch's default is to grab every core, and during
    self-play the cores are better spent on more games in parallel than on
    shaving microseconds off a forward pass over a 6x7 board.
    """

    __slots__ = ("net", "symmetry")

    def __init__(self, net: PolicyValueNet, symmetry: bool = False) -> None:
        self.net = net.eval()
        self.symmetry = symmetry

    @torch.no_grad()
    def __call__(self, positions: Sequence[Position]) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch: returns ``(priors (N, WIDTH), values (N,))``."""
        boards = planes_batch(list(positions))
        if self.symmetry:
            boards = np.concatenate((boards, mirror(boards)))
        logits, values = self.net(torch.from_numpy(boards))
        priors = torch.softmax(logits, dim=-1).numpy()
        values_np = values.numpy()
        if self.symmetry:
            half = len(priors) // 2
            priors = 0.5 * (priors[:half] + mirror_policy(priors[half:]))
            values_np = 0.5 * (values_np[:half] + values_np[half:])
        return priors.astype(np.float32), values_np.astype(np.float32)

    def single(self, position: Position) -> tuple[np.ndarray, float]:
        """One position, for :func:`connect4.az.mcts.run`."""
        priors, values = self([position])
        return priors[0], float(values[0])


#: Torch threads to use by default. Measured on the same 8-core machine, same
#: batch of 128, ch=32/blocks=3: **1 thread costs 76ms, 2 threads cost 10ms**,
#: and everything from 2 to 8 lands within noise of each other. That cliff is
#: much larger than any tuning above it, so the only thing that really matters
#: is not accidentally running single-threaded. Four is a deliberate compromise:
#: past two the returns are flat, and leaving cores free is worth more than a
#: millisecond when several self-play workers share the machine.
DEFAULT_THREADS = 4


def configure_threads(threads: int = DEFAULT_THREADS) -> None:
    """Pin torch's intra-op thread count. See :data:`DEFAULT_THREADS`."""
    torch.set_num_threads(max(1, threads))


def uniform_evaluator(positions: Sequence[Position]) -> tuple[np.ndarray, np.ndarray]:
    """A network-shaped thing that knows nothing.

    Flat priors, zero value -- which turns PUCT into something very close to
    plain UCT with no rollouts, i.e. a search that is barely better than
    guessing. It is the honest baseline: any measured strength above this one is
    strength the network supplied.
    """
    n = len(positions)
    return (
        np.full((n, WIDTH), 1.0 / WIDTH, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
    )
