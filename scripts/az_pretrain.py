"""Warm-start the value head on solver-exact labels before self-play begins.

    uv run python -m scripts.az_pretrain --epochs 12

Self-play from random weights spends its first hours discovering that four in
a row is good. That is a fact the project already has on disk: the UCI
``connect-4`` database is 67,557 8-ply positions labelled with the
*game-theoretic* result under perfect play. Fitting the value head to those
labels first hands the search a leaf evaluator that is already better than
noise on iteration one.

**Only the value head is trained here, and the reason is the labels.** The file
says who wins with perfect play; it says nothing whatsoever about which move to
play, so there is no policy target to fit. Training the policy head on anything
derived from these labels would mean inventing data, and the point of using a
real dataset is not to do that. The policy head therefore stays at its random
initialisation and learns entirely from search visit counts, exactly as it
would without this step.

**The trunk is shared, so the policy head is not left untouched.** Gradients
from the value loss reshape the features both heads read. That is the actual
prize -- the tower learns what a threat looks like from exact labels, and the
policy head starts from a representation that already knows.

**What this step cannot do.** Every row is exactly 8 plies deep. A value head
fitted only to these will be confident and wrong about endgames, which is why
the loss reported on a held-out split here is a sanity check and not a measure
of playing strength. ``scripts/az_arena.py`` measures that, by playing games.

The warm start is an accelerant, not a dependency: ``scripts/train_az.py`` runs
perfectly well from random weights and will simply take longer to get going.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from connect4.az.features import planes_batch
from connect4.az.net import NetConfig, PolicyValueNet, configure_threads
from connect4.az.train import supervised_batches
from connect4.bitboard import WIDTH
from connect4.dataset import LABEL_TO_INDEX, ensure_raw_data, load_samples

#: Dataset label -> value target, from the point of view of the side to move.
#: Every row is 8 plies deep, so ``x`` is always to move and the dataset's
#: "outcome for x" is already mover-relative. No sign flip anywhere, which is
#: the single most likely place for a silent bug in this whole file.
VALUE_OF_LABEL = {
    LABEL_TO_INDEX["win"]: 1.0,
    LABEL_TO_INDEX["draw"]: 0.0,
    LABEL_TO_INDEX["loss"]: -1.0,
}

#: ``NetConfig`` uses ``slots=True``, and on a slotted dataclass the class
#: attribute is the slot descriptor, not the default value -- ``NetConfig.channels``
#: is a ``member_descriptor``. An instance is the only way to read the defaults.
NET_DEFAULTS = NetConfig()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # ``data/raw`` is where scripts/train.py already keeps the decompressed UCI
    # file. Pointing somewhere else makes ensure_raw_data fetch a second copy.
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "raw")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/az/champion.pt"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="stop after this many epochs without a better validation loss (0 disables)",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=NET_DEFAULTS.channels)
    parser.add_argument("--blocks", type=int, default=NET_DEFAULTS.blocks)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="use only the first N rows (testing)")
    return parser.parse_args(argv)


def load_value_dataset(
    data_dir: Path, limit: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(planes, values)`` for the whole UCI file.

    Mirroring is *not* applied here. ``connect4.az.train`` mirrors a random half
    of every batch as it trains, so augmenting on disk as well would double the
    memory for no extra variety.
    """
    path = ensure_raw_data(data_dir)
    samples = load_samples(path)
    if limit:
        samples = samples[:limit]
    boards = planes_batch([s.position for s in samples])
    values = np.array([VALUE_OF_LABEL[s.label] for s in samples], dtype=np.float32)
    return boards, values


def split_problem(train: int, val: int, batch_size: int) -> str | None:
    """Why this train/validation split cannot be trained on, or ``None``.

    Both failures are silent rather than loud, which is why they are worth
    catching here. An empty validation split makes ``baseline`` the mean of an
    empty array -- ``nan`` -- and every subsequent ``val_loss < best_loss``
    comparison false, so the run trains to completion and writes nothing. A
    training split that yields no batches divides by a zero count instead.
    Both are reachable from the documented ``--limit``: at the default
    validation fraction ``--limit 5`` empties the validation split, and
    ``--limit 1`` empties the training one.
    """
    if val < 1:
        return (
            f"the validation split is empty: {train + val} positions at this "
            f"--val-fraction rounds down to zero. Raise --limit or --val-fraction."
        )
    if batch_size < 2:
        # A chunk of one is dropped by `supervised_batches` -- BatchNorm cannot
        # compute a variance from a single row -- so a batch size of one drops
        # every chunk.
        return f"--batch-size must be at least 2, got {batch_size}"
    if train < 2:
        return (
            f"the training split holds {train} position(s), and a batch of one "
            f"is dropped because BatchNorm needs at least two. Raise --limit."
        )
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_threads(args.threads)
    rng = np.random.default_rng(args.seed)

    boards, values = load_value_dataset(args.data_dir, args.limit)
    print(f"{len(boards):,} positions, mean value {values.mean():+.3f}")

    order = rng.permutation(len(boards))
    cut = int(len(order) * args.val_fraction)
    problem = split_problem(train=len(order) - cut, val=cut, batch_size=args.batch_size)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    val_index, train_index = order[:cut], order[cut:]
    train_boards, train_values = boards[train_index], values[train_index]
    val_boards = torch.from_numpy(boards[val_index])
    val_values = torch.from_numpy(values[val_index])

    net = PolicyValueNet(NetConfig(channels=args.channels, blocks=args.blocks))
    print(f"{net.parameter_count():,} parameters")
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # The policy head has no target here, so its parameters would only ever be
    # pushed by weight decay -- shrinking a random initialisation toward zero
    # for no reason. Freezing it leaves it as initialised for self-play to
    # train properly.
    for parameter in net.policy_head.parameters():
        parameter.requires_grad_(False)

    # A constant predictor is the bar to clear. If the network cannot beat the
    # variance of the labels it has learned nothing, and a mean-squared error
    # in the abstract does not tell you which of those happened.
    baseline = float(np.mean((values[val_index] - train_values.mean()) ** 2))
    print(f"baseline (predict the mean) val MSE {baseline:.4f}")

    # 67,557 rows and a 60,000-parameter network start overfitting within a
    # handful of epochs, so the checkpoint that gets written is the best one
    # seen rather than the last one trained.
    #
    # ``--patience`` defaults to five rather than the usual two or three because
    # the validation curve here is genuinely bouncy: on the run this was tuned
    # against it read 0.29, 0.23, 0.27, 0.32, 0.24, 0.33, 0.20 -- epoch seven
    # was the best of the lot, and a patience of three would have stopped at
    # epoch five and thrown it away.
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        net.train()
        # ``requires_grad_(False)`` stops the gradients but not BatchNorm's
        # running statistics, which train mode updates regardless. Putting the
        # head in eval mode is what makes "frozen" actually true.
        net.policy_head.eval()
        total = count = 0.0
        # Policy targets are unused; a uniform placeholder keeps the batching
        # helper's signature honest rather than forking it.
        placeholder = np.full((len(train_boards), WIDTH), 1.0 / WIDTH, dtype=np.float32)
        for batch_boards, _, batch_values in supervised_batches(
            train_boards, placeholder, train_values, args.batch_size, rng
        ):
            _, predicted = net(torch.from_numpy(batch_boards))
            loss = torch.mean((predicted - torch.from_numpy(batch_values)) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_values)
            count += len(batch_values)

        net.eval()
        with torch.no_grad():
            _, predicted = net(val_boards)
            val_loss = float(torch.mean((predicted - val_values) ** 2))
            # The sign is what the search actually consumes: it decides which
            # branch looks better. A model can have a respectable MSE and still
            # get the sign wrong on a third of positions. Drawn rows are
            # excluded -- their target sign is zero, which a tanh output hits
            # with probability zero, so including them would just cap the
            # metric at the draw rate and hide the trend.
            decisive = val_values != 0
            agree = float(
                torch.mean(
                    (
                        torch.sign(predicted[decisive]) == torch.sign(val_values[decisive])
                    ).float()
                )
            )
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch = val_loss, epoch
            # Unfreeze around the save so the checkpoint on disk is a normal,
            # fully trainable network rather than one that silently ignores its
            # policy head when self-play picks it up.
            for parameter in net.policy_head.parameters():
                parameter.requires_grad_(True)
            net.save(args.out)
            for parameter in net.policy_head.parameters():
                parameter.requires_grad_(False)

        print(
            f"epoch {epoch:3d}  train {total / count:.4f}  val {val_loss:.4f}  "
            f"sign agreement {agree:.1%}  {time.perf_counter() - started:.0f}s"
            f"{'  <- saved' if improved else ''}",
            flush=True,
        )
        if args.patience and epoch - best_epoch >= args.patience:
            print(f"no improvement in {args.patience} epochs; stopping")
            break

    if best_epoch == 0:
        # Reachable: `improved` compares against `inf`, so this only happens
        # when every epoch produced a non-finite loss. Saying "saved to ..."
        # here would name a file that does not exist, or worse, an older one
        # from a previous run that this run did not touch.
        print("no epoch improved on the initial loss; nothing was saved.", file=sys.stderr)
        return 1

    print(
        f"best epoch {best_epoch} at val MSE {best_loss:.4f} "
        f"(baseline {baseline:.4f}), saved to {args.out}."
    )
    print("Run scripts/train_az.py --resume to continue from it.")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
