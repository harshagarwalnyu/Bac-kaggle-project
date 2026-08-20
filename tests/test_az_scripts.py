"""Tests for the three AlphaZero entry-point scripts.

These are deliberately shallow -- the substance lives in ``connect4.az`` and is
tested there. What is checked here is the layer that a unit test of the package
cannot reach: argument defaults, and the label-to-value mapping that decides
which way round the network learns the game.

The default-value tests are not padding. ``NetConfig`` is a ``slots=True``
dataclass, and on a slotted dataclass ``NetConfig.channels`` is the slot
descriptor rather than the default -- so an argparse default written that way
sails through import and lint and then dies at ``nn.Conv2d`` with
``unsupported operand type(s) for %: 'member_descriptor' and 'int'``, several
seconds into a run. That is exactly the kind of failure worth one cheap
assertion.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="needs the optional 'az' extra")

from scripts import az_arena, az_pretrain, train_az  # noqa: I001

from connect4.az.net import NetConfig
from connect4.az.features import PLANES
from connect4.bitboard import HEIGHT, WIDTH
from connect4.dataset import LABELS


def test_the_training_loop_takes_usable_network_defaults():
    args = train_az.parse_args([])
    assert isinstance(args.channels, int)
    assert isinstance(args.blocks, int)
    assert (args.channels, args.blocks) == (NetConfig().channels, NetConfig().blocks)


def test_the_pretraining_script_takes_usable_network_defaults():
    args = az_pretrain.parse_args([])
    assert isinstance(args.channels, int)
    assert isinstance(args.blocks, int)


def test_the_promotion_gate_defaults_above_a_dead_heat():
    """A gate at or below 0.5 promotes on coin flips and the loop random-walks."""
    assert train_az.parse_args([]).gate > 0.5


def test_the_pretraining_defaults_point_at_the_decompressed_dataset():
    """``data/`` holds the compressed archive; ``data/raw/`` holds the file."""
    assert az_pretrain.parse_args([]).data_dir.name == "raw"


def test_every_dataset_label_maps_to_a_value():
    assert len(az_pretrain.VALUE_OF_LABEL) == len(LABELS)
    assert set(az_pretrain.VALUE_OF_LABEL.values()) == {1.0, 0.0, -1.0}


def test_a_win_for_the_mover_is_positive_and_a_loss_negative():
    """The sign convention, stated once where a reader will look for it.

    Every dataset row is 8 plies deep, so the labelled player is always the one
    to move, and the label needs no flip. Getting this backwards would train a
    value head that is confidently, consistently wrong -- and nothing else in
    the pipeline would complain.
    """
    from connect4.dataset import LABEL_TO_INDEX

    assert az_pretrain.VALUE_OF_LABEL[LABEL_TO_INDEX["win"]] == 1.0
    assert az_pretrain.VALUE_OF_LABEL[LABEL_TO_INDEX["draw"]] == 0.0
    assert az_pretrain.VALUE_OF_LABEL[LABEL_TO_INDEX["loss"]] == -1.0


def test_the_arena_grades_against_every_skill_level_by_default():
    args = az_arena.parse_args(["some.pt"])
    assert args.skills == list(range(az_arena.MAX_SKILL + 1))


def test_the_arena_reports_a_missing_checkpoint_rather_than_raising(capsys):
    assert az_arena.main(["definitely-not-a-checkpoint.pt"]) == 2
    assert "no such checkpoint" in capsys.readouterr().err


def test_the_benchmark_scores_every_named_opponent():
    """Wiring check: a typo in an opponent name silently drops it from the log."""
    from connect4.az.net import Evaluator, PolicyValueNet
    from connect4.az.player import AZPlayer

    player = AZPlayer(Evaluator(PolicyValueNet(NetConfig(channels=4, blocks=1))), simulations=4)
    scores = train_az.benchmark(player, games=2, rng=np.random.default_rng(0))
    assert set(scores) == {"random", "engine2", "engine4", "engine5"}
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_the_checkpoint_round_trip_check_writes_a_loadable_file(tmp_path):
    from connect4.az.net import PolicyValueNet

    path = tmp_path / "champion.pt"
    train_az.check_checkpoint_round_trip(PolicyValueNet(NetConfig(channels=4, blocks=1)), path)
    assert path.exists()
    assert PolicyValueNet.load(path).config.channels == 4


def test_loading_the_value_dataset_yields_planes_and_bounded_values():
    """Runs against the real file if it is already on disk, and skips if not.

    Downloading a dataset from inside a unit test would make the suite depend on
    the UCI archive being up, which is not something this project should assert.
    """
    data_dir = az_pretrain.parse_args([]).data_dir
    if not (data_dir / "connect-4.data").exists():
        pytest.skip("UCI dataset not downloaded; run scripts/az_pretrain.py first")

    boards, values = az_pretrain.load_value_dataset(data_dir, limit=32)
    assert boards.shape == (32, PLANES, HEIGHT, WIDTH)
    assert set(np.unique(values)) <= {-1.0, 0.0, 1.0}
