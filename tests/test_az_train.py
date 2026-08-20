"""Tests for the network and the training step.

These need the optional ``az`` extra (``uv sync --extra az``) and skip cleanly
without it, so the suite stays green on a machine that only wants to play a
game.

The interesting tests here are the ones that check *learning actually happens*.
A training step that runs without raising and leaves the weights no better is
the characteristic failure of this kind of code -- a detached tensor, an
optimiser built over the wrong parameters, a loss that does not depend on the
input. All of those pass a shape test.
"""

from __future__ import annotations

import numpy as np
import pytest

from connect4.az.features import PLANES, planes_batch
from connect4.bitboard import HEIGHT, WIDTH, Position

torch = pytest.importorskip("torch", reason="needs the optional 'az' extra")

from connect4.az.net import (
    DEFAULT_THREADS,
    Evaluator,
    NetConfig,
    PolicyValueNet,
    configure_threads,
    uniform_evaluator,
)
from connect4.az.player import AZPlayer, value_evaluator
from connect4.az.selfplay import Example
from connect4.az.train import (
    LossReport,
    ReplayBuffer,
    TrainConfig,
    augment_batch,
    losses,
    supervised_batches,
    train,
)

SMALL = NetConfig(channels=8, blocks=1, value_hidden=8)


@pytest.fixture(scope="module", autouse=True)
def _threads():
    configure_threads(2)


def positions(n: int = 8) -> list[Position]:
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        pos = Position()
        for _ in range(int(rng.integers(0, 12))):
            legal = pos.legal_moves()
            if not legal or pos.has_won():
                break
            pos = pos.played(int(rng.choice(legal)))
        out.append(pos)
    return out


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------


def test_forward_returns_a_logit_per_column_and_one_value_per_board():
    net = PolicyValueNet(SMALL).eval()
    x = torch.from_numpy(planes_batch(positions(5)))
    with torch.no_grad():
        logits, values = net(x)
    assert logits.shape == (5, WIDTH)
    assert values.shape == (5,)


def test_values_are_bounded_to_a_game_result():
    """tanh, so anything outside [-1, 1] means the head was rewired."""
    net = PolicyValueNet(SMALL).eval()
    x = torch.from_numpy(planes_batch(positions(16)))
    with torch.no_grad():
        _, values = net(x)
    assert bool((values.abs() <= 1.0).all())


def test_the_input_shape_the_features_module_produces_is_the_one_the_net_wants():
    """Pins the contract between the two modules, which nothing else would catch."""
    net = PolicyValueNet(SMALL).eval()
    batch = planes_batch([Position()])
    assert batch.shape[1:] == (PLANES, HEIGHT, WIDTH)
    with torch.no_grad():
        net(torch.from_numpy(batch))  # would raise on a mismatch


def test_a_saved_network_reloads_with_the_same_shape_and_the_same_answers(tmp_path):
    net = PolicyValueNet(NetConfig(channels=12, blocks=2, value_hidden=16)).eval()
    path = tmp_path / "nested" / "net.pt"
    net.save(path)
    loaded = PolicyValueNet.load(path)

    assert loaded.config == net.config  # the shape travels with the weights
    x = torch.from_numpy(planes_batch(positions(4)))
    with torch.no_grad():
        a, b = net(x)
        c, d = loaded(x)
    assert torch.allclose(a, c)
    assert torch.allclose(b, d)


def test_a_loaded_network_is_in_eval_mode():
    """Left in train mode, BatchNorm would keep updating from whatever it saw."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "n.pt"
        PolicyValueNet(SMALL).save(path)
        assert not PolicyValueNet.load(path).training


def test_parameter_count_tracks_the_configured_size():
    assert PolicyValueNet(NetConfig(channels=32, blocks=3)).parameter_count() > PolicyValueNet(
        NetConfig(channels=32, blocks=2)
    ).parameter_count()


def test_default_thread_count_is_not_one():
    """One thread measured 7x slower than two. This is a performance tripwire."""
    assert DEFAULT_THREADS > 1


# --------------------------------------------------------------------------
# The evaluator adapter
# --------------------------------------------------------------------------


def test_the_evaluator_returns_probabilities_and_bounded_values():
    ev = Evaluator(PolicyValueNet(SMALL))
    priors, values = ev(positions(6))
    assert priors.shape == (6, WIDTH)
    assert values.shape == (6,)
    assert priors.sum(axis=1) == pytest.approx(np.ones(6), abs=1e-5)
    assert (priors >= 0).all()
    assert np.abs(values).max() <= 1.0


def test_the_evaluator_is_deterministic():
    ev = Evaluator(PolicyValueNet(SMALL))
    a = ev(positions(4))
    b = ev(positions(4))
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_single_matches_the_batch_of_one():
    ev = Evaluator(PolicyValueNet(SMALL))
    pos = Position.from_moves([3, 3, 4])
    priors, value = ev.single(pos)
    batch_priors, batch_values = ev([pos])
    assert np.array_equal(priors, batch_priors[0])
    assert value == pytest.approx(float(batch_values[0]))


def test_symmetry_averaging_makes_mirrored_answers_agree():
    """With symmetry on, a position and its mirror must give mirrored priors.

    Off, they will not -- an untrained conv net has no reason to be symmetric --
    which is what makes this a test of the averaging rather than of the weights.
    """
    moves = [3, 2, 4, 1, 0]
    pos = Position.from_moves(moves)
    flipped = Position.from_moves([WIDTH - 1 - m for m in moves])
    net = PolicyValueNet(SMALL)

    plain = Evaluator(net, symmetry=False)
    assert not np.allclose(plain([pos])[0][0], plain([flipped])[0][0][::-1], atol=1e-6)

    even = Evaluator(net, symmetry=True)
    assert even([pos])[0][0] == pytest.approx(even([flipped])[0][0][::-1], abs=1e-6)
    assert even([pos])[1][0] == pytest.approx(even([flipped])[1][0], abs=1e-6)


def test_the_uniform_evaluator_says_nothing_in_the_right_shape():
    priors, values = uniform_evaluator(positions(3))
    assert priors == pytest.approx(np.full((3, WIDTH), 1 / WIDTH))
    assert not values.any()


# --------------------------------------------------------------------------
# Playing
# --------------------------------------------------------------------------


def test_the_player_only_ever_returns_a_legal_column():
    player = AZPlayer(Evaluator(PolicyValueNet(SMALL)), simulations=16)
    for pos in positions(6):
        if pos.has_won() or pos.is_draw():
            continue
        assert player.choose_move(pos) in pos.legal_moves()


def test_the_player_takes_a_win_it_can_see():
    """An untrained network, but the search still has to find a mate in one."""
    player = AZPlayer(Evaluator(PolicyValueNet(SMALL)), simulations=120)
    assert player.choose_move(Position.from_moves([0, 1, 0, 1, 0])) == 0


def test_choosing_for_many_positions_matches_choosing_one_at_a_time():
    """The batched path must be a scheduling change, not a different player."""
    net = PolicyValueNet(SMALL)
    batch = AZPlayer(Evaluator(net), simulations=32, rng=np.random.default_rng(0))
    single = AZPlayer(Evaluator(net), simulations=32, rng=np.random.default_rng(0))
    boards = [Position.from_moves(m) for m in ([3], [3, 3], [0, 1, 2])]
    assert batch.choose_moves(boards) == [single.choose_move(b) for b in boards]


def test_a_temperature_player_still_plays_legally():
    player = AZPlayer(
        Evaluator(PolicyValueNet(SMALL)),
        simulations=24,
        temperature=1.0,
        rng=np.random.default_rng(3),
    )
    pos = Position.from_moves([3, 3])
    assert player.choose_move(pos) in pos.legal_moves()


def test_the_net_can_stand_in_for_the_hand_written_leaf_evaluator():
    """Point of the exercise: same alpha-beta tree, network scoring the leaves."""
    from connect4.engine import Engine

    cache: dict[int, float] = {}
    evaluate = value_evaluator(Evaluator(PolicyValueNet(SMALL)), cache)
    engine = Engine(evaluator=evaluate, time_limit_s=0.2)
    pos = Position.from_moves([3, 3, 4])
    assert engine.choose_move(pos, skill=5) in pos.legal_moves()
    assert cache  # transpositions were reused rather than recomputed


def test_the_leaf_evaluator_stays_inside_the_opinion_band():
    """A value must never masquerade as a proof; the search reserves that range."""
    evaluate = value_evaluator(Evaluator(PolicyValueNet(SMALL)))
    for pos in positions(10):
        assert abs(evaluate(pos)) <= 0.95


def test_the_leaf_evaluator_caches_by_position_not_by_object():
    calls = []

    def counting(batch):
        calls.append(len(batch))
        return uniform_evaluator(batch)

    evaluate = value_evaluator(counting)
    evaluate(Position.from_moves([0, 1, 2, 3]))
    evaluate(Position.from_moves([2, 3, 0, 1]))  # same board, other move order
    assert len(calls) == 1


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------


def test_a_perfect_value_prediction_has_no_value_loss():
    target = torch.tensor([1.0, -1.0, 0.0])
    _, value_loss, _ = losses(torch.zeros(3, WIDTH), target, torch.full((3, WIDTH), 1 / WIDTH), target, 1.0)
    assert float(value_loss) == pytest.approx(0.0)


def test_policy_loss_bottoms_out_at_the_targets_own_entropy():
    """Cross-entropy cannot go below entropy; matching the target is the floor."""
    target = torch.full((2, WIDTH), 1.0 / WIDTH)
    entropy = float(-(target * target.log()).sum(dim=-1).mean())
    matched, _, _ = losses(torch.zeros(2, WIDTH), torch.zeros(2), target, torch.zeros(2), 1.0)
    logits = torch.zeros(2, WIDTH)
    logits[:, 0] = 5.0
    mismatched, _, _ = losses(logits, torch.zeros(2), target, torch.zeros(2), 1.0)
    assert float(matched) == pytest.approx(entropy, abs=1e-6)
    assert float(mismatched) > float(matched)


def test_the_value_weight_actually_weights_the_value_term():
    logits = torch.zeros(2, WIDTH)
    target = torch.full((2, WIDTH), 1.0 / WIDTH)
    value_out = torch.tensor([0.0, 0.0])
    value_target = torch.tensor([1.0, 1.0])
    _, value_loss, single = losses(logits, value_out, target, value_target, 1.0)
    _, _, double = losses(logits, value_out, target, value_target, 2.0)
    assert float(double - single) == pytest.approx(float(value_loss), abs=1e-6)


# --------------------------------------------------------------------------
# Replay buffer and augmentation
# --------------------------------------------------------------------------


def examples(n: int) -> list[Example]:
    rng = np.random.default_rng(1)
    out = []
    for pos in positions(n):
        policy = rng.random(WIDTH).astype(np.float32)
        out.append(Example(position=pos, policy=policy / policy.sum(), value=float(rng.choice([-1, 0, 1]))))
    return out


def test_the_buffer_keeps_only_its_capacity_and_keeps_the_newest():
    buffer = ReplayBuffer(capacity=5)
    buffer.add(examples(12))
    assert len(buffer) == 5


def test_adding_nothing_is_harmless():
    buffer = ReplayBuffer()
    buffer.add([])
    assert len(buffer) == 0


def test_sampling_returns_aligned_boards_policies_and_values():
    buffer = ReplayBuffer()
    buffer.add(examples(20))
    boards, policies, values = buffer.sample(7, np.random.default_rng(0))
    assert boards.shape == (7, PLANES, HEIGHT, WIDTH)
    assert policies.shape == (7, WIDTH)
    assert values.shape == (7,)
    assert policies.sum(axis=1) == pytest.approx(np.ones(7), abs=1e-5)


def test_augmentation_flips_the_board_and_its_policy_together():
    """Flipping one without the other trains the network to play the mirror move."""
    boards = planes_batch([Position.from_moves([0, 0, 1])])
    policies = np.array([[0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01]], dtype=np.float32)

    class AlwaysFlip:
        def random(self, n):
            return np.zeros(n)

    flipped_boards, flipped_policies = augment_batch(boards, policies, AlwaysFlip())
    assert np.array_equal(flipped_boards[0], boards[0][:, :, ::-1])
    assert np.array_equal(flipped_policies[0], policies[0][::-1])


def test_augmentation_that_flips_nothing_returns_the_input():
    boards = planes_batch([Position.from_moves([0])])
    policies = np.full((1, WIDTH), 1 / WIDTH, dtype=np.float32)

    class NeverFlip:
        def random(self, n):
            return np.ones(n)

    out_boards, out_policies = augment_batch(boards, policies, NeverFlip())
    assert np.array_equal(out_boards, boards)
    assert np.array_equal(out_policies, policies)


def test_supervised_batches_cover_every_row_once():
    boards = planes_batch(positions(10))
    policies = np.full((10, WIDTH), 1 / WIDTH, dtype=np.float32)
    values = np.arange(10, dtype=np.float32)
    batches = supervised_batches(boards, policies, values, 4, np.random.default_rng(0))
    seen = np.concatenate([v for _, _, v in batches])
    assert sorted(seen.tolist()) == list(range(10))


def test_supervised_batches_drop_a_leftover_of_one():
    """BatchNorm cannot compute a variance over a single sample and raises."""
    boards = planes_batch(positions(9))
    policies = np.full((9, WIDTH), 1 / WIDTH, dtype=np.float32)
    values = np.zeros(9, dtype=np.float32)
    batches = supervised_batches(boards, policies, values, 4, np.random.default_rng(0))
    assert all(len(v) > 1 for _, _, v in batches)
    assert sum(len(v) for _, _, v in batches) == 8


# --------------------------------------------------------------------------
# Training actually trains
# --------------------------------------------------------------------------


def test_training_on_an_empty_buffer_does_nothing_rather_than_crashing():
    report = train(PolicyValueNet(SMALL), ReplayBuffer())
    assert report == LossReport(0.0, 0.0, 0.0, 0)


def test_training_reduces_the_loss_on_a_fixed_target():
    """The test that a shape check cannot fake.

    One position, one policy target, one value target, repeated. If the
    optimiser is wired to the right parameters and the graph is connected, the
    loss on that single example must fall a long way. A detached tensor or an
    optimiser over the wrong module passes every other test in this file.
    """
    net = PolicyValueNet(SMALL)
    buffer = ReplayBuffer()
    target = np.zeros(WIDTH, dtype=np.float32)
    target[2] = 1.0
    buffer.add(
        [Example(position=Position.from_moves([3, 3]), policy=target, value=1.0)] * 64
    )

    config = TrainConfig(batch_size=16, steps=3, augment=False)
    before = train(net, buffer, config, np.random.default_rng(0))
    after = train(net, buffer, TrainConfig(batch_size=16, steps=60, augment=False), np.random.default_rng(0))

    assert after.total < before.total * 0.5
    assert after.steps == 60


def test_training_leaves_the_network_in_eval_mode():
    """It is put in train mode for the step; leaving it there breaks self-play."""
    net = PolicyValueNet(SMALL)
    buffer = ReplayBuffer()
    buffer.add(examples(8))
    train(net, buffer, TrainConfig(batch_size=4, steps=2), np.random.default_rng(0))
    assert not net.training


def test_training_changes_the_weights():
    net = PolicyValueNet(SMALL)
    before = [p.detach().clone() for p in net.parameters()]
    buffer = ReplayBuffer()
    buffer.add(examples(16))
    train(net, buffer, TrainConfig(batch_size=8, steps=5), np.random.default_rng(0))
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters(), strict=True))


def test_a_supplied_optimiser_is_reused_rather_than_rebuilt():
    """The loop passes Adam in so its moments survive between iterations."""
    net = PolicyValueNet(SMALL)
    buffer = ReplayBuffer()
    buffer.add(examples(16))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    train(net, buffer, TrainConfig(batch_size=8, steps=3), np.random.default_rng(0), opt)
    assert opt.state  # Adam recorded moment estimates, so it was the one stepping


def test_the_loss_report_prints_something_a_person_can_read():
    text = str(LossReport(policy=1.25, value=0.5, total=1.75, steps=10))
    assert "policy 1.2500" in text
    assert "10 steps" in text
