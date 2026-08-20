"""Tests for the from-scratch MLP.

The central test here is :func:`test_gradients_match_numerical_differentiation`.
A hand-written backward pass is the easiest place in this whole project to be
confidently wrong: a missing transpose or a forgotten division by the batch
size still trains, just worse, and nothing ever raises. Gradient checking
catches all of it -- if the analytic gradient matches a finite-difference
estimate to six significant figures, the calculus is right.

Everything is done in float64 for the gradient checks. In float32 the
subtraction ``loss(w + eps) - loss(w - eps)`` loses most of its significant
digits to rounding, and the check would fail on precision alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from connect4.bitboard import Position
from connect4.dataset import LABELS, N_FEATURES
from connect4.model import (
    MLP,
    NeuralEvaluator,
    cross_entropy,
    relu,
    relu_grad,
    softmax,
    train,
)

# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(0)
    probabilities = softmax(rng.standard_normal((17, 5)))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities > 0).all()


def test_softmax_survives_huge_logits():
    """Without the max-subtraction this overflows to nan."""
    extreme = np.array([[1000.0, 999.0, -1000.0]])
    probabilities = softmax(extreme)
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)


def test_softmax_is_shift_invariant():
    """Adding a constant to every logit must not change the distribution."""
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((4, 3))
    assert np.allclose(softmax(logits), softmax(logits + 7.5))


def test_relu_and_its_gradient():
    z = np.array([-2.0, -0.0, 0.0, 3.0])
    assert np.array_equal(relu(z), [0.0, 0.0, 0.0, 3.0])
    assert np.array_equal(relu_grad(z), [0.0, 0.0, 0.0, 1.0])


def test_cross_entropy_is_zero_for_perfect_predictions():
    probabilities = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert cross_entropy(probabilities, np.array([0, 2])) == pytest.approx(0.0, abs=1e-9)


def test_cross_entropy_penalises_confident_mistakes():
    confident_right = cross_entropy(np.array([[0.98, 0.01, 0.01]]), np.array([0]))
    confident_wrong = cross_entropy(np.array([[0.98, 0.01, 0.01]]), np.array([2]))
    assert confident_wrong > confident_right * 50


def test_cross_entropy_does_not_blow_up_on_zero_probability():
    """A probability that underflows to 0 must not produce inf."""
    value = cross_entropy(np.array([[1.0, 0.0, 0.0]]), np.array([1]))
    assert np.isfinite(value)


# --------------------------------------------------------------------------
# Gradient checking -- the important one
# --------------------------------------------------------------------------


def _to_float64(model: MLP) -> None:
    for layer in model.layers:
        layer.weights = layer.weights.astype(np.float64)
        layer.biases = layer.biases.astype(np.float64)


def _numerical_gradient(model: MLP, x, y, array: np.ndarray, index, weight_decay: float, eps=1e-6):
    """Central-difference estimate of d loss / d array[index].

    Central difference has error O(eps^2) versus O(eps) for a forward
    difference, which buys several extra digits of agreement for free.
    """
    original = array[index]

    array[index] = original + eps
    loss_plus = model.loss(x, y, weight_decay)

    array[index] = original - eps
    loss_minus = model.loss(x, y, weight_decay)

    array[index] = original
    return (loss_plus - loss_minus) / (2 * eps)


@pytest.mark.parametrize("weight_decay", [0.0, 1e-3])
def test_gradients_match_numerical_differentiation(weight_decay):
    """Analytic backprop vs. finite differences, on every layer."""
    rng = np.random.default_rng(7)
    model = MLP(layer_sizes=(6, 8, 5, 3), seed=3)
    _to_float64(model)

    x = rng.standard_normal((11, 6))
    y = rng.integers(0, 3, size=11)

    _, pre_activations, activations = model.forward(x)
    gradients = model.backward(y, pre_activations, activations, weight_decay)

    for layer_index, (layer, (grad_w, grad_b)) in enumerate(zip(model.layers, gradients)):
        # Spot-check a handful of entries per layer; checking all of them is
        # the same test, just slower.
        for _ in range(6):
            i = rng.integers(0, layer.weights.shape[0])
            j = rng.integers(0, layer.weights.shape[1])
            numeric = _numerical_gradient(
                model, x, y, layer.weights, (i, j), weight_decay
            )
            analytic = grad_w[i, j]
            assert numeric == pytest.approx(analytic, rel=1e-5, abs=1e-8), (
                f"weight grad mismatch at layer {layer_index} [{i},{j}]: "
                f"analytic {analytic}, numeric {numeric}"
            )

        for _ in range(3):
            k = rng.integers(0, layer.biases.shape[0])
            numeric = _numerical_gradient(model, x, y, layer.biases, (k,), weight_decay)
            analytic = grad_b[k]
            assert numeric == pytest.approx(analytic, rel=1e-5, abs=1e-8), (
                f"bias grad mismatch at layer {layer_index} [{k}]: "
                f"analytic {analytic}, numeric {numeric}"
            )


def test_gradient_is_averaged_over_the_batch_not_summed():
    """A classic silent bug: forgetting ``/= batch`` in the output delta.

    It does not crash and the model still trains, it just behaves as though the
    learning rate were scaled by the batch size. Doubling the batch with
    duplicated rows must leave the gradient unchanged.
    """
    rng = np.random.default_rng(11)
    model = MLP(layer_sizes=(4, 6, 3), seed=1)
    _to_float64(model)

    x = rng.standard_normal((5, 4))
    y = rng.integers(0, 3, size=5)

    def grads_for(xx, yy):
        _, pre, acts = model.forward(xx)
        return model.backward(yy, pre, acts, 0.0)

    single = grads_for(x, y)
    doubled = grads_for(np.vstack([x, x]), np.concatenate([y, y]))

    for (w1, b1), (w2, b2) in zip(single, doubled):
        assert np.allclose(w1, w2, atol=1e-12)
        assert np.allclose(b1, b2, atol=1e-12)


# --------------------------------------------------------------------------
# Training behaviour
# --------------------------------------------------------------------------


def test_can_overfit_a_tiny_dataset():
    """Sanity check on the whole loop: a model that cannot memorise 32 rows is broken."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 12)).astype(np.float32)
    y = rng.integers(0, 3, size=32)

    model = MLP(layer_sizes=(12, 64, 64, 3), seed=0)
    train(model, (x, y), (x, y), epochs=400, batch_size=32,
          learning_rate=3e-3, weight_decay=0.0, patience=400, verbose=False)

    accuracy = (model.predict(x) == y).mean()
    assert accuracy == 1.0, f"failed to memorise a tiny set (got {accuracy:.2f})"


def test_training_restores_the_best_weights_not_the_last():
    """Early stopping must roll back, otherwise the snapshot is pointless."""
    rng = np.random.default_rng(5)
    x_train = rng.standard_normal((200, 8)).astype(np.float32)
    y_train = rng.integers(0, 3, size=200)
    # Validation labels are independent noise, so val accuracy peaks early and
    # then degrades as the model memorises the training set.
    x_val = rng.standard_normal((100, 8)).astype(np.float32)
    y_val = rng.integers(0, 3, size=100)

    model = MLP(layer_sizes=(8, 64, 3), seed=2)
    history = train(model, (x_train, y_train), (x_val, y_val), epochs=60,
                    batch_size=32, learning_rate=3e-3, patience=60, verbose=False)

    restored = float((model.predict(x_val) == y_val).mean())
    assert restored == pytest.approx(history.best_val_accuracy, abs=1e-9)
    assert history.best_val_accuracy >= max(history.val_accuracy) - 1e-9


def test_save_and_load_roundtrip(tmp_path):
    model = MLP(layer_sizes=(N_FEATURES, 16, 3), seed=4)
    x = np.random.default_rng(0).standard_normal((10, N_FEATURES)).astype(np.float32)
    before = model.predict_proba(x)

    path = tmp_path / "model.npz"
    model.save(path)
    after = MLP.load(path).predict_proba(x)

    assert np.allclose(before, after)


# --------------------------------------------------------------------------
# The evaluator adapter
# --------------------------------------------------------------------------


def test_evaluator_output_is_bounded():
    """Must stay inside [-1, 1] so a guess can never outrank a proof."""
    model = MLP(seed=0)
    evaluator = NeuralEvaluator(model)
    rng = np.random.default_rng(3)

    for _ in range(200):
        pos = Position()
        for _ in range(rng.integers(0, 20)):
            legal = [c for c in range(7) if pos.can_play(c)]
            pos.play(int(rng.choice(legal)))
            if pos.has_won():
                break
        value = evaluator(pos)
        assert -1.0 <= value <= 1.0


def test_evaluator_probabilities_form_a_distribution():
    evaluator = NeuralEvaluator(MLP(seed=0))
    probabilities = evaluator.probabilities(Position.from_moves([3, 3, 4, 4]))
    assert set(probabilities) == set(LABELS)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_evaluator_cache_returns_consistent_values():
    evaluator = NeuralEvaluator(MLP(seed=0))
    pos = Position.from_moves([3, 3, 4])
    assert evaluator(pos) == evaluator(pos)
    assert evaluator(pos) == evaluator(pos.copy())
