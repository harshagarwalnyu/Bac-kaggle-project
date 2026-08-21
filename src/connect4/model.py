"""A small multi-layer perceptron, written from scratch in NumPy.

Why not PyTorch
---------------
This network is 98 -> 128 -> 64 -> 3, about 21k parameters, trained on 47k
rows. A framework would add a large dependency and, more to the point, would
hide the part that is actually worth being able to explain: the backward pass.
Everything here -- He initialisation, ReLU, softmax with a numerically stable
log-sum-exp, cross-entropy, Adam, early stopping -- is about 200 lines and each
line has a reason.

The gradients are verified against numerical differentiation in the test suite,
which is the only honest way to claim a hand-written backward pass is correct.

Conventions
-----------
Forward pass for layer ``i``::

    z = a_prev @ W + b
    a = relu(z)          (hidden layers)
    a = softmax(z)       (output layer)

All arrays are ``float32`` and batch-first: ``a`` has shape ``(batch, units)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import cast

import numpy as np

from .bitboard import Position
from .dataset import LABELS, N_FEATURES, encode

# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def relu(z: np.ndarray) -> np.ndarray:
    # numpy's ufunc overloads degrade to `Any` for an unparameterised ndarray
    # input, so the checker cannot tell this is still an array. Narrowing here
    # keeps the honest signature instead of letting `Any` leak into callers.
    return cast(np.ndarray, np.maximum(z, 0.0))


def relu_grad(z: np.ndarray) -> np.ndarray:
    """d relu / dz. Exactly 0 at z == 0 -- the subgradient choice is arbitrary
    and this one keeps dead units dead, which is the conventional pick."""
    return (z > 0.0).astype(z.dtype)


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shifted for numerical stability.

    ``exp`` of a large logit overflows float32 at around 88. Subtracting the
    row max before exponentiating leaves the result unchanged mathematically
    (the shift cancels in the ratio) but bounds every exponent at 0.
    """
    shifted = z - z.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return cast(np.ndarray, exponentiated / exponentiated.sum(axis=1, keepdims=True))


def cross_entropy(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class.

    The clip guards against ``log(0)`` when the network becomes very confident
    and a probability underflows to exactly zero in float32.
    """
    n = targets.shape[0]
    picked = probabilities[np.arange(n), targets]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


# --------------------------------------------------------------------------
# Layers and the network
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Layer:
    weights: np.ndarray
    biases: np.ndarray
    # Adam's first and second moment estimates, one pair per parameter array.
    # Allocated in __post_init__ from the shape of `weights`, which is not
    # known before then. `init=False` says that; the previous `default=None`
    # said the field could hold None, which it never can, and needed a
    # suppression comment on every line to keep the checker quiet about it.
    m_w: np.ndarray = field(init=False)
    v_w: np.ndarray = field(init=False)
    m_b: np.ndarray = field(init=False)
    v_b: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.biases)
        self.v_b = np.zeros_like(self.biases)


class MLP:
    """Fully connected network with ReLU hidden layers and a softmax output."""

    def __init__(
        self,
        layer_sizes: tuple[int, ...] = (N_FEATURES, 128, 64, len(LABELS)),
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.layer_sizes = layer_sizes
        self.layers: list[Layer] = []

        for fan_in, fan_out in pairwise(layer_sizes):
            # He initialisation: variance 2/fan_in. ReLU zeroes half its inputs,
            # halving the variance of each layer's output; the factor of 2
            # compensates, so activations neither vanish nor explode with depth.
            scale = np.sqrt(2.0 / fan_in)
            weights = (rng.standard_normal((fan_in, fan_out)) * scale).astype(np.float32)
            biases = np.zeros(fan_out, dtype=np.float32)
            self.layers.append(Layer(weights=weights, biases=biases))

    # ------------------------------------------------------------- inference

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Return ``(probabilities, pre_activations, activations)``.

        The intermediate lists are what backprop consumes. ``activations[0]``
        is the input itself, which keeps the backward loop uniform.
        """
        activations = [x]
        pre_activations: list[np.ndarray] = []

        for index, layer in enumerate(self.layers):
            z = activations[-1] @ layer.weights + layer.biases
            pre_activations.append(z)
            is_output = index == len(self.layers) - 1
            activations.append(softmax(z) if is_output else relu(z))

        return activations[-1], pre_activations, activations

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    # -------------------------------------------------------------- training

    def backward(
        self,
        targets: np.ndarray,
        pre_activations: list[np.ndarray],
        activations: list[np.ndarray],
        weight_decay: float,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Gradients of the mean loss w.r.t. every weight and bias.

        The output layer's delta is the one piece of real calculus worth
        stating: for softmax composed with cross-entropy, the gradient of the
        loss with respect to the *logits* collapses to ``probs - one_hot``.
        The softmax Jacobian and the log's derivative cancel almost entirely,
        which is exactly why this pairing is the standard one -- no separate
        softmax-backward step is ever needed.
        """
        batch = targets.shape[0]

        probabilities = activations[-1]
        delta = probabilities.copy()
        delta[np.arange(batch), targets] -= 1.0
        delta /= batch  # because the loss is a *mean*, not a sum

        gradients: list[tuple[np.ndarray, np.ndarray]] = [None] * len(self.layers)  # type: ignore

        for index in reversed(range(len(self.layers))):
            layer = self.layers[index]
            a_prev = activations[index]

            grad_w = a_prev.T @ delta
            grad_b = delta.sum(axis=0)

            # L2 regularisation, applied to weights only. Biases are left
            # alone by convention: penalising them just shifts the decision
            # boundary toward the origin without reducing model capacity.
            if weight_decay:
                grad_w = grad_w + weight_decay * layer.weights

            gradients[index] = (grad_w, grad_b)

            if index > 0:
                # Propagate through the weights, then through the ReLU.
                delta = (delta @ layer.weights.T) * relu_grad(pre_activations[index - 1])

        return gradients

    def apply_adam(
        self,
        gradients: list[tuple[np.ndarray, np.ndarray]],
        step: int,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """One Adam update.

        Adam keeps a running mean (``m``) and running uncentred variance
        (``v``) of each gradient, then steps by ``m / sqrt(v)``. Dividing by
        the gradient's own scale means every parameter gets a comparable step
        size regardless of how large its gradients happen to be -- which
        matters here because the raw occupancy features are 0/1 while the
        engineered features are fractions, so their gradients differ by an
        order of magnitude.

        The bias correction terms undo the fact that ``m`` and ``v`` start at
        zero and are therefore biased toward zero for the first few hundred
        steps.
        """
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step

        for layer, (grad_w, grad_b) in zip(self.layers, gradients):
            layer.m_w = beta1 * layer.m_w + (1 - beta1) * grad_w
            layer.v_w = beta2 * layer.v_w + (1 - beta2) * np.square(grad_w)
            m_hat = layer.m_w / bias_correction1
            v_hat = layer.v_w / bias_correction2
            layer.weights -= (learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)).astype(np.float32)

            layer.m_b = beta1 * layer.m_b + (1 - beta1) * grad_b
            layer.v_b = beta2 * layer.v_b + (1 - beta2) * np.square(grad_b)
            m_hat_b = layer.m_b / bias_correction1
            v_hat_b = layer.v_b / bias_correction2
            layer.biases -= (learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)).astype(np.float32)

    def loss(self, x: np.ndarray, y: np.ndarray, weight_decay: float = 0.0) -> float:
        """Full objective, including the regularisation term.

        Gradient checking needs the *same* quantity that ``backward``
        differentiates, so the L2 penalty has to be included here or the
        numerical and analytic gradients will disagree by exactly the decay.
        """
        probabilities = self.predict_proba(x)
        total = cross_entropy(probabilities, y)
        if weight_decay:
            total += 0.5 * weight_decay * sum(
                float(np.sum(np.square(layer.weights))) for layer in self.layers
            )
        return total

    # ------------------------------------------------------- persistence

    def save(self, path: Path) -> None:
        arrays: dict[str, np.ndarray] = {"layer_sizes": np.array(self.layer_sizes)}
        for index, layer in enumerate(self.layers):
            arrays[f"w{index}"] = layer.weights
            arrays[f"b{index}"] = layer.biases
        np.savez_compressed(path, **arrays)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: Path) -> MLP:
        blob = np.load(path)
        sizes = tuple(int(v) for v in blob["layer_sizes"])
        model = cls(layer_sizes=sizes)
        for index, layer in enumerate(model.layers):
            layer.weights = blob[f"w{index}"].astype(np.float32)
            layer.biases = blob[f"b{index}"].astype(np.float32)
        return model


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_accuracy: float = 0.0


def train(
    model: MLP,
    train_xy: tuple[np.ndarray, np.ndarray],
    val_xy: tuple[np.ndarray, np.ndarray],
    epochs: int = 60,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 10,
    seed: int = 0,
    verbose: bool = True,
) -> TrainingHistory:
    """Mini-batch Adam with early stopping on validation accuracy.

    Early stopping keeps a snapshot of the best-scoring weights and restores
    them at the end. Without it the network happily overfits 47k rows -- train
    loss keeps falling while validation accuracy peaks and then decays, and the
    final epoch is not the best one.
    """
    x_train, y_train = train_xy
    x_val, y_val = val_xy

    rng = np.random.default_rng(seed)
    history = TrainingHistory()
    step = 0

    best_snapshot = [(layer.weights.copy(), layer.biases.copy()) for layer in model.layers]
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x_train))
        epoch_loss = 0.0
        batches = 0

        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            xb, yb = x_train[batch_idx], y_train[batch_idx]

            probabilities, pre_activations, activations = model.forward(xb)
            gradients = model.backward(yb, pre_activations, activations, weight_decay)

            step += 1
            model.apply_adam(gradients, step, learning_rate)

            epoch_loss += cross_entropy(probabilities, yb)
            batches += 1

        val_probabilities = model.predict_proba(x_val)
        val_loss = cross_entropy(val_probabilities, y_val)
        val_accuracy = float((val_probabilities.argmax(axis=1) == y_val).mean())

        history.train_loss.append(epoch_loss / max(batches, 1))
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_accuracy)

        if val_accuracy > history.best_val_accuracy:
            history.best_val_accuracy = val_accuracy
            history.best_epoch = epoch
            best_snapshot = [(layer.weights.copy(), layer.biases.copy()) for layer in model.layers]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"  epoch {epoch:3d}  train_loss {history.train_loss[-1]:.4f}"
                f"  val_loss {val_loss:.4f}  val_acc {val_accuracy:.4f}"
            )

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"  early stop: no improvement for {patience} epochs")
            break

    for layer, (weights, biases) in zip(model.layers, best_snapshot):
        layer.weights, layer.biases = weights, biases

    return history


# --------------------------------------------------------------------------
# The engine seam
# --------------------------------------------------------------------------


class NeuralEvaluator:
    """Adapts a trained :class:`MLP` into an evaluator the engine can use.

    The network outputs a distribution over (loss, draw, win) *for the player
    to move*. The engine wants a single scalar in roughly [-1, 1], so we take
    the expected outcome::

        value = P(win) - P(loss)

    Draw contributes zero, which is exactly right. The result is bounded by
    construction, so it can never stray into the proven-score range and start
    outranking a real proof.

    Caveat worth stating plainly: every training position had exactly 8 stones.
    Asking this network about a 30-ply position is extrapolation. The engine
    still only ever uses it at leaf nodes it could not resolve exactly, and the
    UI labels its output as an opinion rather than a verdict.
    """

    def __init__(self, model: MLP) -> None:
        self.model = model
        self._cache: dict[int, float] = {}

    def __call__(self, pos: Position) -> float:
        # The search revisits transpositions constantly and a forward pass is
        # far more expensive than a dict lookup, so memoise on the position key.
        key = pos.key()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        features = encode(pos).reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]
        value = float(probabilities[LABELS.index("win")] - probabilities[LABELS.index("loss")])

        self._cache[key] = value
        return value

    def probabilities(self, pos: Position) -> dict[str, float]:
        """Full distribution, for the UI's dataset-brain panel."""
        features = encode(pos).reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]
        return {name: float(p) for name, p in zip(LABELS, probabilities)}
