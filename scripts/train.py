"""Train the dataset brain and report honestly on how good it is.

Run with::

    uv run python -m scripts.train

Three numbers matter, and the script prints all three:

* **65.83%** -- always guess "win". The dataset is skewed enough that this is a
  genuinely hard bar, and a model that does not clear it comfortably has
  learned nothing.
* **Logistic regression** on the same features. If a linear model on the raw
  planes matches the network, the network's depth is decoration.
* **The MLP.**

Beating a baseline is a claim, and a claim needs a control. A test-set number
without those two comparisons is not evidence of anything.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from connect4.dataset import LABELS, build_dataset
from connect4.model import MLP, train

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Rows are true classes, columns predicted. Written out rather than
    imported so the reported numbers have no hidden convention in them."""
    n = len(LABELS)
    matrix = np.zeros((n, n), dtype=np.int64)
    for true, predicted in zip(y_true, y_pred):
        matrix[true, predicted] += 1
    return matrix


def per_class_report(matrix: np.ndarray) -> list[dict[str, float]]:
    """Precision, recall and F1 per class.

    Accuracy alone hides the failure mode this dataset invites: a model can sit
    at 70% while never once predicting "draw", because draws are only 9.6% of
    the rows. Per-class recall makes that visible immediately.
    """
    report = []
    for index, name in enumerate(LABELS):
        true_positive = int(matrix[index, index])
        predicted = int(matrix[:, index].sum())
        actual = int(matrix[index, :].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report.append(
            {"label": name, "support": actual, "precision": precision,
             "recall": recall, "f1": f1}
        )
    return report


def print_report(title: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    accuracy = float((y_true == y_pred).mean())
    matrix = confusion_matrix(y_true, y_pred)
    report = per_class_report(matrix)

    print(f"\n{title}")
    print(f"  accuracy: {accuracy:.4f}")
    print(f"  {'':>6} {'prec':>7} {'recall':>7} {'f1':>7} {'support':>8}")
    for row in report:
        print(f"  {row['label']:>6} {row['precision']:7.3f} {row['recall']:7.3f}"
              f" {row['f1']:7.3f} {row['support']:8d}")

    print("  confusion (rows = truth, cols = predicted):")
    print(f"  {'':>6} " + " ".join(f"{name:>7}" for name in LABELS))
    for name, row in zip(LABELS, matrix):
        print(f"  {name:>6} " + " ".join(f"{v:7d}" for v in row))

    return {"accuracy": accuracy, "per_class": report, "confusion": matrix.tolist()}


def majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    majority = int(np.bincount(y_train).argmax())
    predictions = np.full_like(y_test, majority)
    return print_report(
        f"BASELINE 1 -- always predict '{LABELS[majority]}'", y_test, predictions
    )


def logistic_baseline(splits: dict) -> dict | None:
    """A linear control. Skipped rather than fatal if sklearn is absent."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:  # pragma: no cover
        print("\nBASELINE 2 -- skipped (scikit-learn not installed)")
        return None

    x_train, y_train = splits["train"]
    x_test, y_test = splits["test"]

    model = LogisticRegression(max_iter=2000, n_jobs=-1)
    started = time.perf_counter()
    model.fit(x_train, y_train)
    elapsed = time.perf_counter() - started

    result = print_report("BASELINE 2 -- multinomial logistic regression", y_test,
                          model.predict(x_test))
    result["fit_seconds"] = elapsed
    return result


def main() -> None:
    print("Loading dataset...")
    splits = build_dataset(DATA_DIR)
    for name, (x, y) in splits.items():
        print(f"  {name:>5}: {len(y):6d} rows, class mix "
              f"{np.round(np.bincount(y, minlength=3) / len(y), 4).tolist()}")

    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    x_test, y_test = splits["test"]

    results: dict[str, object] = {
        "majority": majority_baseline(y_train, y_test),
        "logistic": logistic_baseline(splits),
    }

    print("\nTraining the MLP...")
    model = MLP(seed=0)
    started = time.perf_counter()
    history = train(
        model,
        (x_train, y_train),
        (x_val, y_val),
        epochs=120,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-5,
        patience=15,
        seed=0,
    )
    elapsed = time.perf_counter() - started
    print(f"  trained in {elapsed:.1f}s; best epoch {history.best_epoch} "
          f"(val accuracy {history.best_val_accuracy:.4f})")

    mlp_result = print_report("MODEL -- from-scratch MLP", y_test, model.predict(x_test))
    mlp_result["train_seconds"] = elapsed
    mlp_result["best_epoch"] = history.best_epoch
    mlp_result["best_val_accuracy"] = history.best_val_accuracy
    mlp_result["layer_sizes"] = list(model.layer_sizes)
    mlp_result["parameters"] = sum(
        layer.weights.size + layer.biases.size for layer in model.layers
    )
    results["mlp"] = mlp_result

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "evaluator.npz")
    (MODEL_DIR / "training_report.json").write_text(
        json.dumps(
            {
                "results": results,
                "history": {
                    "train_loss": history.train_loss,
                    "val_loss": history.val_loss,
                    "val_accuracy": history.val_accuracy,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    majority_accuracy = results["majority"]["accuracy"]  # type: ignore[index]
    print(f"\nSaved model to {MODEL_DIR / 'evaluator.npz'}")
    print(f"MLP beats the majority baseline by "
          f"{(mlp_result['accuracy'] - majority_accuracy) * 100:+.2f} points.")


if __name__ == "__main__":
    main()
