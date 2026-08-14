"""Loading and feature-encoding the UCI / Kaggle ``connect-4`` dataset.

What this dataset actually is
-----------------------------
67,557 rows, one per *legal 8-ply position in which neither player has won and
the next move is not forced*. The label is the game-theoretic outcome for
player ``x`` under perfect play by both sides: ``win`` (65.83%), ``loss``
(24.62%), ``draw`` (9.55%).

Three consequences that shape everything downstream, and that are easy to get
wrong:

1. **It is not a dataset of games, it is a dataset of one specific depth.**
   Every position has exactly 8 stones. A model trained on it has seen nothing
   of the opening and nothing of the endgame. Expecting it to evaluate a
   30-ply position is extrapolation, not inference -- and the validation
   report measures exactly how badly that goes.

2. **The labels are perfect-play outcomes, not observed results.** They come
   from John Tromp's solver, so they are ground truth. A model that fits them
   is learning to approximate a solver, which is why it makes sense to compare
   it against one.

3. **8 plies means 4 stones each, so ``x`` is always to move.** The label is
   therefore always from the point of view of the player to move -- the same
   convention the search engine uses internally. No sign flip is needed
   anywhere, which removes the single most likely source of a silent bug.

Feature design
--------------
Two blocks, concatenated:

*Raw occupancy* (84 features) -- one binary plane for the mover's stones and
one for the opponent's, 42 cells each. The "empty" plane is deliberately
omitted: it is exactly ``1 - mover - opponent``, so it adds no information a
network cannot derive, and it would inflate the input by 50%.

*Engineered* (14 features) -- domain knowledge the raw planes make hard to
learn from 67k examples. The interesting ones are the odd/even threat counts.
Connect 4 strategy has a parity theorem: with both players filling columns,
the first player tends to win on *odd* rows and the second on *even* rows, so
a threat's row parity matters more than the threat count itself. Handing that
to the model directly is worth far more than another hidden layer.
"""

from __future__ import annotations

import gzip
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bitboard import HEIGHT, WIDTH, Position, popcount

# Where the label sits, and what each label means for the player to move.
LABELS = ("loss", "draw", "win")
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABELS)}

UCI_URL = "https://archive.ics.uci.edu/static/public/26/connect+4.zip"

RAW_FEATURES = 2 * WIDTH * HEIGHT  # 84
ENGINEERED_FEATURES = 14
N_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES  # 98

_COL_STRIDE = HEIGHT + 1


# --------------------------------------------------------------------------
# Acquiring the file
# --------------------------------------------------------------------------


def ensure_raw_data(data_dir: Path) -> Path:
    """Return a path to ``connect-4.data``, downloading and unpacking if needed.

    The UCI distribution ships a zip containing a ``.Z`` file (Unix ``compress``,
    LZW). Python's stdlib cannot read ``.Z``, but the format is close enough to
    gzip that ``gzip`` handles it on most systems; where it does not, we fall
    back to the ``unlzw3`` package if present and otherwise raise with a clear
    instruction rather than a confusing decode error.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    plain = data_dir / "connect-4.data"
    if plain.exists() and plain.stat().st_size > 0:
        return plain

    archive = data_dir / "connect4.zip"
    if not archive.exists():
        urllib.request.urlretrieve(UCI_URL, archive)

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_dir)

    compressed = data_dir / "connect-4.data.Z"
    if not compressed.exists():
        raise FileNotFoundError(f"expected {compressed} inside {archive}")

    try:
        with gzip.open(compressed, "rb") as src, plain.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    except OSError as exc:  # pragma: no cover - platform dependent
        try:
            import unlzw3

            plain.write_bytes(unlzw3.unlzw(compressed.read_bytes()))
        except ImportError:
            raise RuntimeError(
                f"Could not decompress {compressed} ({exc}). "
                "Install `unlzw3` (pip install unlzw3) or decompress it manually "
                "with `gzip -d connect-4.data.Z`."
            ) from exc

    return plain


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Sample:
    position: Position
    label: int  # index into LABELS


def parse_line(line: str) -> Sample:
    """Turn one CSV row into a :class:`Position` plus its label.

    The dataset's attribute order is column-major and bottom-up --
    ``a1, a2, ..., a6, b1, ...`` -- where ``a`` is the leftmost column and row
    1 is the bottom. Our bit index is ``col * 7 + row`` with row 0 at the
    bottom, so field ``i`` maps to column ``i // 6`` and row ``i % 6``. The two
    layouts agree exactly; no transposition or flipping is required.
    """
    fields = line.strip().split(",")
    if len(fields) != WIDTH * HEIGHT + 1:
        raise ValueError(f"expected {WIDTH * HEIGHT + 1} fields, got {len(fields)}")

    x_bits = 0
    o_bits = 0
    for index, value in enumerate(fields[:-1]):
        if value == "b":
            continue
        col, row = divmod(index, HEIGHT)
        bit = 1 << (col * _COL_STRIDE + row)
        if value == "x":
            x_bits |= bit
        elif value == "o":
            o_bits |= bit
        else:
            raise ValueError(f"unexpected cell value {value!r}")

    label = fields[-1].strip()
    if label not in LABEL_TO_INDEX:
        raise ValueError(f"unexpected label {label!r}")

    # Eight plies means four stones each, and `x` is to move. `position` holds
    # the mover's stones by definition, so it holds x's.
    position = Position(position=x_bits, mask=x_bits | o_bits, moves=8)
    return Sample(position=position, label=LABEL_TO_INDEX[label])


def validate_sample(sample: Sample) -> None:
    """Assert the invariants the dataset documentation promises.

    Cheap, and it catches an off-by-one in the index mapping instantly: a
    wrong stride produces floating stones and unbalanced counts long before it
    produces a plausible-looking board.
    """
    pos = sample.position
    x_count = popcount(pos.position)
    o_count = popcount(pos.position ^ pos.mask)

    if x_count != 4 or o_count != 4:
        raise ValueError(f"expected 4 stones each, got x={x_count} o={o_count}")
    # The documentation promises "neither player has won yet". It also promises
    # the next move is not forced, but "forced" there is a game-theoretic
    # property, not a board property -- it cannot be checked without a solver,
    # so we assert only the half that is decidable from the position itself.
    if pos.has_won():
        raise ValueError("position already contains a four-in-a-row")

    # No stone may float: every occupied cell must have support beneath it.
    for col in range(WIDTH):
        seen_empty = False
        for row in range(HEIGHT):
            occupied = bool(pos.mask & (1 << (col * _COL_STRIDE + row)))
            if not occupied:
                seen_empty = True
            elif seen_empty:
                raise ValueError(f"floating stone in column {col} row {row}")


def load_samples(path: Path, validate: bool = True) -> list[Sample]:
    samples: list[Sample] = []
    with path.open("r", encoding="ascii") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                sample = parse_line(line)
                if validate:
                    validate_sample(sample)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            samples.append(sample)
    return samples


# --------------------------------------------------------------------------
# Feature encoding
# --------------------------------------------------------------------------

# Precomputed cell -> feature-slot table, so encoding is a tight loop over set
# bits rather than a nested loop over all 42 cells.
_CELL_BITS: list[tuple[int, int]] = [
    (1 << (col * _COL_STRIDE + row), col * HEIGHT + row)
    for col in range(WIDTH)
    for row in range(HEIGHT)
]

# Row parity masks, for the odd/even threat features. Row indices are 0-based
# here, so "odd row" in the strategy literature (rows 1, 3, 5 counting from 1)
# is rows 0, 2, 4 in our indexing.
_ODD_ROW_MASK = sum(
    1 << (col * _COL_STRIDE + row) for col in range(WIDTH) for row in range(0, HEIGHT, 2)
)
_EVEN_ROW_MASK = sum(
    1 << (col * _COL_STRIDE + row) for col in range(WIDTH) for row in range(1, HEIGHT, 2)
)


def encode(pos: Position) -> np.ndarray:
    """Feature vector for ``pos``, always from the mover's point of view."""
    features = np.zeros(N_FEATURES, dtype=np.float32)

    mover = pos.position
    opponent = pos.position ^ pos.mask

    for bit, slot in _CELL_BITS:
        if mover & bit:
            features[slot] = 1.0
        elif opponent & bit:
            features[RAW_FEATURES // 2 + slot] = 1.0

    mover_threats = pos.winning_spots()
    opponent_threats = pos.opponent_winning_spots()

    offset = RAW_FEATURES
    # Threat counts, scaled into roughly [0, 1] so no single feature dominates
    # the first layer's gradients.
    features[offset + 0] = popcount(mover_threats) / 4.0
    features[offset + 1] = popcount(opponent_threats) / 4.0
    # Odd/even threat parity -- the strategically decisive signal.
    features[offset + 2] = popcount(mover_threats & _ODD_ROW_MASK) / 4.0
    features[offset + 3] = popcount(mover_threats & _EVEN_ROW_MASK) / 4.0
    features[offset + 4] = popcount(opponent_threats & _ODD_ROW_MASK) / 4.0
    features[offset + 5] = popcount(opponent_threats & _EVEN_ROW_MASK) / 4.0
    # Column heights, which encode the shape of the board compactly.
    for col in range(WIDTH):
        height = popcount(pos.mask & (((1 << HEIGHT) - 1) << (col * _COL_STRIDE)))
        features[offset + 6 + col] = height / HEIGHT
    # Centre control: the single strongest positional feature in Connect 4.
    centre_bits = ((1 << HEIGHT) - 1) << ((WIDTH // 2) * _COL_STRIDE)
    features[offset + 13] = (popcount(mover & centre_bits) - popcount(opponent & centre_bits)) / 6.0

    return features


def encode_batch(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(samples), N_FEATURES), dtype=np.float32)
    y = np.zeros(len(samples), dtype=np.int64)
    for i, sample in enumerate(samples):
        x[i] = encode(sample.position)
        y[i] = sample.label
    return x, y


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split into train/val/test, preserving the class balance in each.

    Stratification matters here because the classes are badly skewed (66/25/10).
    A plain random split can hand the test set a noticeably different draw rate
    than the training set purely by chance, which then shows up as a mysterious
    train/test gap that has nothing to do with the model.
    """
    rng = np.random.default_rng(seed)
    splits: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}

    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)

        n = len(indices)
        n_val = int(round(n * val_fraction))
        n_test = int(round(n * test_fraction))

        splits["val"].append(indices[:n_val])
        splits["test"].append(indices[n_val : n_val + n_test])
        splits["train"].append(indices[n_val + n_test :])

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, parts in splits.items():
        idx = np.concatenate(parts)
        rng.shuffle(idx)  # interleave classes so mini-batches are mixed
        result[name] = (x[idx], y[idx])
    return result


def build_dataset(
    data_dir: Path,
    cache: bool = True,
    seed: int = 0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """End-to-end: download if needed, parse, encode, split. Cached as ``.npz``."""
    cache_path = data_dir / f"encoded_seed{seed}.npz"
    if cache and cache_path.exists():
        blob = np.load(cache_path)
        return {
            name: (blob[f"{name}_x"], blob[f"{name}_y"])
            for name in ("train", "val", "test")
        }

    raw_path = ensure_raw_data(data_dir)
    samples = load_samples(raw_path)
    x, y = encode_batch(samples)
    splits = stratified_split(x, y, seed=seed)

    if cache:
        np.savez_compressed(
            cache_path,
            **{f"{name}_{part}": arr for name, pair in splits.items() for part, arr in zip("xy", pair)},
        )
    return splits
