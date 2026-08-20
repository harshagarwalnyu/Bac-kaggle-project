"""Tests for dataset parsing, feature encoding and splitting.

The risky part of this module is the index mapping. The dataset numbers its
cells column-major and bottom-up; the bitboard numbers them ``col * 7 + row``,
also bottom-up. Those agree, but "they agree" is exactly the kind of claim that
is comfortable to believe and expensive to be wrong about -- a transposed board
still parses, still validates its stone counts, and still trains. It just
learns nonsense.

So the mapping is pinned down here from two directions: a hand-written row
whose board is written out explicitly, and a round-trip through the encoder.
"""

from __future__ import annotations

import gzip
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pytest

from connect4.bitboard import HEIGHT, WIDTH, Position, popcount
from connect4.dataset import (
    LABEL_TO_INDEX,
    LABELS,
    N_FEATURES,
    RAW_FEATURES,
    UCI_URL,
    build_dataset,
    encode,
    encode_batch,
    ensure_raw_data,
    load_samples,
    parse_line,
    stratified_split,
    validate_sample,
)

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "raw" / "connect-4.data"
requires_data = pytest.mark.skipif(
    not DATA_FILE.exists(), reason="raw dataset not downloaded; run scripts/train.py"
)


# Four stones each, nobody has four in a row, nothing floating -- i.e. a board
# the real dataset could plausibly contain. x is split by an o at (3,0), so the
# bottom row is x x x o x, which is three then one.
LEGAL_CELLS: dict[tuple[int, int], str] = {
    (0, 0): "x", (1, 0): "x", (2, 0): "x", (4, 0): "x",
    (3, 0): "o", (5, 0): "o", (6, 0): "o", (0, 1): "o",
}

# The same shape but with o's bottom row unbroken -- illegal for this dataset,
# used to prove the validator actually looks.
#
# It has to be *o* that has the four, not x. ``has_won`` asks whether the side
# that just moved has won, and at 8 plies that side is o. An x four-in-a-row
# would be a different kind of impossible: x would have had to complete it on a
# move that never happened.
WON_CELLS: dict[tuple[int, int], str] = {
    (0, 0): "o", (1, 0): "o", (2, 0): "o", (3, 0): "o",
    (0, 1): "x", (1, 1): "x", (2, 1): "x", (4, 0): "x",
}


def _row(cells: dict[tuple[int, int], str], label: str = "win") -> str:
    """Build a CSV row from an explicit ``{(col, row): 'x'|'o'}`` map."""
    fields = []
    for col in range(WIDTH):
        for row in range(HEIGHT):
            fields.append(cells.get((col, row), "b"))
    fields.append(label)
    return ",".join(fields)


# --------------------------------------------------------------------------
# Parsing and the index mapping
# --------------------------------------------------------------------------


def test_parse_places_stones_in_the_right_cells():
    """All 42 cells of a hand-built row, checked against the rendered grid.

    A wrong stride, or rows counted from the top instead of the bottom, still
    yields eight stones that still validate -- and a completely different
    position. Checking every cell is the only way to rule that out.
    """
    cells = {
        (0, 0): "x", (0, 1): "x", (1, 0): "x", (2, 0): "x",
        (3, 0): "o", (3, 1): "o", (3, 2): "o", (4, 0): "o",
    }
    grid = parse_line(_row(cells)).position.to_grid()

    # to_grid puts row 0 of the *display* at the top, so board row r -- counted
    # from the bottom, as both the dataset and the bitboard count it -- lands at
    # display row HEIGHT - 1 - r.
    codes = {"x": 1, "o": 2, "b": 0}
    for col in range(WIDTH):
        for row in range(HEIGHT):
            want = codes[cells.get((col, row), "b")]
            assert grid[HEIGHT - 1 - row][col] == want, f"cell col={col} row={row}"


def test_parse_reads_the_label():
    for name in LABELS:
        assert parse_line(_row({}, label=name)).label == LABEL_TO_INDEX[name]


def test_parse_rejects_a_bad_field_count():
    with pytest.raises(ValueError, match="expected 43 fields"):
        parse_line("x,o,b,win")


def test_parse_rejects_an_unknown_cell_value():
    fields = _row({}).split(",")
    fields[0] = "q"
    with pytest.raises(ValueError, match="unexpected cell value"):
        parse_line(",".join(fields))


def test_parse_rejects_an_unknown_label():
    with pytest.raises(ValueError, match="unexpected label"):
        parse_line(_row({}, label="tie"))


def test_x_is_always_the_player_to_move():
    """8 plies = 4 each, x moves 1st, 3rd, 5th, 7th -- so x is on move at ply 8.

    This is what lets the labels be used without a sign flip anywhere.
    """
    sample = parse_line(_row(LEGAL_CELLS))
    assert sample.position.current_player() == 1
    assert popcount(sample.position.position) == 4  # `position` holds the mover's stones


# --------------------------------------------------------------------------
# Validation catches the bugs it is there to catch
# --------------------------------------------------------------------------


def test_validate_accepts_a_legal_position():
    validate_sample(parse_line(_row(LEGAL_CELLS)))


def test_validate_rejects_floating_stones():
    """The single most likely symptom of a wrong stride.

    (0, 5) is the top of column 0 with nothing beneath it -- unreachable in a
    real game, and exactly what a mis-strided decode produces.
    """
    bad = parse_line(_row({(0, 5): "x", (1, 0): "x", (2, 0): "x", (4, 0): "x",
                           (3, 0): "o", (5, 0): "o", (6, 0): "o", (1, 1): "o"}))
    with pytest.raises(ValueError, match="floating stone"):
        validate_sample(bad)


def test_validate_rejects_unbalanced_stone_counts():
    sample = parse_line(_row({(c, 0): "x" for c in range(5)}
                             | {(c, 1): "o" for c in range(3)}))
    with pytest.raises(ValueError, match="expected 4 stones each"):
        validate_sample(sample)


def test_validate_rejects_a_finished_game():
    # x has four along the bottom row -- impossible for a position the dataset
    # claims is still in progress.
    with pytest.raises(ValueError, match="four-in-a-row"):
        validate_sample(parse_line(_row(WON_CELLS)))


# --------------------------------------------------------------------------
# Feature encoding
# --------------------------------------------------------------------------


def test_encode_has_the_declared_shape():
    features = encode(Position.from_moves([3, 3, 4]))
    assert features.shape == (N_FEATURES,)
    assert features.dtype == np.float32


def test_raw_planes_round_trip_back_to_the_board():
    """Decode the occupancy planes and check they reproduce the position."""
    pos = Position.from_moves([3, 3, 4, 2, 4, 5, 0])
    features = encode(pos)
    half = RAW_FEATURES // 2

    mover, opponent = pos.position, pos.position ^ pos.mask
    for col in range(WIDTH):
        for row in range(HEIGHT):
            bit = 1 << (col * (HEIGHT + 1) + row)
            slot = col * HEIGHT + row
            assert features[slot] == (1.0 if mover & bit else 0.0)
            assert features[half + slot] == (1.0 if opponent & bit else 0.0)


def test_encoding_is_from_the_movers_point_of_view():
    """The same board with the turn flipped must swap the two planes.

    This is the property that makes a single network usable for both players.
    """
    pos = Position.from_moves([3, 3, 4, 4])
    flipped = Position(position=pos.position ^ pos.mask, mask=pos.mask, moves=pos.moves + 1)

    a = encode(pos)
    b = encode(flipped)
    half = RAW_FEATURES // 2
    assert np.array_equal(a[:half], b[half:RAW_FEATURES])
    assert np.array_equal(a[half:RAW_FEATURES], b[:half])


def test_all_features_stay_in_a_sane_range():
    """Nothing should dominate the first layer purely by magnitude."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        pos = Position()
        for _ in range(rng.integers(0, 30)):
            legal = pos.legal_moves()
            if not legal:
                break
            pos.play(int(rng.choice(legal)))
            if pos.has_won():
                break
        features = encode(pos)
        assert np.isfinite(features).all()
        assert features.min() >= -1.0 and features.max() <= 2.0


def test_empty_board_encodes_to_almost_all_zeros():
    features = encode(Position())
    assert not features.any()


def test_column_height_features_track_the_board():
    pos = Position.from_moves([0, 0, 0, 1])  # column 0 has 3 stones, column 1 has 1
    features = encode(pos)
    heights = features[RAW_FEATURES + 6 : RAW_FEATURES + 13]
    assert heights[0] == pytest.approx(3 / HEIGHT)
    assert heights[1] == pytest.approx(1 / HEIGHT)
    assert heights[2] == 0.0


def test_encode_batch_matches_encoding_one_at_a_time():
    samples = [parse_line(_row(LEGAL_CELLS, label=name)) for name in LABELS]
    x, y = encode_batch(samples)
    assert x.shape == (3, N_FEATURES)
    assert list(y) == [LABEL_TO_INDEX[n] for n in LABELS]
    for i, sample in enumerate(samples):
        assert np.array_equal(x[i], encode(sample.position))


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def _class_balance(y: np.ndarray) -> np.ndarray:
    return np.bincount(y, minlength=len(LABELS)) / len(y)


def test_split_preserves_class_balance():
    rng = np.random.default_rng(0)
    y = rng.choice(3, size=5000, p=[0.25, 0.10, 0.65])
    x = rng.standard_normal((5000, 4)).astype(np.float32)

    splits = stratified_split(x, y, val_fraction=0.15, test_fraction=0.15)
    overall = _class_balance(y)
    for name, (_, part_y) in splits.items():
        assert np.allclose(_class_balance(part_y), overall, atol=0.01), name


def test_split_partitions_every_row_exactly_once():
    """No row may be dropped, and none may appear in two splits."""
    rng = np.random.default_rng(1)
    y = rng.choice(3, size=1000)
    x = np.arange(1000, dtype=np.float32).reshape(-1, 1)  # each row is its own id

    splits = stratified_split(x, y)
    ids = np.concatenate([part_x.ravel() for part_x, _ in splits.values()])
    assert len(ids) == 1000
    assert len(np.unique(ids)) == 1000


def test_split_keeps_features_aligned_with_labels():
    """Shuffling x and y with different permutations would silently destroy the data."""
    rng = np.random.default_rng(2)
    y = rng.choice(3, size=900)
    x = np.stack([y.astype(np.float32)] * 3, axis=1)  # feature == label, by construction

    for part_x, part_y in stratified_split(x, y).values():
        assert np.array_equal(part_x[:, 0].astype(np.int64), part_y)


def test_split_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(3)
    y = rng.choice(3, size=600)
    x = rng.standard_normal((600, 2)).astype(np.float32)

    a = stratified_split(x, y, seed=7)
    b = stratified_split(x, y, seed=7)
    for name in a:
        assert np.array_equal(a[name][1], b[name][1])


# --------------------------------------------------------------------------
# The real file
# --------------------------------------------------------------------------


@requires_data
def test_the_whole_dataset_parses_and_validates():
    """67,557 rows, every one of them legal. Published figures, matched exactly."""
    samples = load_samples(DATA_FILE, validate=True)
    assert len(samples) == 67_557

    counts = np.bincount([s.label for s in samples], minlength=3)
    assert counts[LABEL_TO_INDEX["loss"]] == 16_635
    assert counts[LABEL_TO_INDEX["draw"]] == 6_449
    assert counts[LABEL_TO_INDEX["win"]] == 44_473


@requires_data
def test_every_real_position_has_exactly_eight_stones():
    samples = load_samples(DATA_FILE, validate=False)
    for sample in samples[:2000]:
        assert popcount(sample.position.mask) == 8


# --------------------------------------------------------------------------
# Reading a file
# --------------------------------------------------------------------------


def _legal_row(base: int, label: str = "win") -> str:
    """A legal eight-stone row: two stacked columns starting at ``base``.

    Columns alternate which colour sits on the bottom, so neither rank forms a
    horizontal four. Four stones each, x to move, nothing decided.
    """
    cells: dict[tuple[int, int], str] = {}
    for offset in range(4):
        col = base + offset
        bottom, top = ("x", "o") if offset % 2 == 0 else ("o", "x")
        cells[(col, 0)] = bottom
        cells[(col, 1)] = top
    return _row(cells, label=label)


def test_load_skips_blank_lines(tmp_path: Path):
    """Trailing newlines are the normal shape of a text file, not an error."""
    path = tmp_path / "d.data"
    path.write_text(f"\n{_legal_row(0)}\n\n{_legal_row(1)}\n\n", encoding="ascii")
    assert len(load_samples(path)) == 2


def test_load_names_the_file_and_line_that_failed(tmp_path: Path):
    """A bare "bad label" in a 67,557-row file is not a usable error message."""
    path = tmp_path / "d.data"
    path.write_text(f"{_legal_row(0)}\n{_row({}, label='landslide')}\n", encoding="ascii")

    with pytest.raises(ValueError) as failure:
        load_samples(path)
    assert "d.data:2" in str(failure.value)


def test_load_reports_a_validation_failure_against_the_line_too(tmp_path: Path):
    """The line number has to survive `validate_sample`, not just `parse_line`."""
    path = tmp_path / "d.data"
    # Four stones each, so the count check passes and the gravity check is the
    # one that has to fire: column 3's upper stone hangs two rows above its pile.
    floating = _row(
        {
            (0, 0): "x", (0, 1): "o",
            (1, 0): "o", (1, 1): "x",
            (2, 0): "x", (2, 1): "o",
            (3, 0): "o", (3, 3): "x",
        }
    )
    path.write_text(f"{floating}\n", encoding="ascii")

    with pytest.raises(ValueError) as failure:
        load_samples(path)
    assert "d.data:1" in str(failure.value)
    assert "floating" in str(failure.value)


def test_load_can_be_told_not_to_validate(tmp_path: Path):
    """The real file is validated once; re-checking 67,557 rows per run is waste."""
    path = tmp_path / "d.data"
    path.write_text(f"{_row({(0, 1): 'x', (1, 0): 'o'})}\n", encoding="ascii")
    assert len(load_samples(path, validate=False)) == 1


# --------------------------------------------------------------------------
# Fetching and unpacking
# --------------------------------------------------------------------------


def _uci_archive(directory: Path, body: str) -> Path:
    """A stand-in for the UCI zip: one ``.Z`` member, LZW-ish, gzip-readable.

    The real file is Unix `compress` output. `gzip.open` reads it on the
    platforms we care about, which is the path under test here; the `unlzw3`
    fallback is platform-dependent and marked no-cover in the source.
    """
    archive = directory / "connect4.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("connect-4.data.Z", gzip.compress(body.encode("ascii")))
    return archive


def test_an_already_unpacked_file_is_left_alone(tmp_path: Path):
    """67,557 rows is not something to re-download because a flag says so."""
    plain = tmp_path / "connect-4.data"
    plain.write_text(f"{_legal_row(0)}\n", encoding="ascii")
    before = plain.read_bytes()

    assert ensure_raw_data(tmp_path) == plain
    assert plain.read_bytes() == before, "the existing file was overwritten"


def test_an_empty_file_does_not_count_as_unpacked(tmp_path: Path):
    """A zero-byte file is what a failed download leaves behind, not a dataset."""
    (tmp_path / "connect-4.data").write_bytes(b"")
    _uci_archive(tmp_path, f"{_legal_row(0)}\n")

    plain = ensure_raw_data(tmp_path)
    assert plain.read_text(encoding="ascii").strip() == _legal_row(0)


def test_a_present_archive_is_unpacked_without_downloading(tmp_path: Path, monkeypatch):
    """No network in the test suite. If this reaches urlretrieve, it is a bug."""
    monkeypatch.setattr(
        urllib.request,
        "urlretrieve",
        lambda *a, **k: pytest.fail("downloaded despite a local archive"),
    )
    _uci_archive(tmp_path, f"{_legal_row(0)}\n{_legal_row(1)}\n")

    samples = load_samples(ensure_raw_data(tmp_path))
    assert len(samples) == 2


def test_an_archive_without_the_expected_member_says_so(tmp_path: Path):
    """"No such file" pointing at a path that plainly exists wastes an afternoon."""
    archive = tmp_path / "connect4.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("README", "wrong member")

    with pytest.raises(FileNotFoundError) as failure:
        ensure_raw_data(tmp_path)
    assert "connect-4.data.Z" in str(failure.value)
    assert "connect4.zip" in str(failure.value)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def _corpus(rows_per_label: int = 12) -> str:
    lines = [
        _legal_row(base % 4, label=label)
        for label in LABELS
        for base in range(rows_per_label)
    ]
    return "\n".join(lines) + "\n"


def test_build_dataset_goes_from_archive_to_splits(tmp_path: Path):
    _uci_archive(tmp_path, _corpus())

    splits = build_dataset(tmp_path, cache=False)

    assert set(splits) == {"train", "val", "test"}
    total = sum(len(y) for _, y in splits.values())
    assert total == len(LABELS) * 12
    for x, y in splits.values():
        assert x.shape == (len(y), N_FEATURES)


def test_build_dataset_reuses_its_cache_instead_of_reparsing(tmp_path: Path):
    """The cache is the whole point: parsing 67,557 rows per run is a minute lost."""
    _uci_archive(tmp_path, _corpus())

    first = build_dataset(tmp_path, cache=True)
    assert (tmp_path / "encoded_seed0.npz").exists()

    # Delete the raw inputs. A second call that still succeeds can only have
    # come from the cache, which is a stronger claim than "the arrays match".
    (tmp_path / "connect-4.data").unlink()
    (tmp_path / "connect4.zip").unlink()

    second = build_dataset(tmp_path, cache=True)
    for name in first:
        assert np.array_equal(first[name][0], second[name][0])
        assert np.array_equal(first[name][1], second[name][1])


def test_the_cache_is_keyed_by_seed(tmp_path: Path):
    """One cache file for every seed would silently serve the wrong split."""
    _uci_archive(tmp_path, _corpus())

    build_dataset(tmp_path, cache=True, seed=0)
    build_dataset(tmp_path, cache=True, seed=1)

    assert (tmp_path / "encoded_seed0.npz").exists()
    assert (tmp_path / "encoded_seed1.npz").exists()


def test_a_missing_archive_is_downloaded_once(tmp_path: Path, monkeypatch):
    """The download is stubbed, not skipped: the call has to happen, with the
    published UCI url, and land at the path the unpacking step then reads."""
    calls: list[tuple[str, Path]] = []

    def fake_retrieve(url, filename):
        calls.append((url, Path(filename)))
        with zipfile.ZipFile(filename, "w") as zf:
            zf.writestr("connect-4.data.Z", gzip.compress(f"{_legal_row(0)}\n".encode("ascii")))

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)

    plain = ensure_raw_data(tmp_path)
    assert len(calls) == 1
    assert calls[0][0] == UCI_URL
    assert calls[0][1] == tmp_path / "connect4.zip"
    assert len(load_samples(plain)) == 1
