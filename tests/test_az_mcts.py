"""Tests for the AlphaZero feature encoding and tree search.

Deliberately torch-free. The search takes its evaluator as a callable, so
everything here runs against hand-written stubs -- which means these tests
catch search bugs rather than network bugs, and keep running whether or not the
optional ``az`` extra is installed.

The two properties worth the most here are the sign of the backup and the
handling of terminal nodes. Both are silent when wrong: a search with an
inverted backup still runs, still returns a distribution, and simply prefers
losing. There is no traceback to follow, so the only defence is a test that
knows the right answer independently.
"""

from __future__ import annotations

import numpy as np
import pytest

from connect4.az.features import PLANES, mirror, mirror_policy, planes, planes_batch
from connect4.az.mcts import MCTSConfig, Node, Search, run, run_batch
from connect4.bitboard import HEIGHT, WIDTH, Position


def uniform(position: Position) -> tuple[np.ndarray, float]:
    """A stub that knows nothing: flat prior, dead-even value."""
    return np.full(WIDTH, 1.0 / WIDTH, dtype=np.float32), 0.0


def uniform_batch(positions: list[Position]) -> tuple[np.ndarray, np.ndarray]:
    n = len(positions)
    return np.full((n, WIDTH), 1.0 / WIDTH, dtype=np.float32), np.zeros(n, dtype=np.float32)


def quiet(simulations: int = 200) -> MCTSConfig:
    """A config with the root noise off, so results are about the search."""
    return MCTSConfig(simulations=simulations, add_noise=False)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def test_planes_have_the_declared_shape_and_dtype():
    board = planes(Position.from_moves([3, 3, 4]))
    assert board.shape == (PLANES, HEIGHT, WIDTH)
    assert board.dtype == np.float32


def test_empty_board_is_all_zeros():
    assert not planes(Position()).any()


def test_planes_agree_with_the_rendered_grid():
    """Every one of the 42 cells, checked against a rendering that predates this module.

    A transposed or upside-down encoding trains perfectly well and learns
    nonsense, exactly as in the dataset tests. ``to_grid`` numbers rows from the
    top for display; these planes number them from the bottom.
    """
    pos = Position.from_moves([3, 3, 4, 2, 4, 5, 0, 0, 1])
    board = planes(pos)
    grid = pos.to_grid()
    mover = pos.current_player()

    for col in range(WIDTH):
        for row in range(HEIGHT):
            cell = grid[HEIGHT - 1 - row][col]
            assert board[0][row][col] == (1.0 if cell == mover else 0.0), (col, row)
            assert board[1][row][col] == (1.0 if cell == 3 - mover else 0.0), (col, row)


def test_planes_are_from_the_movers_point_of_view():
    """Same board, other player to move: the two planes swap. One net, both sides."""
    pos = Position.from_moves([3, 3, 4, 4])
    flipped = Position(position=pos.position ^ pos.mask, mask=pos.mask, moves=pos.moves + 1)

    a, b = planes(pos), planes(flipped)
    assert np.array_equal(a[0], b[1])
    assert np.array_equal(a[1], b[0])


def test_planes_never_show_a_stone_belonging_to_both():
    pos = Position.from_moves([0, 0, 1, 2, 3, 3, 4])
    board = planes(pos)
    assert not (board[0] * board[1]).any()
    assert board.sum() == pos.moves


def test_planes_batch_matches_encoding_one_at_a_time():
    positions = [Position.from_moves(m) for m in ([], [3], [3, 3], [0, 1, 2])]
    batch = planes_batch(positions)
    assert batch.shape == (4, PLANES, HEIGHT, WIDTH)
    for i, pos in enumerate(positions):
        assert np.array_equal(batch[i], planes(pos))


def test_planes_batch_of_nothing_still_has_the_right_shape():
    """An empty batch has to keep its trailing dimensions or torch rejects it."""
    assert planes_batch([]).shape == (0, PLANES, HEIGHT, WIDTH)


def test_mirroring_a_board_is_the_same_as_mirroring_the_moves():
    """The augmentation has to be a real symmetry of the game, not just an axis flip."""
    moves = [3, 2, 4, 1, 0]
    original = planes_batch([Position.from_moves(moves)])
    reflected = planes_batch([Position.from_moves([WIDTH - 1 - m for m in moves])])
    assert np.array_equal(mirror(original), reflected)


def test_mirroring_twice_is_the_identity():
    batch = planes_batch([Position.from_moves([0, 1, 2, 3])])
    assert np.array_equal(mirror(mirror(batch)), batch)
    policy = np.array([[0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1]])
    assert np.allclose(mirror_policy(mirror_policy(policy)), policy)


# --------------------------------------------------------------------------
# Node bookkeeping
# --------------------------------------------------------------------------


def test_a_node_whose_mover_has_already_lost_is_terminal_at_minus_one():
    """Four in a row means the side *to move* has lost -- there is no win-to-move node."""
    pos = Position.from_moves([0, 1, 0, 1, 0, 1, 0])  # x completes a vertical four
    node = Node(pos)
    assert node.is_terminal
    assert node.terminal_value == -1.0


def drawn_board() -> Position:
    """A full 42-stone board with no four in a row anywhere.

    Built rather than played, because filling the columns in any obvious order
    completes a four long before the board is full. The pattern is: even columns
    hold three x then three o, odd columns the reverse. Vertically that is runs
    of three; horizontally the colours alternate every column; and on both
    diagonals the column parity flips at every step while the top/bottom half
    flips at most once along the line, so the longest run is two.
    """
    stride = HEIGHT + 1
    mover = 0
    mask = 0
    for col in range(WIDTH):
        for row in range(HEIGHT):
            bit = 1 << (col * stride + row)
            mask |= bit
            if (col % 2 == 0) == (row < HEIGHT // 2):
                mover |= bit
    return Position(position=mover, mask=mask, moves=WIDTH * HEIGHT)


def test_a_full_board_is_terminal_at_zero():
    pos = drawn_board()
    assert pos.is_draw()
    assert not pos.has_won()
    node = Node(pos)
    assert node.is_terminal
    assert node.terminal_value == 0.0
    assert not node.legal


def test_an_ordinary_node_is_not_terminal():
    node = Node(Position.from_moves([3, 3]))
    assert not node.is_terminal
    assert node.value() == 0.0


def test_backup_alternates_sign():
    """The single most consequential line in the module, pinned from outside.

    A leaf worth +1 to its own mover is worth -1 to the parent and +1 again to
    the grandparent. Invert this and the search prefers to lose, quietly.
    """
    root = Node(Position())
    child = Node(root.position.played(3))
    grandchild = Node(child.position.played(3))
    path = [(root, 3), (child, 3)]

    Search._backup(path, 1.0)

    assert child.value_sum[3] == -1.0  # the leaf's gain is the parent's loss
    assert root.value_sum[3] == 1.0
    assert root.total_visits == 1
    assert grandchild.total_visits == 0


def test_select_never_returns_a_full_column():
    columns = [0] * HEIGHT  # column 0 filled to the brim
    pos = Position.from_moves(columns)
    node = Node(pos)
    node.prior = np.zeros(WIDTH, dtype=np.float32)
    node.prior[0] = 1.0  # a prior that insists on the illegal move
    assert node.select(1.5, 0.0) != 0


def test_fpu_reduction_discourages_unvisited_children():
    """With a reduction set, an unvisited child inherits the parent's value minus it."""
    node = Node(Position())
    node.prior = np.full(WIDTH, 1.0 / WIDTH, dtype=np.float32)
    node.visits[3] = 10
    node.value_sum[3] = 2.0  # column 3 is measured, and mildly good: Q = 0.2
    node.total_visits = 10
    # Without a reduction an unvisited column is assumed to be an even game,
    # and its undiluted U term (nothing has been subtracted from a denominator
    # of 1) beats a measured 0.2. With a reduction the unvisited columns
    # inherit the parent's value minus it, and the measured column wins.
    assert node.select(1.5, 0.0) != 3
    assert node.select(1.5, 1.0) == 3


# --------------------------------------------------------------------------
# The search
# --------------------------------------------------------------------------


def test_every_simulation_is_accounted_for():
    """Root visits total one less than the simulations: the first one expands the root.

    That first simulation walks an empty path, so there is no edge to credit.
    Anything else means simulations are being lost or double-counted.
    """
    config = quiet(50)
    search = run(Position.from_moves([3, 3, 3]), uniform, config)
    assert search.simulations_done == 50
    assert int(search.visit_counts().sum()) == 49


def test_a_full_column_never_gets_a_visit():
    pos = Position.from_moves([0, 0, 0, 0, 0, 0, 1])
    search = run(pos, uniform, quiet(60))
    assert search.visit_counts()[0] == 0


def test_the_search_takes_a_win_that_is_on_the_board():
    """x has three stacked in column 0 and it is x's move. There is one answer."""
    pos = Position.from_moves([0, 1, 0, 1, 0, 1])
    search = run(pos, uniform, quiet(200))
    assert int(np.argmax(search.visit_counts())) == 0


def test_the_search_blocks_a_threat_it_cannot_ignore():
    """o to move, x threatening a vertical four in column 0. Anything else loses.

    This is the property that needs terminal handling *and* the sign of the
    backup to both be right: the loss is two plies away, so the search only sees
    it by propagating a terminal value up through one flip.
    """
    pos = Position.from_moves([0, 1, 0, 1, 0])
    search = run(pos, uniform, quiet(300))
    assert int(np.argmax(search.visit_counts())) == 0


def test_a_won_position_is_scored_as_lost_by_the_side_to_move():
    pos = Position.from_moves([0, 1, 0, 1, 0, 1])
    search = run(pos, uniform, quiet(200))
    assert search.root_value() > 0.5  # the mover is about to win, and knows it


def test_a_search_from_a_finished_game_does_nothing():
    pos = Position.from_moves([0, 1, 0, 1, 0, 1, 0])
    search = Search(pos, quiet(100))
    assert search.done
    assert search.descend() is None
    assert not search.visit_counts().any()


def test_the_prior_actually_steers_the_search():
    """Same position, same value, different priors -- different visit distributions."""

    def biased(position: Position) -> tuple[np.ndarray, float]:
        prior = np.full(WIDTH, 0.01, dtype=np.float32)
        prior[6] = 0.94
        return prior, 0.0

    flat = run(Position(), uniform, quiet(80)).visit_counts()
    steered = run(Position(), biased, quiet(80)).visit_counts()
    assert steered[6] > flat[6]


def test_priors_on_illegal_columns_are_redistributed_not_merely_dropped():
    """A prior that is entirely illegal must still leave a usable distribution."""

    def insists_on_a_full_column(position: Position) -> tuple[np.ndarray, float]:
        prior = np.zeros(WIDTH, dtype=np.float32)
        prior[0] = 1.0
        return prior, 0.0

    pos = Position.from_moves([0, 0, 0, 0, 0, 0])  # column 0 is full
    search = run(pos, insists_on_a_full_column, quiet(40))
    prior = search.root.prior
    assert prior[0] == 0.0
    assert prior.sum() == pytest.approx(1.0)
    # Nothing legal was preferred, so the fallback has to be even-handed.
    assert prior[1:] == pytest.approx(np.full(WIDTH - 1, 1.0 / (WIDTH - 1)))


def test_a_prior_of_all_zeros_falls_back_to_uniform():
    def says_nothing(position: Position) -> tuple[np.ndarray, float]:
        return np.zeros(WIDTH, dtype=np.float32), 0.0

    search = run(Position(), says_nothing, quiet(30))
    assert search.root.prior == pytest.approx(np.full(WIDTH, 1.0 / WIDTH))


def test_root_noise_changes_the_prior_but_not_its_total():
    config = MCTSConfig(simulations=30, add_noise=True, dirichlet_fraction=0.25)
    a = run(Position(), uniform, config, np.random.default_rng(0)).root.prior
    b = run(Position(), uniform, config, np.random.default_rng(1)).root.prior
    assert not np.allclose(a, b)  # different noise, different prior
    assert a.sum() == pytest.approx(1.0)
    assert b.sum() == pytest.approx(1.0)


def test_noise_is_confined_to_the_root():
    """Deeper nodes must see the network's opinion undisturbed.

    Noise exists to vary self-play openings. Applied throughout the tree it
    would stop being exploration and start being a corrupted evaluator.
    """
    search = run(Position(), uniform, MCTSConfig(simulations=200, add_noise=True))
    child = next(c for c in search.root.children if c is not None and c.expanded)
    assert child.prior == pytest.approx(np.full(WIDTH, 1.0 / WIDTH))


def test_a_seeded_search_repeats_itself_exactly():
    a = run(Position(), uniform, MCTSConfig(simulations=60), np.random.default_rng(7))
    b = run(Position(), uniform, MCTSConfig(simulations=60), np.random.default_rng(7))
    assert np.array_equal(a.visit_counts(), b.visit_counts())


# --------------------------------------------------------------------------
# The two-step protocol, and its misuse
# --------------------------------------------------------------------------


def test_descending_twice_without_expanding_is_an_error():
    search = Search(Position(), quiet(10))
    search.descend()
    with pytest.raises(RuntimeError, match="without an intervening expand"):
        search.descend()


def test_expanding_with_nothing_pending_is_an_error():
    search = Search(Position(), quiet(10))
    with pytest.raises(RuntimeError, match="no pending leaf"):
        search.expand(np.full(WIDTH, 1 / WIDTH), 0.0)


def test_a_search_is_not_done_while_a_leaf_is_outstanding():
    """Otherwise the batch driver drops the last evaluation of every search."""
    search = Search(Position(), quiet(1))
    assert search.descend() is not None
    assert not search.done
    search.expand(np.full(WIDTH, 1 / WIDTH), 0.0)
    assert search.done


# --------------------------------------------------------------------------
# Policy extraction
# --------------------------------------------------------------------------


def test_temperature_zero_is_the_most_visited_move():
    search = run(Position.from_moves([0, 1, 0, 1, 0, 1]), uniform, quiet(200))
    policy = search.policy(0.0)
    assert policy.sum() == pytest.approx(1.0)
    assert int(np.argmax(policy)) == int(np.argmax(search.visit_counts()))
    assert policy.max() == 1.0


def test_temperature_one_is_the_visit_counts_normalised():
    search = run(Position(), uniform, quiet(120))
    counts = search.visit_counts()
    assert search.policy(1.0) == pytest.approx(counts / counts.sum())


def test_a_low_temperature_sharpens_without_overflowing():
    """Counts reach the hundreds; a naive ``counts ** (1/0.05)`` overflows to inf."""
    search = run(Position(), uniform, quiet(200))
    policy = search.policy(0.05)
    assert np.isfinite(policy).all()
    assert policy.sum() == pytest.approx(1.0)
    assert policy.max() > search.policy(1.0).max()


def test_policy_of_an_unsearched_position_is_uniform_over_legal_moves():
    search = Search(Position.from_moves([0] * HEIGHT), quiet(10))
    policy = search.policy(1.0)
    assert policy[0] == 0.0
    assert policy.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Batched driving
# --------------------------------------------------------------------------


def test_run_batch_finishes_every_search():
    positions = [Position.from_moves(m) for m in ([], [3], [3, 3], [0, 1, 0, 1, 0])]
    searches = [Search(p, quiet(40)) for p in positions]
    run_batch(searches, uniform_batch)
    for search in searches:
        assert search.done
        assert search.simulations_done == 40


def test_run_batch_tolerates_a_search_that_is_already_over():
    finished = Search(Position.from_moves([0, 1, 0, 1, 0, 1, 0]), quiet(20))
    ordinary = Search(Position(), quiet(20))
    run_batch([finished, ordinary], uniform_batch)
    assert ordinary.simulations_done == 20
    assert finished.simulations_done == 0


def test_batched_and_single_stepping_agree():
    """The batch driver must be a scheduling change and nothing more."""
    config = MCTSConfig(simulations=80, add_noise=False)
    single = run(Position.from_moves([3, 2]), uniform, config)
    batched = Search(Position.from_moves([3, 2]), config)
    run_batch([batched], uniform_batch)
    assert np.array_equal(single.visit_counts(), batched.visit_counts())
