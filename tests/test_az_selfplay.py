"""Tests for self-play, the evaluation cache and the arena.

Still torch-free: everything here runs against stub evaluators and stub
players. What is being checked is the *bookkeeping* -- which is where the bugs
in a self-play pipeline actually live. A sign error in the value labelling or a
scoreboard that credits the wrong side produces a training run that looks
entirely healthy for hours and learns to lose.
"""

from __future__ import annotations

import numpy as np
import pytest

from connect4.az.arena import (
    EnginePlayer,
    RandomPlayer,
    Record,
    match,
    openings,
    play_games,
)
from connect4.az.cache import CachedEvaluator
from connect4.az.mcts import MCTSConfig
from connect4.az.selfplay import SelfPlayConfig
from connect4.az.selfplay import play_games as self_play
from connect4.bitboard import HEIGHT, WIDTH, Position
from connect4.engine import Engine


def uniform_batch(positions) -> tuple[np.ndarray, np.ndarray]:
    n = len(positions)
    return np.full((n, WIDTH), 1.0 / WIDTH, dtype=np.float32), np.zeros(n, dtype=np.float32)


def quick(games_in_parallel: int = 8, simulations: int = 24) -> SelfPlayConfig:
    return SelfPlayConfig(
        games_in_parallel=games_in_parallel,
        mcts=MCTSConfig(simulations=simulations),
    )


# --------------------------------------------------------------------------
# Self-play bookkeeping
# --------------------------------------------------------------------------


def test_self_play_produces_the_requested_number_of_games():
    games = list(self_play(uniform_batch, 6, quick(4), np.random.default_rng(0)))
    assert len(games) == 6
    assert all(game for game in games)


def test_every_game_reaches_a_real_ending():
    """A game must stop because it ended, not because the loop got bored."""
    for game in self_play(uniform_batch, 4, quick(4), np.random.default_rng(1)):
        last = game[-1].position
        final = last.played(int(np.argmax(game[-1].policy)))
        # Every recorded position is mid-game by construction; the ending is one
        # move past the last of them.
        assert not last.has_won()
        assert len(game) == last.moves + 1
        assert final.moves <= WIDTH * 6


def test_values_alternate_and_end_at_plus_one_for_a_decisive_game():
    """The winner's last position is +1, and the sign flips every ply back.

    This is the labelling that makes or breaks the whole method. If it is
    inverted the network learns, very efficiently, to lose.
    """
    for game in self_play(uniform_batch, 8, quick(8), np.random.default_rng(2)):
        values = [e.value for e in game]
        if values[-1] == 0.0:
            assert all(v == 0.0 for v in values)  # a draw is zero throughout
            continue
        assert values[-1] == 1.0
        for i in range(len(values) - 1):
            assert values[i] == -values[i + 1]


def test_a_drawn_game_labels_every_position_zero():
    """Forced by construction rather than hoped for: fill the board without a four.

    Random self-play draws too rarely to rely on, so the property is checked on
    a game built to be a draw.
    """
    from connect4.az.selfplay import _Game

    game = _Game()
    stride = 7
    mover = mask = 0
    for col in range(WIDTH):
        for row in range(6):
            bit = 1 << (col * stride + row)
            mask |= bit
            if (col % 2 == 0) == (row < 3):
                mover |= bit
    game.positions = [Position(), Position.from_moves([3])]
    game.policies = [np.full(WIDTH, 1 / WIDTH), np.full(WIDTH, 1 / WIDTH)]
    game.position = Position(position=mover, mask=mask, moves=42)

    assert [e.value for e in game.examples()] == [0.0, 0.0]


def test_every_recorded_policy_is_a_distribution_over_legal_moves():
    for game in self_play(uniform_batch, 3, quick(3), np.random.default_rng(3)):
        for example in game:
            assert example.policy.sum() == pytest.approx(1.0)
            illegal = set(range(WIDTH)) - set(example.position.legal_moves())
            assert all(example.policy[c] == 0.0 for c in illegal)


def test_recorded_positions_replay_the_game_in_order():
    for game in self_play(uniform_batch, 2, quick(2), np.random.default_rng(4)):
        for ply, example in enumerate(game):
            assert example.position.moves == ply


def test_self_play_is_reproducible_for_a_seed():
    a = list(self_play(uniform_batch, 4, quick(4), np.random.default_rng(11)))
    b = list(self_play(uniform_batch, 4, quick(4), np.random.default_rng(11)))
    assert [len(g) for g in a] == [len(g) for g in b]
    assert [e.value for e in a[0]] == [e.value for e in b[0]]


def test_more_games_than_run_in_parallel_still_all_get_played():
    """The refill path: finished games are replaced until the quota is met."""
    games = list(self_play(uniform_batch, 9, quick(2), np.random.default_rng(5)))
    assert len(games) == 9


# --------------------------------------------------------------------------
# The evaluation cache
# --------------------------------------------------------------------------


class CountingEvaluator:
    """Records exactly which positions it was asked about."""

    def __init__(self) -> None:
        self.calls = 0
        self.evaluated = 0

    def __call__(self, positions) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        self.evaluated += len(positions)
        # A distinct, position-dependent answer, so a mis-scattered cache entry
        # shows up as a wrong number rather than as a coincidence.
        values = np.array([float(p.moves) for p in positions], dtype=np.float32)
        priors = np.stack([np.full(WIDTH, 1.0 + p.moves, dtype=np.float32) for p in positions])
        return priors, values


def test_the_cache_answers_a_repeat_without_asking_again():
    inner = CountingEvaluator()
    cached = CachedEvaluator(inner)
    positions = [Position(), Position.from_moves([3])]

    first_priors, first_values = cached(positions)
    second_priors, second_values = cached(positions)

    assert inner.evaluated == 2  # the second call asked about nothing new
    assert np.array_equal(first_priors, second_priors)
    assert np.array_equal(first_values, second_values)
    assert cached.hit_rate == 0.5


def test_the_cache_deduplicates_within_a_single_batch():
    inner = CountingEvaluator()
    cached = CachedEvaluator(inner)
    pos = Position.from_moves([3, 3])
    priors, values = cached([pos, Position(), pos, pos])

    assert inner.evaluated == 2
    assert values[0] == values[2] == values[3] == 2.0
    assert values[1] == 0.0
    assert np.array_equal(priors[0], priors[3])


def test_cached_answers_match_the_uncached_ones_exactly():
    """A cache that returns different numbers is not a cache, it is a bug."""
    inner = CountingEvaluator()
    cached = CachedEvaluator(CountingEvaluator())
    positions = [Position.from_moves(m) for m in ([], [3], [3, 3], [0, 1], [3])]
    assert np.array_equal(cached(positions)[0], inner(positions)[0])
    assert np.array_equal(cached(positions)[1], inner(positions)[1])


def test_transpositions_share_an_entry():
    """Same board by a different move order is the same board, and must hit."""
    inner = CountingEvaluator()
    cached = CachedEvaluator(inner)
    cached([Position.from_moves([0, 1, 2, 3])])
    cached([Position.from_moves([2, 3, 0, 1])])
    assert inner.evaluated == 1
    assert cached.hits == 1


def test_clearing_the_cache_forces_a_re_evaluation():
    """The weights changing is exactly when a stale answer becomes a wrong one."""
    inner = CountingEvaluator()
    cached = CachedEvaluator(inner)
    cached([Position()])
    cached.clear()
    cached([Position()])
    assert inner.evaluated == 2
    assert len(cached) == 1


def test_the_cache_stays_within_its_capacity():
    inner = CountingEvaluator()
    cached = CachedEvaluator(inner, capacity=4)
    for i in range(12):
        cached([Position.from_moves([3] * (i % 6)), Position.from_moves([0] * (i % 6))])
    assert len(cached) <= 4


def test_reset_stats_leaves_the_entries_alone():
    inner = CountingEvaluator()
    cached = CachedEvaluator(inner)
    cached([Position()])
    cached([Position()])
    cached.reset_stats()
    assert cached.hits == cached.misses == 0
    assert len(cached) == 1


# --------------------------------------------------------------------------
# The arena
# --------------------------------------------------------------------------


class ScriptedPlayer:
    """Plays a fixed column when it can, otherwise the leftmost legal one."""

    def __init__(self, column: int) -> None:
        self.column = column

    def choose_moves(self, positions) -> list[int]:
        return [
            self.column if p.can_play(self.column) else p.legal_moves()[0] for p in positions
        ]


def test_openings_are_distinct_and_start_in_the_centre():
    lines = openings(6, plies=2)
    assert len(lines) == 6
    assert len({tuple(line) for line in lines}) == 6
    assert lines[0] == [WIDTH // 2, WIDTH // 2]


def test_single_ply_openings_are_supported():
    lines = openings(3, plies=1)
    assert lines == [[3], [2], [4]]


def test_the_arena_credits_the_winner():
    """Two stackers on different columns: the one moving first gets there first."""
    record = play_games(ScriptedPlayer(0), ScriptedPlayer(1), [[]])
    assert record.wins == 1
    assert record.losses == 0


def test_the_arena_credits_the_loser():
    """A position where the player moving first is simply lost, and loses.

    Two stackers racing from an empty board is no test at all -- whoever moves
    first gets there first, so the scoreboard would look right even if it only
    ever printed "wins". The opening here hands the *other* side three in a
    column and the move after next, so ``first`` cannot win and does not.
    """
    record = play_games(ScriptedPlayer(6), ScriptedPlayer(0), [[0, 6, 0, 6, 0]])
    assert record.losses == 1
    assert record.wins == 0


def test_an_odd_length_opening_does_not_invert_the_scoreboard():
    """The parity trap. With a one-ply opening, ``first`` moves on odd plies.

    Assume otherwise and every result comes back the wrong way round -- which
    would read as a network that trains beautifully and never gets promoted.
    """
    record = play_games(ScriptedPlayer(0), ScriptedPlayer(1), [[6]])
    assert record.wins == 1
    assert record.losses == 0


def test_a_match_alternates_colours():
    """Identical players must come out even; anything else is a seat bias leaking in."""
    record = match(ScriptedPlayer(3), ScriptedPlayer(3), games=8, opening_plies=2)
    assert record.played == 8
    assert record.wins == record.losses
    assert record.score == pytest.approx(0.5)


def test_a_record_scores_a_draw_as_half():
    assert Record(wins=3, draws=2, losses=5).score == pytest.approx(0.4)
    assert Record().score == 0.0
    assert Record(wins=1, draws=1, losses=1).played == 3


def test_every_arena_game_ends():
    record = play_games(RandomPlayer(np.random.default_rng(0)), ScriptedPlayer(3), openings(8))
    assert record.played == 8


def test_the_engine_beats_random_convincingly():
    """A sanity check on the adapter: if this fails, the plumbing is wrong.

    Skill 5 losing to random play would mean the arena is feeding positions or
    reading results backwards, not that the engine has an off day.
    """
    engine = EnginePlayer(Engine(time_limit_s=0.05), skill=5)
    record = match(engine, RandomPlayer(np.random.default_rng(0)), games=6)
    assert record.score > 0.8


# --------------------------------------------------------------------------
# Regressions from the review of the AlphaZero branch.


def test_openings_generates_the_depth_it_was_asked_for():
    """The old implementation returned two-move lines whatever `plies` said.

    `match` forwards `opening_plies` straight through, so a caller asking for
    deeper openings silently got the shallow ones and never found out.
    """
    for plies in range(1, 5):
        lines = openings(4, plies)
        assert len(lines) == 4
        assert {len(line) for line in lines} == {plies}


def test_openings_are_distinct_at_every_depth():
    for plies in range(1, 4):
        lines = openings(WIDTH**plies, plies)
        assert len({tuple(line) for line in lines}) == len(lines)


def test_openings_still_fans_out_from_the_centre():
    """Column 3 is the only winning first move; the order must still start there."""
    assert openings(3, 1) == [[3], [2], [4]]
    assert openings(2, 2) == [[3, 3], [3, 2]]


def test_openings_refuses_to_return_fewer_lines_than_asked_for():
    """Truncating silently would report a 40-game score computed from 14 games."""
    with pytest.raises(ValueError, match="only 7"):
        openings(8, 1)
    with pytest.raises(ValueError, match="only 49"):
        openings(50, 2)
    # One short of the cap is fine, so the boundary is not off by one.
    assert len(openings(49, 2)) == 49


def test_every_opening_is_a_line_a_game_could_actually_contain():
    """Deeper than a column is tall, which the old product-of-columns could not do.

    `openings(1, 7)` used to be seven moves in column three; `from_moves`
    rejects the seventh, and `play_games` calls `from_moves` on every line.
    """
    for plies in (1, 2, 6, 7, 10):
        for line in openings(3, plies):
            assert len(line) == plies
            Position.from_moves(line)  # raises if the line is not playable


def test_openings_are_positions_with_a_game_still_left_in_them():
    """A line that already won or drew hands the two players nothing to play."""
    for plies in (4, 7, 10):
        for line in openings(5, plies):
            position = Position.from_moves(line)
            assert not position.has_won()
            assert not position.is_draw()


def test_the_supply_of_deep_openings_is_smaller_than_the_naive_count():
    """WIDTH ** plies counts sequences no game can contain; this counts lines."""
    with pytest.raises(ValueError, match=r"only \d+"):
        openings(WIDTH**7, 7)


def test_openings_rejects_a_negative_count():
    with pytest.raises(ValueError, match="must not be negative"):
        openings(-1, 2)


def test_openings_rejects_a_zero_ply_request():
    with pytest.raises(ValueError, match="at least 1"):
        openings(4, 0)


def test_openings_refuses_a_depth_no_game_can_reach():
    """A full board is a finished game, so a 42-ply opening cannot exist.

    Answering this by enumeration means walking the whole Connect 4 tree, so
    the point of the check is that it returns rather than that it is correct.
    """
    with pytest.raises(ValueError, match="only 42 discs"):
        openings(1, WIDTH * HEIGHT)
