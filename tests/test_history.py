"""Tests for the persistent game log.

The interesting properties of an append-only file are the ones that only show
up when something goes wrong: a truncated final line from a crash, a file
written by a newer version, a disk that refuses the write. Those are what most
of this module is about -- the happy path is three assertions.
"""

from __future__ import annotations

import json

import pytest

from connect4.history import GameHistory, GameRecord, describe_outcome


def add(history: GameHistory, game_id: str, winner: int = 1, bot_player: int = 2,
        moves: list[int] | None = None) -> GameRecord:
    return history.record(
        game_id=game_id,
        moves=moves if moves is not None else [0, 1, 0, 1, 0, 1, 0],
        winner=winner,
        bot_player=bot_player,
        skill=5,
        started=1000.0,
    )


# --------------------------------------------------------------------------
# In memory
# --------------------------------------------------------------------------


def test_a_recorded_game_comes_back():
    history = GameHistory(None)
    add(history, "a", moves=[3, 3, 4])
    (stored,) = history.recent()
    assert stored.id == "a"
    assert stored.moves == [3, 3, 4]
    assert stored.plies == 3


def test_recording_is_newest_first():
    history = GameHistory(None)
    for game_id in ("a", "b", "c"):
        add(history, game_id)
    assert [g.id for g in history.recent()] == ["c", "b", "a"]


def test_recording_the_same_game_twice_is_a_no_op():
    """A finished game is re-described on every subsequent GET, so this is not
    a hypothetical -- without it the log grows one line per page refresh."""
    history = GameHistory(None)
    first = add(history, "a")
    second = add(history, "a")
    assert len(history) == 1
    assert second is first


def test_the_record_does_not_alias_the_caller_s_move_list():
    """The API hands over the live game's move list; undo mutates it."""
    history = GameHistory(None)
    moves = [3, 3, 4]
    record = add(history, "a", moves=moves)
    moves.append(5)
    assert record.moves == [3, 3, 4]


def test_the_limit_drops_the_oldest_game():
    history = GameHistory(None, limit=3)
    for game_id in "abcde":
        add(history, game_id)
    assert [g.id for g in history.recent()] == ["e", "d", "c"]


def test_recent_can_be_truncated():
    history = GameHistory(None)
    for game_id in "abcde":
        add(history, game_id)
    assert [g.id for g in history.recent(2)] == ["e", "d"]


def test_get_finds_by_id_and_misses_cleanly():
    history = GameHistory(None)
    add(history, "a")
    assert history.get("a").id == "a"
    assert history.get("nope") is None


# --------------------------------------------------------------------------
# Outcome wording
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("winner", "bot_player", "expected"),
    [
        (0, 2, "draw"),
        (1, 2, "you won"),
        (2, 2, "bot won"),
        # Bot playing first owns player 1, which flips both decisive cases.
        (1, 1, "bot won"),
        (2, 1, "you won"),
    ],
)
def test_outcome_is_phrased_from_the_human_side(winner, bot_player, expected):
    assert describe_outcome(winner, bot_player) == expected


def test_the_summary_tallies_the_three_results():
    history = GameHistory(None)
    add(history, "a", winner=1, bot_player=2)   # human won
    add(history, "b", winner=2, bot_player=2)   # bot won
    add(history, "c", winner=2, bot_player=2)   # bot won
    add(history, "d", winner=0, bot_player=2)   # draw
    assert history.summary() == {
        "games": 4, "human_wins": 1, "bot_wins": 2, "draws": 1,
    }


def test_an_empty_summary_is_all_zeroes():
    assert GameHistory(None).summary() == {
        "games": 0, "human_wins": 0, "bot_wins": 0, "draws": 0,
    }


# --------------------------------------------------------------------------
# On disk
# --------------------------------------------------------------------------


def test_games_survive_a_restart(tmp_path):
    path = tmp_path / "games.jsonl"
    history = GameHistory(path)
    add(history, "a", moves=[3, 3])
    add(history, "b", moves=[4])

    reloaded = GameHistory(path)
    assert [g.id for g in reloaded.recent()] == ["b", "a"]
    assert reloaded.get("a").moves == [3, 3]


def test_the_file_is_one_json_object_per_line(tmp_path):
    path = tmp_path / "games.jsonl"
    history = GameHistory(path)
    add(history, "a")
    add(history, "b")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    # Oldest first on disk, which is what "append-only" means.
    assert [json.loads(line)["id"] for line in lines] == ["a", "b"]


def test_a_truncated_final_line_is_skipped_not_fatal(tmp_path):
    """The crash case the format was chosen for."""
    path = tmp_path / "games.jsonl"
    history = GameHistory(path)
    add(history, "a")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "b", "moves": [3, 4')  # power cut mid-write

    reloaded = GameHistory(path)
    assert [g.id for g in reloaded.recent()] == ["a"]


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "games.jsonl"
    GameHistory(path)
    add(GameHistory(path), "a")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n")
    assert len(GameHistory(path)) == 1


def test_unknown_fields_from_a_newer_version_do_not_break_loading(tmp_path):
    path = tmp_path / "games.jsonl"
    path.write_text(
        json.dumps({
            "id": "a", "moves": [3], "winner": 1, "bot_player": 2, "skill": 5,
            "started": 1.0, "ended": 2.0, "outcome": "you won",
            "opening_book_hit": True,   # a field this version has never seen
        }) + "\n",
        encoding="utf-8",
    )
    history = GameHistory(path)
    assert history.get("a").moves == [3]


def test_only_the_last_limit_lines_are_loaded(tmp_path):
    path = tmp_path / "games.jsonl"
    writer = GameHistory(path, limit=100)
    for index in range(10):
        add(writer, f"g{index}")

    reloaded = GameHistory(path, limit=3)
    assert [g.id for g in reloaded.recent()] == ["g9", "g8", "g7"]


def test_a_missing_file_is_an_empty_history(tmp_path):
    assert len(GameHistory(tmp_path / "nothing-here.jsonl")) == 0


def test_the_directory_is_created_on_demand(tmp_path):
    path = tmp_path / "nested" / "deeper" / "games.jsonl"
    add(GameHistory(path), "a")
    assert path.exists()


def test_clear_empties_memory_and_disk(tmp_path):
    path = tmp_path / "games.jsonl"
    history = GameHistory(path)
    add(history, "a")
    history.clear()

    assert len(history) == 0
    assert not path.exists()
    assert len(GameHistory(path)) == 0


def test_clear_on_an_empty_history_is_harmless(tmp_path):
    GameHistory(tmp_path / "games.jsonl").clear()


def test_a_write_failure_does_not_lose_the_game(tmp_path):
    """Durability is best-effort; the in-memory record is not.

    Pointing the log at a path whose parent is a *file* makes mkdir fail, which
    is the cheapest portable way to break the write.
    """
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    history = GameHistory(blocker / "games.jsonl")

    add(history, "a")
    assert [g.id for g in history.recent()] == ["a"]
