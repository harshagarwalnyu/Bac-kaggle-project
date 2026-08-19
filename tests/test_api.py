"""Tests for the HTTP layer.

The engine and the network are tested elsewhere; nothing here re-checks that a
search is correct. What this module protects is the *seam* -- the places where a
correct engine can still be served wrongly:

* the grid the client draws is the position the server holds,
* whose turn it is, and who won, survive the trip through JSON,
* the analysis attached to a response describes the position in that same
  response and not the one before it,
* difficulty does what it claims, including its two hard floors,
* deliberate refusals (full column, finished game) come back as refusals rather
  than as tracebacks or, worse, as silently accepted moves.

Every test drives the app through ``TestClient``, so the lifespan runs and the
real engine plays. That is slower than mocking it, and it is the only way these
tests would have caught a wrong evaluator being wired in.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from connect4.api import (
    HUMAN,
    MAX_GAMES,
    ROOT,
    SOLVER_SKILL,
    Store,
    _history_path,
    _pick,
    app,
)
from connect4.bitboard import HEIGHT, WIDTH, Position
from connect4.engine import Analysis, MoveEvaluation, SearchStats
from connect4.history import GameHistory


@pytest.fixture(scope="module")
def client():
    # Module-scoped: the lifespan loads the network from disk, and paying for
    # that once instead of per-test takes the suite from minutes to seconds.
    with TestClient(app) as test_client:
        # The suite must not append to the developer's real games.jsonl, and a
        # test that asserts on history must not inherit yesterday's games.
        # Swapping in a pathless history gives both: same code, no disk.
        app.state.history = GameHistory(None)
        yield test_client


def new_game(client, **body) -> dict:
    response = client.post("/api/games", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def game_from(client, moves, **body) -> dict:
    """A game that has already been played up to ``moves``.

    Set-up positions go in through the opening parameter rather than through a
    run of ``POST /moves``, because that endpoint now only accepts moves from
    the side whose turn it actually is -- correctly, since letting a client play
    both sides is the bug that made a double-click move twice. Replaying an
    opening is the supported way to reach a position, and it is one request
    instead of a dozen.
    """
    return new_game(client, moves=list(moves), **body)


def play(client, game_id: str, column: int) -> dict:
    return client.post(f"/api/games/{game_id}/moves", json={"column": column}).json()


FULL_COLUMN = [0] * HEIGHT
"""Six stones stacked in column 0, alternating owner, with nobody winning.

Six plays into the *same* column is the trick: the sides alternate, so the
column comes out yellow-red-yellow-red-yellow-red -- no vertical four, and a
single column cannot make a horizontal or diagonal one either. The obvious
version (drop in column 0, answer in column 6) hands player 1 four in a row and
ends the game on the seventh ply, which silently turns every assertion about
the resulting analysis into an assertion about ``None``.
"""

VERTICAL_FOUR = [0, 6, 0, 6, 0, 6, 0]
"""The shortest deterministic finished game: player 1 wins in column 0.

Exactly the accident ``FULL_COLUMN`` exists to avoid, used here on purpose.
"""


def fill_column(client, column: int = 0) -> dict:
    """A game whose column ``column`` is full and which is still playable."""
    body = game_from(client, [column] * HEIGHT)
    assert body["game"]["status"] == "playing", "the filler ended the game"
    return body


# --------------------------------------------------------------------------
# Shape of the world
# --------------------------------------------------------------------------


def test_health_reports_whether_the_network_loaded():
    with TestClient(app) as client:
        body = client.get("/api/health").json()
    assert body["ok"] is True
    assert isinstance(body["model_loaded"], bool)


def test_a_new_game_is_empty_and_the_human_starts(client):
    body = new_game(client)
    game = body["game"]

    assert game["moves"] == []
    assert game["ply"] == 0
    assert game["status"] == "playing"
    assert game["turn"] == HUMAN
    assert game["bot_player"] != HUMAN
    assert game["legal_moves"] == list(range(WIDTH))
    assert game["grid"] == [[0] * WIDTH for _ in range(HEIGHT)]
    assert game["last_move"] is None


def test_bot_first_gives_the_bot_the_first_colour(client):
    game = new_game(client, bot_first=True)["game"]
    assert game["bot_player"] == HUMAN, "playing first means owning player 1"


def test_grid_matches_the_engines_own_rendering(client):
    """The client draws whatever is in ``grid``, so it had better be the board.

    Compared against ``Position.to_grid`` rather than against a hand-written
    expectation, because the question is whether the *serialisation* is faithful,
    not whether the bitboard is right -- that is settled in test_bitboard.py.
    """
    columns = [3, 3, 4, 2, 4]
    body = game_from(client, columns)

    assert body["game"]["grid"] == Position.from_moves(columns).to_grid()
    assert body["game"]["moves"] == columns
    assert body["game"]["last_move"] == columns[-1]
    assert body["game"]["ply"] == len(columns)


def test_turn_alternates(client):
    assert game_from(client, [0])["game"]["turn"] == 2
    assert game_from(client, [0, 1])["game"]["turn"] == 1


# --------------------------------------------------------------------------
# Endings
# --------------------------------------------------------------------------


def test_a_win_is_reported_with_the_right_winner(client):
    """Player 1 takes a vertical four in column 0 while player 2 answers in 1.

    ``has_won`` speaks about the side that just *moved*, and the off-by-one
    there is easy to get wrong in the serialiser -- so it is pinned down.
    """
    body = game_from(client, [0, 1, 0, 1, 0, 1])
    assert body["game"]["status"] == "playing"

    # Played rather than replayed, so the *transition* into a won state goes
    # through the same handler a real client would use.
    body = play(client, body["game"]["id"], 0)
    game = body["game"]
    assert game["status"] == "won"
    assert game["winner"] == 1
    assert game["turn"] == 0, "a finished game has nobody to move"
    assert game["legal_moves"] == []
    assert body["analysis"] is None, "there is nothing to search once it is over"


def test_moves_are_refused_after_the_game_ends(client):
    game_id = game_from(client, VERTICAL_FOUR)["game"]["id"]

    response = client.post(f"/api/games/{game_id}/moves", json={"column": 3})
    assert response.status_code == 409
    assert "over" in response.json()["detail"]

    assert client.post(f"/api/games/{game_id}/bot-move", json={}).status_code == 409


def test_a_full_column_is_refused(client):
    game_id = fill_column(client, 0)["game"]["id"]

    body = client.get(f"/api/games/{game_id}").json()
    assert 0 not in body["game"]["legal_moves"]

    response = client.post(f"/api/games/{game_id}/moves", json={"column": 0})
    assert response.status_code == 409
    assert "full" in response.json()["detail"]


@pytest.mark.parametrize("column", [-1, WIDTH, 99])
def test_out_of_range_columns_are_rejected_by_validation(client, column):
    """422 from the schema, not an IndexError from the board."""
    game_id = new_game(client)["game"]["id"]
    assert client.post(f"/api/games/{game_id}/moves",
                       json={"column": column}).status_code == 422


def test_unknown_game_is_a_404(client):
    assert client.get("/api/games/nope").status_code == 404
    assert client.post("/api/games/nope/moves", json={"column": 0}).status_code == 404


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def test_analysis_covers_exactly_the_legal_columns(client):
    body = new_game(client)
    analysis = body["analysis"]

    columns = sorted(entry["column"] for entry in analysis["columns"])
    assert columns == body["game"]["legal_moves"]
    assert analysis["best_move"] in columns


def test_analysis_describes_the_position_in_the_same_response(client):
    """The single most valuable invariant here.

    An analysis of the *previous* position renders perfectly and is completely
    wrong, so the check is not "is there an analysis" but "is it about this
    board". A column that is full cannot be analysed, which makes it a usable
    fingerprint of which position was searched.
    """
    game_id = fill_column(client, 0)["game"]["id"]

    body = client.get(f"/api/games/{game_id}").json()
    analysed = {entry["column"] for entry in body["analysis"]["columns"]}
    assert analysed == set(body["game"]["legal_moves"])
    assert 0 not in analysed


def test_every_column_carries_both_brains(client):
    analysis = new_game(client)["analysis"]

    for entry in analysis["columns"]:
        engine, network = entry["engine"], entry["network"]
        assert isinstance(engine["exact"], bool)
        assert isinstance(engine["label"], str) and engine["label"]

        # The evaluator is bounded in [-1, 1] by construction; that bound is
        # what stops a guess from ever outranking a proof, so it is asserted
        # here as well as in test_model.py.
        assert -1.0 <= network["preference"] <= 1.0
        probabilities = network["probabilities"]
        assert set(probabilities) == {"win", "draw", "loss"}
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-3)


def test_stats_are_real_numbers(client):
    stats = new_game(client)["analysis"]["stats"]
    assert stats["nodes"] > 0
    assert stats["depth"] >= 1
    assert stats["elapsed_ms"] >= 0
    assert stats["table_hits"] >= 0


def test_principal_variation_is_playable_from_here(client):
    """A variation the board cannot actually follow would draw ghosts in
    impossible cells, so every move in it must be legal in turn."""
    body = new_game(client)
    pos = Position()
    for column in body["analysis"]["principal_variation"]:
        assert pos.can_play(column), f"unplayable move {column} in the variation"
        pos.play(column)
        if pos.has_won():
            break


def test_a_forced_block_is_reported_as_a_proven_loss(client):
    """With an immediate threat on the board, the engine must say so.

    Player 1 has three in column 0 and it is player 2's move. Every reply other
    than column 0 is a proven loss, and the engine should be returning proofs
    rather than opinions about them.
    """
    analysis = game_from(client, [0, 1, 0, 1, 0])["analysis"]
    assert analysis["best_move"] == 0, "the block is the only move"

    others = [e for e in analysis["columns"] if e["column"] != 0]
    assert all(e["engine"]["exact"] and e["engine"]["score"] < 0 for e in others)
    assert all("loss in" in e["engine"]["label"] for e in others)


# --------------------------------------------------------------------------
# The bot
# --------------------------------------------------------------------------


def test_the_bot_plays_a_legal_move_and_reports_what_it_saw(client):
    game_id = new_game(client)["game"]["id"]
    play(client, game_id, 3)

    body = client.post(f"/api/games/{game_id}/bot-move", json={}).json()

    assert body["played"] in range(WIDTH)
    assert body["game"]["moves"][-1] == body["played"]
    assert body["game"]["turn"] == HUMAN

    # `played_analysis` is the opinion that produced the move, so the move must
    # appear in it. `analysis` is about the position afterwards -- a different
    # board, and it must not be confused with the first.
    played_columns = {e["column"] for e in body["played_analysis"]["columns"]}
    assert body["played"] in played_columns
    assert body["analysis"]["columns"] != body["played_analysis"]["columns"]


@pytest.mark.parametrize("skill", [0, 5])
def test_the_bot_always_takes_a_win_on_the_board(client, skill):
    """The first hard floor. Missing a four that is sitting there reads as a
    bug, not as easy mode -- so it must hold even at skill 0."""
    # Bot is player 1. Give it three in column 0 with player 2 answering in 1.
    game_id = game_from(client, [0, 1, 0, 1, 0, 1], bot_first=True)["game"]["id"]

    body = client.post(f"/api/games/{game_id}/bot-move", json={"skill": skill}).json()
    assert body["played"] == 0
    assert body["game"]["status"] == "won"


@pytest.mark.parametrize("skill", [0, 3, 5])
def test_the_bot_always_blocks_an_immediate_threat(client, skill):
    """The second hard floor: weaker play, never suicidal play."""
    # Human threatens to complete column 0.
    game_id = game_from(client, [0, 1, 0, 1, 0])["game"]["id"]

    body = client.post(f"/api/games/{game_id}/bot-move", json={"skill": skill}).json()
    assert body["played"] == 0, f"skill {skill} walked into an immediate loss"


def test_lower_skill_takes_a_worse_move_not_a_random_one(client):
    """Difficulty is the n-th best move, so it must be *ranked*, not sampled.

    Asking twice at the same skill must give the same answer -- if it did not,
    the difficulty dial would be noise wearing a rank's clothing.
    """
    game_id = new_game(client)["game"]["id"]
    play(client, game_id, 3)

    first = client.post(f"/api/games/{game_id}/bot-move", json={"skill": 2}).json()

    replay_id = new_game(client)["game"]["id"]
    play(client, replay_id, 3)
    second = client.post(f"/api/games/{replay_id}/bot-move", json={"skill": 2}).json()

    assert first["played"] == second["played"]

    ranked = [e["column"] for e in first["played_analysis"]["columns"]]
    ranked.sort(key=lambda c: -next(
        e["engine"]["score"] for e in first["played_analysis"]["columns"] if e["column"] == c
    ))
    assert first["played"] in ranked
    assert first["played"] != ranked[0], "skill 2 should not be playing the best move"


def test_skill_sent_with_a_move_sticks_to_the_game(client):
    game_id = new_game(client, skill=5)["game"]["id"]
    play(client, game_id, 3)
    body = client.post(f"/api/games/{game_id}/bot-move", json={"skill": 1}).json()
    assert body["game"]["skill"] == 1


@pytest.mark.parametrize("skill", [-1, SOLVER_SKILL + 1, 99])
def test_out_of_range_skill_is_rejected(client, skill):
    game_id = new_game(client)["game"]["id"]
    play(client, game_id, 3)
    assert client.post(f"/api/games/{game_id}/bot-move",
                       json={"skill": skill}).status_code == 422


# --------------------------------------------------------------------------
# Undo
# --------------------------------------------------------------------------


def test_undo_takes_back_a_whole_turn(client):
    """Both plies, so the human is on move again -- popping one would hand the
    turn to the wrong player."""
    game_id = new_game(client)["game"]["id"]
    play(client, game_id, 3)
    client.post(f"/api/games/{game_id}/bot-move", json={})

    body = client.post(f"/api/games/{game_id}/undo").json()
    assert body["game"]["moves"] == []
    assert body["game"]["turn"] == HUMAN


def test_undo_on_an_empty_game_is_harmless(client):
    game_id = new_game(client)["game"]["id"]
    body = client.post(f"/api/games/{game_id}/undo").json()
    assert body["game"]["moves"] == []


def test_undo_reopens_a_finished_game(client):
    game_id = game_from(client, VERTICAL_FOUR)["game"]["id"]
    assert client.get(f"/api/games/{game_id}").json()["game"]["status"] == "won"

    body = client.post(f"/api/games/{game_id}/undo").json()
    assert body["game"]["status"] == "playing"
    assert body["analysis"] is not None, "a resumed game gets its analysis back"


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def test_the_store_evicts_the_oldest_game_rather_than_growing_forever():
    store = Store(capacity=3)
    first = store.create(skill=5, bot_player=2)
    for _ in range(3):
        store.create(skill=5, bot_player=2)

    assert len(store._games) == 3
    with pytest.raises(Exception):  # HTTPException(404)
        store.get(first.id)


def test_the_default_capacity_is_a_bound_not_a_suggestion():
    assert MAX_GAMES > 0


def test_games_are_independent(client):
    a = new_game(client)["game"]["id"]
    b = new_game(client)["game"]["id"]
    play(client, a, 0)
    assert client.get(f"/api/games/{b}").json()["game"]["moves"] == []


# --------------------------------------------------------------------------
# Static assets
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/", "/style.css", "/app.js"])
def test_the_front_end_is_served(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.content, f"{path} is empty"


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------


def finished_game(client) -> dict:
    """A game that is already over, archived, and won by player one.

    The opening goes in whole rather than ply by ply, because the archive is
    written by whatever describes a finished position -- it does not care
    whether the moves arrived one request at a time.
    """
    body = game_from(client, VERTICAL_FOUR)
    assert body["game"]["status"] == "won"
    return body


def test_a_finished_game_is_archived_exactly_once(client):
    before = client.get("/api/history").json()["summary"]["games"]
    game_id = finished_game(client)["game"]["id"]

    # Re-reading a finished game must not append it again.
    client.get(f"/api/games/{game_id}")
    client.get(f"/api/games/{game_id}")

    listing = client.get("/api/history").json()
    assert listing["summary"]["games"] == before + 1
    ids = [entry["id"] for entry in listing["games"]]
    assert ids.count(game_id) == 1


def test_an_unfinished_game_is_not_archived(client):
    game_id = new_game(client)["game"]["id"]
    play(client, game_id, 3)
    ids = [entry["id"] for entry in client.get("/api/history").json()["games"]]
    assert game_id not in ids


def test_the_archived_record_replays_to_the_finished_position(client):
    final = finished_game(client)["game"]
    game_id = final["id"]

    record = client.get(f"/api/history/{game_id}").json()
    assert record["moves"] == final["moves"]
    assert record["plies"] == len(final["moves"])
    assert record["winner"] == final["winner"]
    # The whole point of storing moves rather than a board.
    replayed = Position.from_moves(record["moves"])
    assert replayed.to_grid() == final["grid"]
    assert replayed.has_won()


def test_the_outcome_is_phrased_from_the_humans_point_of_view(client):
    body = finished_game(client)["game"]
    record = client.get(f"/api/history/{body['id']}").json()
    human_won = body["winner"] != body["bot_player"]
    assert record["outcome"] == ("you won" if human_won else "bot won")


def test_history_is_newest_first(client):
    ids = []
    for _ in range(3):
        ids.append(finished_game(client)["game"]["id"])
    listing = [entry["id"] for entry in client.get("/api/history").json()["games"]]
    assert listing[:3] == list(reversed(ids))


def test_unknown_history_entry_is_a_404(client):
    assert client.get("/api/history/nope").status_code == 404


def test_history_can_be_cleared(client):
    finished_game(client)
    assert client.get("/api/history").json()["summary"]["games"] > 0

    body = client.delete("/api/history").json()
    assert body["cleared"] is True
    assert body["summary"] == {"games": 0, "human_wins": 0, "bot_wins": 0, "draws": 0}
    assert client.get("/api/history").json()["games"] == []


def test_the_summary_counts_wins_losses_and_draws(client):
    client.delete("/api/history")
    final = finished_game(client)["game"]

    summary = client.get("/api/history").json()["summary"]
    assert summary["games"] == 1
    assert summary["draws"] == 0
    if final["winner"] == final["bot_player"]:
        assert (summary["bot_wins"], summary["human_wins"]) == (1, 0)
    else:
        assert (summary["human_wins"], summary["bot_wins"]) == (1, 0)


# --------------------------------------------------------------------------
# Solver mode
# --------------------------------------------------------------------------


def test_solver_skill_is_accepted_and_reported(client):
    body = new_game(client, skill=SOLVER_SKILL)
    assert body["game"]["skill"] == SOLVER_SKILL
    assert body["game"]["solver"] is True


def test_ordinary_skills_are_not_flagged_as_solver(client):
    assert new_game(client, skill=5)["game"]["solver"] is False


def test_skill_above_the_solver_is_rejected(client):
    response = client.post("/api/games", json={"skill": SOLVER_SKILL + 1})
    assert response.status_code == 422


def test_health_advertises_the_solver(client):
    body = client.get("/api/health").json()
    assert body["solver_skill"] == SOLVER_SKILL
    assert body["max_skill"] == SOLVER_SKILL
    # Solver mode is only meaningful if it actually gets more clock.
    assert body["solver_time_limit_s"] > body["time_limit_s"]


# --------------------------------------------------------------------------
# Openings
# --------------------------------------------------------------------------


def test_an_opening_is_replayed_into_the_new_game(client):
    """A game is its move list, so handing one over must reach that position."""
    body = game_from(client, [3, 3])
    assert body["game"]["moves"] == [3, 3]
    assert body["game"]["grid"] == Position.from_moves([3, 3]).to_grid()
    assert body["game"]["turn"] == HUMAN, "an even opening comes back to the human"


@pytest.mark.parametrize(
    "opening",
    [
        [0] * (HEIGHT + 1),  # one stone more than the column can hold
        [WIDTH],  # off the right-hand edge
        [-1],
    ],
)
def test_an_illegal_opening_is_refused_rather_than_clamped(client, opening):
    """Clamping would quietly produce a *different* position than the one asked
    for, which is worse than an error: the client would never find out."""
    assert client.post("/api/games", json={"moves": opening}).status_code == 422


def test_a_rejected_opening_leaves_no_game_behind(client):
    before = len(app.state.store._games)
    client.post("/api/games", json={"moves": [0] * (HEIGHT + 1)})
    assert len(app.state.store._games) == before


# --------------------------------------------------------------------------
# Whose turn it is
# --------------------------------------------------------------------------


def test_the_human_cannot_play_twice_in_a_row(client):
    """A double-click on a column used to play both sides."""
    game_id = new_game(client)["game"]["id"]
    assert client.post(f"/api/games/{game_id}/moves", json={"column": 3}).status_code == 200

    second = client.post(f"/api/games/{game_id}/moves", json={"column": 3})
    assert second.status_code == 409
    assert client.get(f"/api/games/{game_id}").json()["game"]["moves"] == [3]


def test_the_bot_will_not_move_out_of_turn(client):
    """The mirror: a stray retry used to hand the bot two plies in a row."""
    game_id = new_game(client)["game"]["id"]
    response = client.post(f"/api/games/{game_id}/bot-move", json={})
    assert response.status_code == 409
    assert client.get(f"/api/games/{game_id}").json()["game"]["moves"] == []


def test_simultaneous_moves_on_one_game_produce_exactly_one_ply(client):
    """The turn check and the append have to be one indivisible step.

    FastAPI runs synchronous handlers in a threadpool, so these really do run
    at the same time. Checking whose turn it is and *then* appending lets both
    requests read the same empty board, both conclude it is the human's turn,
    and both append -- one click, two plies, and the human has played the bot's
    move for it.
    """
    import threading

    game_id = new_game(client)["game"]["id"]
    start = threading.Barrier(6)
    codes: list[int] = []
    guard = threading.Lock()

    def play():
        start.wait()
        response = client.post(f"/api/games/{game_id}/moves", json={"column": 3})
        with guard:
            codes.append(response.status_code)

    threads = [threading.Thread(target=play) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert codes.count(200) == 1, f"more than one move was accepted: {codes}"
    assert set(codes) <= {200, 409}, f"an unexpected status came back: {codes}"
    assert client.get(f"/api/games/{game_id}").json()["game"]["moves"] == [3]


# --------------------------------------------------------------------------
# Undo, when the bot opened
# --------------------------------------------------------------------------


def test_undo_gives_the_move_back_to_the_human_whoever_opened(client):
    """Bot-first games go bot, human, bot, human, so "a turn" is a different
    pair of plies than it is in a human-first game. Undo is defined by the turn
    it hands back, not by a fixed count -- popping two from the bot's opening
    ply alone would delete a move nobody asked to take back.
    """
    game_id = game_from(client, [3, 3, 4], bot_first=True)["game"]["id"]

    body = client.post(f"/api/games/{game_id}/undo").json()

    # The human's 3 and the bot's reply 4 come off; the bot's *opening* stays.
    assert body["game"]["moves"] == [3]
    human = 3 - body["game"]["bot_player"]
    assert body["game"]["turn"] == human


def test_undo_never_strands_the_bot_on_move(client):
    """The invariant behind the test above, checked from the other opening."""
    game_id = game_from(client, [3, 3], bot_first=True)["game"]["id"]
    body = client.post(f"/api/games/{game_id}/undo").json()
    assert body["game"]["turn"] == 3 - body["game"]["bot_player"]


def test_undo_keeps_the_bots_opening_when_that_is_all_there_is(client):
    """The gap the test above leaves: one bot ply, and nothing of the human's.

    Popping it empties the board and leaves the bot on move -- and nothing ever
    asks the bot to play from there. The client requests a bot move after a
    human move or a new game, never after an undo; the undo button disables
    itself at zero plies; and a human move into the bot's turn is refused with
    409. The game is unrecoverable except by starting another one, so undo has
    to decline instead.
    """
    game_id = game_from(client, [3], bot_first=True)["game"]["id"]

    body = client.post(f"/api/games/{game_id}/undo").json()

    human = 3 - body["game"]["bot_player"]
    assert body["game"]["moves"] == [3], "the bot's opening is not the human's to undo"
    assert body["game"]["turn"] == human

    # The proof that it matters: the human can still play.
    followed = client.post(f"/api/games/{game_id}/moves", json={"column": 2})
    assert followed.status_code == 200
    assert followed.json()["game"]["moves"] == [3, 2]


def test_repeated_undo_never_walks_a_bot_first_game_off_the_board(client):
    """Undo is idempotent once the human has nothing left to take back."""
    game_id = game_from(client, [3, 3, 4, 4], bot_first=True)["game"]["id"]

    seen = []
    for _ in range(5):
        body = client.post(f"/api/games/{game_id}/undo").json()
        seen.append(tuple(body["game"]["moves"]))
        assert body["game"]["turn"] == 3 - body["game"]["bot_player"]

    # The human's own last ply comes off first (the bot has not replied to it
    # yet), then the pair below it, and then there is nothing left to give.
    assert seen == [(3, 3, 4), (3,), (3,), (3,), (3,)], seen


# --------------------------------------------------------------------------
# Difficulty, when the easy move loses
# --------------------------------------------------------------------------


def _evaluation(column: int, score: float, exact: bool = False, mate_in=None):
    return MoveEvaluation(column=column, score=score, exact=exact, mate_in=mate_in)


def test_an_unsafe_easy_move_is_replaced_by_a_near_one_not_the_best_one(client):
    """Safety must not silently promote easy mode to perfect play.

    Ranked best-first, the skill-0 candidate is index 5. It loses, so a
    replacement is needed -- but scanning from the top would return column 0,
    the strongest move on the board, which is exactly what the dial promised
    not to do. Walking outward finds the neighbour at index 6, the weaker of
    the two adjacent safe moves and so the one that stays closest to the
    strength the dial asked for.
    """
    ranked = [_evaluation(column, 1.0 - column * 0.1) for column in range(WIDTH)]
    ranked[5] = _evaluation(5, -1.0, exact=True, mate_in=2)

    picked = _pick(Analysis(best_move=0, evaluations=ranked), skill=0)
    assert picked == 6
    assert picked != ranked[0].column, "safety promoted easy mode to perfect play"


def test_a_won_position_is_still_won_at_the_easiest_setting():
    """Floor 1 outranks everything above."""
    ranked = [
        _evaluation(2, 1.0, exact=True, mate_in=1),
        _evaluation(3, 0.0),
    ]
    assert _pick(Analysis(best_move=2, evaluations=ranked), skill=0) == 2


# --------------------------------------------------------------------------
# Rematch
# --------------------------------------------------------------------------


def test_a_rematch_replays_the_archived_game_one_ply_short(client):
    """The position worth thinking about again is the one before the mistake --
    replaying the whole thing hands back a game that is already over."""
    archived = finished_game(client)["game"]
    record_id = archived["id"]

    body = client.post(f"/api/history/{record_id}/rematch", json={}).json()

    assert body["game"]["moves"] == archived["moves"][:-1]
    assert body["game"]["status"] == "playing"
    assert body["game"]["id"] != record_id, "a rematch is a new game"
    assert body["replayed_from"] == record_id


def test_a_rematch_inherits_colours_and_difficulty(client):
    """Winning a rematch by quietly switching sides is not winning a rematch."""
    archived = finished_game(client)["game"]
    body = client.post(f"/api/history/{archived['id']}/rematch", json={}).json()

    assert body["game"]["bot_player"] == archived["bot_player"]
    assert body["game"]["skill"] == archived["skill"]


def test_a_rematch_can_start_from_the_very_beginning(client):
    archived = finished_game(client)["game"]
    body = client.post(f"/api/history/{archived['id']}/rematch", json={"ply": 0}).json()
    assert body["game"]["moves"] == []
    assert body["replayed_plies"] == 0


def test_a_rematch_of_the_whole_finished_game_is_refused(client):
    """The one position a rematch cannot be played from."""
    archived = finished_game(client)["game"]
    response = client.post(
        f"/api/history/{archived['id']}/rematch",
        json={"ply": len(archived["moves"])},
    )
    assert response.status_code == 409


def test_a_refused_rematch_leaves_no_game_behind(client):
    """A 409 must not cost a slot in the store.

    The store is capacity-bounded and evicts the oldest game to make room, so a
    rejected rematch that still created a game would eventually throw away a
    game somebody is playing to make room for one nobody can play.
    """
    archived = finished_game(client)["game"]
    live = new_game(client)["game"]["id"]

    for _ in range(MAX_GAMES + 1):
        response = client.post(
            f"/api/history/{archived['id']}/rematch",
            json={"ply": len(archived["moves"])},
        )
        assert response.status_code == 409

    assert client.get(f"/api/games/{live}").status_code == 200, (
        "refused rematches evicted a live game"
    )


def test_a_drawn_game_is_refused_as_a_rematch_position_too(client):
    """A full board is as finished as a won one, and was not being caught."""
    drawn = [
        0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0,
        2, 3, 2, 3, 2, 3,
        3, 2, 3, 2, 3, 2,
        4, 5, 4, 5, 4, 5,
        5, 4, 5, 4, 5, 4,
        6, 6, 6, 6, 6, 6,
    ]
    body = game_from(client, drawn)["game"]
    assert body["status"] == "draw", "the fixture stopped being a drawn game"

    response = client.post(
        f"/api/history/{body['id']}/rematch", json={"ply": len(drawn)}
    )
    assert response.status_code == 409


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_history_defaults_to_a_file_and_can_be_turned_off(monkeypatch):
    """Unset, set, and set-but-empty are three different answers."""
    monkeypatch.delenv("CONNECT4_HISTORY", raising=False)
    assert _history_path() == ROOT / "data" / "games.jsonl"

    monkeypatch.setenv("CONNECT4_HISTORY", "/somewhere/else.jsonl")
    assert _history_path() == Path("/somewhere/else.jsonl")

    # Empty means memory only. Path("") would write to the working directory,
    # which is the least useful reading of "I do not want a file".
    monkeypatch.setenv("CONNECT4_HISTORY", "   ")
    assert _history_path() is None
    assert GameHistory(_history_path()).path is None


def test_a_rematch_of_an_unknown_game_is_a_404(client):
    assert client.post("/api/history/nope/rematch", json={}).status_code == 404


def test_a_negative_rematch_ply_is_rejected(client):
    archived = finished_game(client)["game"]
    response = client.post(f"/api/history/{archived['id']}/rematch", json={"ply": -1})
    assert response.status_code == 422


def test_a_rematch_of_a_corrupt_archive_line_is_refused_not_crashed(client):
    """The archive is a file, and files arrive damaged.

    ``Position.from_moves`` is the single authority on legality and it raises on
    anything it rejects. A prefix of a game played through this API is always
    legal, so the only way to reach that raise is a record this process did not
    write -- the JSONL log can be hand-edited, truncated mid-write, or left
    behind by an older version with a different notion of a legal move. Without
    the refusal, one bad line turns every rematch of that record into a 500.

    Eight stones in one column is the clearest example: six fit, and a rematch
    replays all but the last ply, so seven of them still have to be refused.
    """
    record = client.app.state.history.record(
        game_id="corrupt-line",
        moves=[0, 0, 0, 0, 0, 0, 0, 0],
        winner=0,
        bot_player=2,
        skill=3,
        started=0.0,
    )

    response = client.post(f"/api/history/{record.id}/rematch", json={})

    assert response.status_code == 422, response.text
    assert response.json()["detail"], "a refusal should say what was wrong"


def test_a_pick_with_nothing_scored_still_names_a_column():
    """``_pick`` mirrors ``Engine.choose_move`` and has to mirror its fallback.

    An analysis that scored nothing still carries a legal move, and answering -1
    here is what the caller turns into "no legal move" for a board that has six.
    """
    starved = Analysis(best_move=4, evaluations=[], stats=SearchStats())
    for skill in range(6):
        assert _pick(starved, skill) == 4
