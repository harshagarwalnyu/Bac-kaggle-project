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

import pytest
from fastapi.testclient import TestClient

from connect4.api import HUMAN, MAX_GAMES, SOLVER_SKILL, Store, app
from connect4.bitboard import HEIGHT, WIDTH, Position
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


def play(client, game_id: str, column: int) -> dict:
    return client.post(f"/api/games/{game_id}/moves", json={"column": column}).json()


def fill_column(client, game_id: str, column: int = 0) -> dict:
    """Fill one column without ending the game.

    Six *consecutive* plays into the same column is the trick. Because the
    endpoint alternates players automatically, the column comes out
    yellow-red-yellow-red-yellow-red -- no vertical four, and a single column
    cannot make a horizontal or diagonal one either. The obvious version of
    this helper (drop in column 0, answer in column 6) hands player 1 four in a
    row and ends the game on the seventh ply, which silently turns every
    assertion about the resulting analysis into an assertion about ``None``.
    """
    body: dict = {}
    for _ in range(HEIGHT):
        body = play(client, game_id, column)
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
    game_id = new_game(client)["game"]["id"]
    columns = [3, 3, 4, 2, 4]
    for column in columns:
        body = play(client, game_id, column)

    assert body["game"]["grid"] == Position.from_moves(columns).to_grid()
    assert body["game"]["moves"] == columns
    assert body["game"]["last_move"] == columns[-1]
    assert body["game"]["ply"] == len(columns)


def test_turn_alternates(client):
    game_id = new_game(client)["game"]["id"]
    assert play(client, game_id, 0)["game"]["turn"] == 2
    assert play(client, game_id, 1)["game"]["turn"] == 1


# --------------------------------------------------------------------------
# Endings
# --------------------------------------------------------------------------


def test_a_win_is_reported_with_the_right_winner(client):
    """Player 1 takes a vertical four in column 0 while player 2 answers in 1.

    ``has_won`` speaks about the side that just *moved*, and the off-by-one
    there is easy to get wrong in the serialiser -- so it is pinned down.
    """
    game_id = new_game(client)["game"]["id"]
    for column in (0, 1, 0, 1, 0, 1):
        body = play(client, game_id, column)
        assert body["game"]["status"] == "playing"

    body = play(client, game_id, 0)
    game = body["game"]
    assert game["status"] == "won"
    assert game["winner"] == 1
    assert game["turn"] == 0, "a finished game has nobody to move"
    assert game["legal_moves"] == []
    assert body["analysis"] is None, "there is nothing to search once it is over"


def test_moves_are_refused_after_the_game_ends(client):
    game_id = new_game(client)["game"]["id"]
    for column in (0, 1, 0, 1, 0, 1, 0):
        play(client, game_id, column)

    response = client.post(f"/api/games/{game_id}/moves", json={"column": 3})
    assert response.status_code == 409
    assert "over" in response.json()["detail"]

    assert client.post(f"/api/games/{game_id}/bot-move", json={}).status_code == 409


def test_a_full_column_is_refused(client):
    game_id = new_game(client)["game"]["id"]
    fill_column(client, game_id, 0)

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
    game_id = new_game(client)["game"]["id"]
    fill_column(client, game_id, 0)

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
    game_id = new_game(client)["game"]["id"]
    for column in (0, 1, 0, 1, 0):
        play(client, game_id, column)

    analysis = client.get(f"/api/games/{game_id}").json()["analysis"]
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
    game_id = new_game(client, bot_first=True)["game"]["id"]
    # Bot is player 1. Give it three in column 0 with player 2 answering in 1.
    for column in (0, 1, 0, 1, 0, 1):
        play(client, game_id, column)

    body = client.post(f"/api/games/{game_id}/bot-move", json={"skill": skill}).json()
    assert body["played"] == 0
    assert body["game"]["status"] == "won"


@pytest.mark.parametrize("skill", [0, 3, 5])
def test_the_bot_always_blocks_an_immediate_threat(client, skill):
    """The second hard floor: weaker play, never suicidal play."""
    game_id = new_game(client)["game"]["id"]
    for column in (0, 1, 0, 1, 0):  # human threatens to complete column 0
        play(client, game_id, column)

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
    game_id = new_game(client)["game"]["id"]
    for column in (0, 1, 0, 1, 0, 1, 0):
        play(client, game_id, column)
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


def win_for_player_one(client, game_id: str) -> dict:
    """Play a vertical four in column 0 while the opponent answers in column 6.

    This is exactly the accident that `fill_column` exists to avoid, used here
    on purpose: it is the shortest deterministic finished game.
    """
    body: dict = {}
    for i in range(4):
        body = play(client, game_id, 0)
        if i < 3:
            body = play(client, game_id, 6)
    assert body["game"]["status"] == "won"
    return body


def test_a_finished_game_is_archived_exactly_once(client):
    before = client.get("/api/history").json()["summary"]["games"]
    game_id = new_game(client)["game"]["id"]
    win_for_player_one(client, game_id)

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
    game_id = new_game(client)["game"]["id"]
    final = win_for_player_one(client, game_id)["game"]

    record = client.get(f"/api/history/{game_id}").json()
    assert record["moves"] == final["moves"]
    assert record["plies"] == len(final["moves"])
    assert record["winner"] == final["winner"]
    # The whole point of storing moves rather than a board.
    replayed = Position.from_moves(record["moves"])
    assert replayed.to_grid() == final["grid"]
    assert replayed.has_won()


def test_the_outcome_is_phrased_from_the_humans_point_of_view(client):
    game_id = new_game(client)["game"]["id"]
    body = win_for_player_one(client, game_id)["game"]
    record = client.get(f"/api/history/{game_id}").json()
    human_won = body["winner"] != body["bot_player"]
    assert record["outcome"] == ("you won" if human_won else "bot won")


def test_history_is_newest_first(client):
    ids = []
    for _ in range(3):
        game_id = new_game(client)["game"]["id"]
        win_for_player_one(client, game_id)
        ids.append(game_id)
    listing = [entry["id"] for entry in client.get("/api/history").json()["games"]]
    assert listing[:3] == list(reversed(ids))


def test_unknown_history_entry_is_a_404(client):
    assert client.get("/api/history/nope").status_code == 404


def test_history_can_be_cleared(client):
    game_id = new_game(client)["game"]["id"]
    win_for_player_one(client, game_id)
    assert client.get("/api/history").json()["summary"]["games"] > 0

    body = client.delete("/api/history").json()
    assert body["cleared"] is True
    assert body["summary"] == {"games": 0, "human_wins": 0, "bot_wins": 0, "draws": 0}
    assert client.get("/api/history").json()["games"] == []


def test_the_summary_counts_wins_losses_and_draws(client):
    client.delete("/api/history")
    game_id = new_game(client)["game"]["id"]
    final = win_for_player_one(client, game_id)["game"]

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
