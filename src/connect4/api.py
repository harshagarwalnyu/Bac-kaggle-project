"""HTTP API for the glass-box bot.

Run with::

    uv run python -m connect4.api          # http://127.0.0.1:8000

Two design decisions worth stating up front, because both are deliberate.

**A game is stored as its move list, not as a board.** Replaying seven columns
is free, and it makes undo, sharing and debugging trivial -- a game's entire
state fits in a URL. Storing a bitboard instead would save nothing and cost the
history.

**Analysis is a first-class response, not a debug endpoint.** Every bot move
comes back with both brains' full opinion of every column, because showing the
disagreement *is* the product. The UI does not have to ask a second time, and
cannot accidentally render an opinion about a different position than the one
on screen -- the analysis is attached to the position it describes.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from connect4.bitboard import WIDTH, Position
from connect4.engine import MAX_PLIES, Analysis, Engine, heuristic_evaluator
from connect4.history import GameHistory
from connect4.model import MLP, NeuralEvaluator

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "evaluator.npz"
STATIC_DIR = ROOT / "web"
HISTORY_PATH = Path(os.environ.get("CONNECT4_HISTORY", ROOT / "data" / "games.jsonl"))

# Games are held in memory. That is the right call for a single-player local
# app -- a database would be ceremony around a dict -- but memory is finite, so
# the oldest games are evicted rather than trusted to be cleaned up.
MAX_GAMES = 200

# Seconds the bot may think per move. Tunable because the right value is a
# property of the machine and the audience, not of the code: a demo wants two
# seconds of visible deliberation, a test suite wants none of it.
TIME_LIMIT_S = float(os.environ.get("CONNECT4_TIME_LIMIT", "2.0"))
MAX_DEPTH = int(os.environ.get("CONNECT4_MAX_DEPTH", str(MAX_PLIES)))

# Skill 6 is a separate mode, not another notch on the same dial, and it gets
# its own engine because the difference is the *budget*, not the move choice.
# Connect 4 is a solved game -- the first player wins by move 41 with perfect
# play -- but proving that from an empty board takes billions of nodes, which
# CPython is not going to do inside a web request. What this budget does buy is
# real: from roughly the eighth stone onward the search resolves whole lines
# exactly, and the UI says "proven" only when it genuinely did. Overclaiming a
# solve would be the one dishonest thing this project could ship.
SOLVER_TIME_LIMIT_S = float(os.environ.get("CONNECT4_SOLVER_TIME_LIMIT", "12.0"))
SOLVER_SKILL = 6
MAX_SKILL = SOLVER_SKILL

HUMAN, BOT = 1, 2


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


@dataclass
class Game:
    """A game is its move list. Everything else is derived."""

    id: str
    moves: list[int] = field(default_factory=list)
    skill: int = 5
    bot_player: int = BOT
    created: float = field(default_factory=time.time)

    @property
    def position(self) -> Position:
        return Position.from_moves(self.moves)


class Store:
    """A dict with a bound on it."""

    def __init__(self, capacity: int = MAX_GAMES) -> None:
        self._games: dict[str, Game] = {}
        self._capacity = capacity

    def create(self, skill: int, bot_player: int) -> Game:
        while len(self._games) >= self._capacity:
            oldest = min(self._games.values(), key=lambda g: g.created)
            del self._games[oldest.id]
        game = Game(id=uuid.uuid4().hex[:12], skill=skill, bot_player=bot_player)
        self._games[game.id] = game
        return game

    def get(self, game_id: str) -> Game:
        game = self._games.get(game_id)
        if game is None:
            raise HTTPException(status_code=404, detail="no such game")
        return game


# --------------------------------------------------------------------------
# Wire format
# --------------------------------------------------------------------------


class NewGameRequest(BaseModel):
    skill: int = Field(default=5, ge=0, le=MAX_SKILL)
    bot_first: bool = False
    # An opening to start from, as a list of columns. This is what makes a
    # position shareable and a saved game resumable, and it exists because a
    # game here *is* its move list -- replaying one is the only way to reach a
    # position without inventing a second, board-shaped way to describe one.
    # Illegal sequences are rejected outright rather than clamped, so a typo
    # cannot quietly produce a different position than the one asked for.
    moves: list[int] | None = None


class MoveRequest(BaseModel):
    # Bounded here rather than in the handler, so an out-of-range column is a
    # 422 with a clear message instead of an IndexError somewhere downstream.
    column: int = Field(ge=0, lt=WIDTH)


class BotMoveRequest(BaseModel):
    # Per-request rather than per-game, so the difficulty dial takes effect on
    # the very next move instead of the next game.
    skill: int | None = Field(default=None, ge=0, le=MAX_SKILL)


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def describe_game(game: Game) -> dict:
    """The whole client-visible truth about a game."""
    pos = game.position
    # ``has_won`` speaks about the side that just moved.
    finished_by_win = pos.has_won()
    # Ply 0 is player 1's, so the mover at ply i is 1 when i is even.
    last_player = 1 + (pos.moves - 1) % 2 if pos.moves else 0

    if finished_by_win:
        status, winner = "won", last_player
    elif pos.is_draw():
        status, winner = "draw", 0
    else:
        status, winner = "playing", 0

    return {
        "id": game.id,
        "moves": list(game.moves),
        "grid": pos.to_grid(),
        "ply": pos.moves,
        "turn": 0 if status != "playing" else pos.current_player(),
        # Sorted, because ``legal_moves`` comes back centre-first -- that
        # ordering is a search optimisation (it makes alpha-beta cut early) and
        # has no business leaking into the wire format, where the only sensible
        # order is the one the client draws.
        "legal_moves": [] if status != "playing" else sorted(pos.legal_moves()),
        "status": status,
        "winner": winner,
        "bot_player": game.bot_player,
        "skill": game.skill,
        "solver": game.skill >= SOLVER_SKILL,
        "last_move": game.moves[-1] if game.moves else None,
    }


def describe_analysis(analysis: Analysis, pos: Position, evaluator: NeuralEvaluator) -> dict:
    """Both brains' opinion of every column, from the mover's point of view.

    The two are computed from the same position but are genuinely independent:
    the engine's number is the result of a search that may end in a proof, and
    the network's is a single forward pass with no lookahead at all. Where they
    disagree, the disagreement is the interesting part, so nothing here tries
    to reconcile them.
    """
    columns = []
    for evaluation in analysis.evaluations:
        child = pos.played(evaluation.column)

        # ``probabilities`` describes whoever is to move in ``child`` -- the
        # opponent. Swapping win and loss re-states it for the player who is
        # actually choosing this move.
        if child.has_won():
            # No forward pass needed, and none would be meaningful: the game
            # is over down this line and the network was never shown finished
            # boards.
            network = {"win": 1.0, "draw": 0.0, "loss": 0.0}
            preference = 1.0
        else:
            opponent_view = evaluator.probabilities(child)
            network = {
                "win": opponent_view["loss"],
                "draw": opponent_view["draw"],
                "loss": opponent_view["win"],
            }
            preference = -evaluator(child)

        columns.append(
            {
                "column": evaluation.column,
                "engine": {
                    "score": round(evaluation.score, 4),
                    "label": evaluation.label(),
                    "exact": evaluation.exact,
                    "mate_in": evaluation.mate_in,
                },
                "network": {
                    "preference": round(float(preference), 4),
                    "probabilities": {k: round(float(v), 4) for k, v in network.items()},
                },
            }
        )

    stats = analysis.stats
    return {
        "best_move": analysis.best_move,
        "columns": columns,
        "principal_variation": analysis.principal_variation,
        "stats": {
            "nodes": stats.nodes,
            "depth": stats.depth_reached,
            "table_hits": stats.table_hits,
            "elapsed_ms": round(stats.elapsed_ms, 1),
            "exact": stats.exact,
            "aborted": stats.aborted,
        },
    }


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the network once, at startup.

    If the model file is missing the app still starts and still plays -- it
    falls back to the hand-written heuristic and says so. A missing artifact
    should degrade the analysis panel, not take the server down.
    """
    if MODEL_PATH.exists():
        app.state.evaluator = NeuralEvaluator(MLP.load(MODEL_PATH))
        app.state.model_loaded = True
    else:
        app.state.evaluator = None
        app.state.model_loaded = False

    # The engine plays with the *hand-written* evaluator, not the network, and
    # that is a measured decision rather than a default. At equal search depth
    # the network engine loses 4-8 head to head (scripts/validate.py), because
    # it was trained exclusively on 8-stone positions and a depth-7 search asks
    # it about 15-stone ones. It is excellent on the distribution it was shown
    # -- 0.923 decisive accuracy at ply 8, against 0.734 for the heuristic --
    # and unreliable off it.
    #
    # So the network keeps the job it is actually good at: it publishes an
    # opinion on every column, next to the search's, and the UI shows both.
    app.state.engine = Engine(
        evaluator=heuristic_evaluator,
        max_depth=MAX_DEPTH,
        time_limit_s=TIME_LIMIT_S,
    )

    # Solver mode. Same code, two changes that matter: six times the clock, and
    # a transposition table that survives between moves. The second is the
    # bigger of the two -- consecutive searches in one game overlap enormously,
    # so keeping the table turns each move into a continuation of the last
    # rather than a fresh start.
    app.state.solver = Engine(
        evaluator=heuristic_evaluator,
        max_depth=MAX_PLIES,
        time_limit_s=SOLVER_TIME_LIMIT_S,
        persist_table=True,
    )

    app.state.store = Store()
    app.state.history = GameHistory(HISTORY_PATH)
    yield


app = FastAPI(title="Connect 4 glass-box bot", lifespan=lifespan)


def _engine_for(skill: int) -> Engine:
    """Solver mode gets the long-budget engine; every other skill gets the
    normal one. Difficulty below 5 is chosen from the *same* honest analysis,
    so there is no reason to think less about it."""
    return app.state.solver if skill >= SOLVER_SKILL else app.state.engine


def _analysis_payload(pos: Position, skill: int = 5) -> dict | None:
    """Analyse ``pos``, or return ``None`` if the game is already over.

    This is the *advisory* search -- what the panel shows and what the assist
    toggle recommends -- not the search the bot moved on. It is deliberately
    capped at the normal budget even in solver mode: otherwise a single
    bot-move request pays the solver's long clock twice, once to choose the
    move and once to describe the position it created, and the user waits
    twice as long for no extra strength in the move actually played.

    The solver engine is still the one asked, so the advisory search inherits
    its warm transposition table and is far cheaper than a cold one.
    """
    if pos.has_won() or pos.is_draw():
        return None
    analysis = _engine_for(skill).analyse(pos, time_limit_s=TIME_LIMIT_S)
    # With no trained model the engine still has plenty to say; the dataset
    # panel simply reports nothing rather than the server refusing to answer.
    return describe_analysis(analysis, pos, app.state.evaluator or _NULL_EVALUATOR)


def _archive(game: Game) -> None:
    """Log a game the moment it ends. No-op while it is still being played, and
    idempotent afterwards, so it is safe to call on every response."""
    described = describe_game(game)
    if described["status"] == "playing":
        return
    app.state.history.record(
        game_id=game.id,
        moves=game.moves,
        winner=described["winner"],
        bot_player=game.bot_player,
        skill=game.skill,
        started=game.created,
    )


def _respond(game: Game, **extra) -> dict:
    """The one shape every game endpoint returns.

    Centralised so that "archive finished games" is a property of the API
    rather than a line four handlers have to remember to copy.
    """
    _archive(game)
    return {
        "game": describe_game(game),
        "analysis": _analysis_payload(game.position, game.skill),
        **extra,
    }


class _NullEvaluator:
    """Stands in when no trained model is present. Says nothing, confidently."""

    def __call__(self, pos: Position) -> float:
        return 0.0

    def probabilities(self, pos: Position) -> dict[str, float]:
        return {"win": 0.0, "draw": 0.0, "loss": 0.0}


_NULL_EVALUATOR = _NullEvaluator()


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "model_loaded": app.state.model_loaded,
        "max_skill": MAX_SKILL,
        "solver_skill": SOLVER_SKILL,
        "time_limit_s": TIME_LIMIT_S,
        "solver_time_limit_s": SOLVER_TIME_LIMIT_S,
    }


@app.post("/api/games")
def new_game(request: NewGameRequest) -> dict:
    opening = list(request.moves or ())
    if opening:
        # Validated before any state is created, so a rejected opening leaves
        # no half-built game behind. ``from_moves`` is the single authority on
        # what a legal sequence is; re-implementing that check here would be a
        # second definition of legality waiting to disagree with the first.
        try:
            Position.from_moves(opening)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    game = app.state.store.create(
        skill=request.skill,
        bot_player=HUMAN if request.bot_first else BOT,
    )
    game.moves.extend(opening)
    return _respond(game)


@app.get("/api/games/{game_id}")
def get_game(game_id: str) -> dict:
    return _respond(app.state.store.get(game_id))


@app.post("/api/games/{game_id}/moves")
def play_move(game_id: str, request: MoveRequest) -> dict:
    game = app.state.store.get(game_id)
    pos = game.position

    if pos.has_won() or pos.is_draw():
        raise HTTPException(status_code=409, detail="the game is already over")
    if pos.current_player() == game.bot_player:
        # Without this, a client can post twice in a row and play both sides:
        # the handler alternates players implicitly from the move count, so the
        # second move is silently accepted as the bot's. That is not a
        # hypothetical -- a double-click on a column does it.
        raise HTTPException(status_code=409, detail="it is not your turn")
    if not pos.can_play(request.column):
        raise HTTPException(status_code=409, detail=f"column {request.column} is full")

    game.moves.append(request.column)
    return _respond(game)


@app.post("/api/games/{game_id}/bot-move")
def bot_move(game_id: str, request: BotMoveRequest) -> dict:
    game = app.state.store.get(game_id)
    pos = game.position

    if pos.has_won() or pos.is_draw():
        raise HTTPException(status_code=409, detail="the game is already over")
    if pos.current_player() != game.bot_player:
        # The mirror of the check in ``play_move``. A bot-move request that
        # arrives on the human's turn used to be honoured, which let a retry or
        # a stray click hand the bot two plies in a row.
        raise HTTPException(status_code=409, detail="it is not the bot's turn")

    if request.skill is not None:
        game.skill = request.skill

    # Analysed once and reused: ``choose_move`` runs its own search, so asking
    # it for a move and then separately analysing the same position would pay
    # for the search twice and could -- if anything were nondeterministic --
    # report an opinion that did not match the move played.
    analysis = _engine_for(game.skill).analyse(pos)
    column = _pick(analysis, game.skill)
    if column < 0:
        raise HTTPException(status_code=409, detail="no legal move")

    played_analysis = describe_analysis(
        analysis, pos, app.state.evaluator or _NULL_EVALUATOR
    )
    game.moves.append(column)

    return _respond(game, played=column, played_analysis=played_analysis)


def _pick(analysis: Analysis, skill: int) -> int:
    """Difficulty as *n-th best true move*, mirroring ``Engine.choose_move``.

    Duplicated here rather than calling ``choose_move`` because that method
    searches again from scratch; this one reuses the analysis already paid for.
    The two hard floors are kept identical, since they are what stop easy mode
    from looking broken.
    """
    ranked = analysis.evaluations
    if not ranked:
        return -1

    top = ranked[0]
    # Floor 1: an immediate win on the board is always taken.
    if skill >= 5 or (top.exact and top.score > 0 and (top.mate_in or 99) <= 1):
        return top.column

    index = min(5 - skill, len(ranked) - 1)
    candidate = ranked[index]

    def losing(move) -> bool:
        return bool(move.exact and move.score < 0 and (move.mate_in or 99) <= 2)

    # Floor 2: never walk into a loss the bot can see when something safe
    # exists -- weaker play, not suicidal play.
    #
    # Searching *outward* from the candidate, not from the top of the list.
    # Scanning from index 0 finds the strongest safe move, which means the
    # easiest setting starts playing perfectly at exactly the moment the
    # position gets sharp -- the opposite of what the dial promises. Walking
    # out from where difficulty put us keeps the replacement as close to that
    # strength as safety allows.
    if losing(candidate):
        for offset in range(1, len(ranked)):
            for probe in (index + offset, index - offset):
                if 0 <= probe < len(ranked) and not losing(ranked[probe]):
                    return ranked[probe].column
    return candidate.column


@app.post("/api/games/{game_id}/undo")
def undo(game_id: str) -> dict:
    """Take back a full turn, so that it is the human's move again.

    Popping a fixed two plies is wrong whenever the bot moved first: that game
    goes bot, human, bot, human, so two pops from an even-length list land on
    the bot's turn and quietly delete its opening move as well. Popping until
    the turn comes back round states the actual intent, and it is correct for
    both openings without needing to know which one this is.

    At least one ply always goes, otherwise "undo" on the human's own turn --
    which is when the button is reachable -- would do nothing at all.
    """
    game = app.state.store.get(game_id)
    human = 3 - game.bot_player

    if game.moves:
        game.moves.pop()
    while game.moves and game.position.current_player() != human:
        game.moves.pop()

    return _respond(game)


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------


def describe_record(record) -> dict:
    """A past game, in list form.

    The move list travels with every entry because it is what makes the record
    useful rather than decorative: the client can replay a game, and
    ``POST /api/history/{id}/rematch`` can resume from it.
    """
    return {
        "id": record.id,
        "moves": record.moves,
        "plies": record.plies,
        "winner": record.winner,
        "outcome": record.outcome,
        "bot_player": record.bot_player,
        "skill": record.skill,
        "solver": record.skill >= SOLVER_SKILL,
        "started": record.started,
        "ended": record.ended,
        "duration_s": round(max(0.0, record.ended - record.started), 1),
    }


@app.get("/api/history")
def list_history(limit: int = 25) -> dict:
    limit = max(1, min(limit, 200))
    history: GameHistory = app.state.history
    return {
        "summary": history.summary(),
        "games": [describe_record(r) for r in history.recent(limit)],
    }


@app.get("/api/history/{record_id}")
def get_history_entry(record_id: str) -> dict:
    record = app.state.history.get(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="no such archived game")
    return describe_record(record)


class RematchRequest(BaseModel):
    """How much of the archived game to take back into the new one.

    ``ply`` counts from the start, so 0 replays nothing (same opening, same
    colours, empty board) and omitting it replays everything up to the move
    before the game ended -- the position you would want to think about again.
    """

    ply: int | None = Field(default=None, ge=0)


@app.post("/api/history/{record_id}/rematch")
def rematch(record_id: str, request: RematchRequest) -> dict:
    """Start a fresh game from an archived one.

    The archive stores move lists rather than boards precisely so that this is
    possible, and the endpoint was named in ``describe_record`` before it
    existed. Replaying the moves through the normal ``Game`` object -- rather
    than restoring a board -- means the new game is an ordinary game in every
    respect: it can be undone, analysed and archived like any other.

    Colours and difficulty are inherited, because a rematch you win by quietly
    switching sides is not a rematch.
    """
    record = app.state.history.get(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="no such archived game")

    # One ply short of the end by default: replaying the whole thing would hand
    # back a game that is already over, which is the one position from which a
    # rematch cannot be played.
    default_ply = max(0, len(record.moves) - 1)
    ply = default_ply if request.ply is None else min(request.ply, len(record.moves))

    game = app.state.store.create(skill=record.skill, bot_player=record.bot_player)
    game.moves.extend(record.moves[:ply])

    if game.position.has_won():
        # Only reachable when the caller asked for the full move list of a game
        # that ended in a win. Refuse rather than hand back a dead game.
        raise HTTPException(status_code=409, detail="that ply is already a finished game")

    return _respond(game, replayed_from=record_id, replayed_plies=ply)


@app.delete("/api/history")
def clear_history() -> dict:
    app.state.history.clear()
    return {"cleared": True, "summary": app.state.history.summary()}


if STATIC_DIR.exists():
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("CONNECT4_HOST", "127.0.0.1"),
        port=int(os.environ.get("CONNECT4_PORT", "8000")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
