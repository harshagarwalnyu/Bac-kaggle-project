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
from connect4.engine import Analysis, Engine, heuristic_evaluator
from connect4.model import MLP, NeuralEvaluator

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "evaluator.npz"
STATIC_DIR = ROOT / "web"

# Games are held in memory. That is the right call for a single-player local
# app -- a database would be ceremony around a dict -- but memory is finite, so
# the oldest games are evicted rather than trusted to be cleaned up.
MAX_GAMES = 200

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
    skill: int = Field(default=5, ge=0, le=5)
    bot_first: bool = False


class MoveRequest(BaseModel):
    # Bounded here rather than in the handler, so an out-of-range column is a
    # 422 with a clear message instead of an IndexError somewhere downstream.
    column: int = Field(ge=0, lt=WIDTH)


class BotMoveRequest(BaseModel):
    # Per-request rather than per-game, so the difficulty dial takes effect on
    # the very next move instead of the next game.
    skill: int | None = Field(default=None, ge=0, le=5)


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
        "legal_moves": [] if status != "playing" else pos.legal_moves(),
        "status": status,
        "winner": winner,
        "bot_player": game.bot_player,
        "skill": game.skill,
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
        max_depth=12,
        time_limit_s=2.0,
    )
    app.state.store = Store()
    yield


app = FastAPI(title="Connect 4 glass-box bot", lifespan=lifespan)


def _analysis_payload(pos: Position) -> dict | None:
    """Analyse ``pos``, or return ``None`` if the game is already over."""
    if pos.has_won() or pos.is_draw():
        return None
    analysis = app.state.engine.analyse(pos)
    # With no trained model the engine still has plenty to say; the dataset
    # panel simply reports nothing rather than the server refusing to answer.
    return describe_analysis(analysis, pos, app.state.evaluator or _NULL_EVALUATOR)


class _NullEvaluator:
    """Stands in when no trained model is present. Says nothing, confidently."""

    def __call__(self, pos: Position) -> float:
        return 0.0

    def probabilities(self, pos: Position) -> dict[str, float]:
        return {"win": 0.0, "draw": 0.0, "loss": 0.0}


_NULL_EVALUATOR = _NullEvaluator()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "model_loaded": app.state.model_loaded}


@app.post("/api/games")
def new_game(request: NewGameRequest) -> dict:
    game = app.state.store.create(
        skill=request.skill,
        bot_player=HUMAN if request.bot_first else BOT,
    )
    return {"game": describe_game(game), "analysis": _analysis_payload(game.position)}


@app.get("/api/games/{game_id}")
def get_game(game_id: str) -> dict:
    game = app.state.store.get(game_id)
    return {"game": describe_game(game), "analysis": _analysis_payload(game.position)}


@app.post("/api/games/{game_id}/moves")
def play_move(game_id: str, request: MoveRequest) -> dict:
    game = app.state.store.get(game_id)
    pos = game.position

    if pos.has_won() or pos.is_draw():
        raise HTTPException(status_code=409, detail="the game is already over")
    if not pos.can_play(request.column):
        raise HTTPException(status_code=409, detail=f"column {request.column} is full")

    game.moves.append(request.column)
    return {"game": describe_game(game), "analysis": _analysis_payload(game.position)}


@app.post("/api/games/{game_id}/bot-move")
def bot_move(game_id: str, request: BotMoveRequest) -> dict:
    game = app.state.store.get(game_id)
    pos = game.position

    if pos.has_won() or pos.is_draw():
        raise HTTPException(status_code=409, detail="the game is already over")

    if request.skill is not None:
        game.skill = request.skill

    # Analysed once and reused: ``choose_move`` runs its own search, so asking
    # it for a move and then separately analysing the same position would pay
    # for the search twice and could -- if anything were nondeterministic --
    # report an opinion that did not match the move played.
    analysis = app.state.engine.analyse(pos)
    column = _pick(analysis, game.skill)
    if column < 0:
        raise HTTPException(status_code=409, detail="no legal move")

    played_analysis = describe_analysis(
        analysis, pos, app.state.evaluator or _NULL_EVALUATOR
    )
    game.moves.append(column)

    return {
        "game": describe_game(game),
        "played": column,
        "played_analysis": played_analysis,
        "analysis": _analysis_payload(game.position),
    }


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

    # Floor 2: never walk into a loss the bot can see when something safe
    # exists -- weaker play, not suicidal play.
    if candidate.exact and candidate.score < 0 and (candidate.mate_in or 99) <= 2:
        for alternative in ranked:
            if not (alternative.exact and alternative.score < 0
                    and (alternative.mate_in or 99) <= 2):
                return alternative.column
    return candidate.column


@app.post("/api/games/{game_id}/undo")
def undo(game_id: str) -> dict:
    """Take back a full turn: the human's move and the bot's reply.

    Popping a single ply would hand the turn to the wrong player, so both go.
    """
    game = app.state.store.get(game_id)
    for _ in range(2):
        if game.moves:
            game.moves.pop()
    return {"game": describe_game(game), "analysis": _analysis_payload(game.position)}


if STATIC_DIR.exists():
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
