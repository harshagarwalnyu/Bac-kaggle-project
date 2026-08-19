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
import threading
import time
import uuid
from collections.abc import Iterable
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
TIME_LIMIT_S = float(os.environ.get("CONNECT4_TIME_LIMIT", "1.0"))
MAX_DEPTH = int(os.environ.get("CONNECT4_MAX_DEPTH", str(MAX_PLIES)))

# Whether the server thinks ahead while it is waiting for the human. Off is
# a supported way to run -- it costs a second a turn and buys back a core.
WARM_AHEAD = os.environ.get("CONNECT4_WARM", "1") != "0"

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
    app.state.analysis_cache = AnalysisCache()

    # A third engine, configured exactly like the one the request path uses,
    # so that an answer it computes in the background is the same answer --
    # not a cheaper one. It needs its own instance because a search mutates
    # the engine it runs on.
    app.state.warmer = Warmer(
        engine=Engine(
            evaluator=heuristic_evaluator,
            max_depth=MAX_DEPTH,
            time_limit_s=TIME_LIMIT_S,
        ),
        cache=app.state.analysis_cache,
        enabled=WARM_AHEAD,
    )
    try:
        yield
    finally:
        # A daemon thread would not hold the process open, but it would keep
        # burning a core through the shutdown of a test suite that starts the
        # app a few hundred times.
        app.state.warmer.close()


app = FastAPI(title="Connect 4 glass-box bot", lifespan=lifespan)


class AnalysisCache:
    """Remembers the search's answer for a position, so it is never asked twice.

    One turn used to cost *three* searches of which two were identical. Playing
    a stone returns an analysis of the position that stone creates; the client
    then immediately asks for a bot move, and the bot analyses -- the very same
    position -- to choose one. Same board, same engine, same answer, a second
    apart.

    Caching is sound rather than merely convenient because the normal engine
    clears its transposition table at the start of every ``analyse``, so a
    position's analysis is a pure function of the position. (The solver engine
    keeps its table, so a later search could in principle prove more; returning
    the earlier answer is a deliberate trade of the last drop of depth for not
    making a person wait twelve seconds twice for one move.)

    Bounded and FIFO because a long session would otherwise accumulate one entry
    per position ever seen. Insertion order is dict order in CPython, so the
    oldest key is simply the first one.
    """

    def __init__(self, capacity: int = 512) -> None:
        self._entries: dict[tuple[int, bool], Analysis] = {}
        self._capacity = capacity
        # The warmer writes from its own thread while requests read and write
        # from FastAPI's threadpool, so every method here is called
        # concurrently. Storing one value is atomic on its own, but eviction is
        # two steps -- name the oldest key, then drop it -- and
        # ``next(iter(entries))`` raises RuntimeError("dictionary changed size
        # during iteration") if another thread inserts between them. The
        # counters have the same problem more quietly: ``+= 1`` is a read and a
        # write, so a racing pair loses a count and the tests that assert on
        # searches-per-turn start flickering. Holding this is free -- every
        # critical section below is a dict lookup, never a search.
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        """Forget everything. Used by tests that count searches; keeping the
        same object matters, because the warmer holds a reference to it and a
        replacement instance would leave the two writing to different dicts."""
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def peek(self, key: tuple[int, bool]) -> Analysis | None:
        """Look without counting.

        The background thread asks whether a position is already known, and
        its curiosity is nobody's cache hit."""
        with self._lock:
            return self._entries.get(key)

    def get(self, key: tuple[int, bool]) -> Analysis | None:
        with self._lock:
            found = self._entries.get(key)
            if found is None:
                self.misses += 1
            else:
                self.hits += 1
            return found

    def put(self, key: tuple[int, bool], analysis: Analysis) -> None:
        # Two threads racing on the same key can only ever store the same
        # value, so the store itself was never the hazard; the eviction that
        # follows it is, and so are the counters. See ``__init__``.
        with self._lock:
            entries = self._entries
            entries[key] = analysis
            while len(entries) > self._capacity:
                entries.pop(next(iter(entries)), None)


class Warmer:
    """Thinks about the human's likely replies while the human is thinking.

    A turn costs two searches and neither can be dropped: one chooses the bot's
    move, the other describes the position that move leaves you in, and the
    second one *is* the analysis panel. What can change is *when* they happen.
    Between the bot's reply landing and the human's next stone there are several
    seconds in which the server does nothing at all.

    So it fills them. Given the position the human is about to move from, it
    analyses the positions the human could create -- best first, according to
    the analysis the panel is already showing, because people mostly play
    reasonable moves -- and drops each result into the same cache the request
    path reads. When the human finally plays, that search has already happened
    and their move comes back immediately.

    Three rules keep this honest rather than merely fast:

    * It uses its own engine, configured identically to the request path's --
      same budget, same evaluator, same table policy -- so a warmed answer is
      the answer a request would have got, not a cheaper one. (As with any
      timed search the depth reached can land a ply either side; what cannot
      happen is a deliberately shallower search being passed off as a full
      one.) A search mutates the engine it runs on, which is why this cannot
      share an instance.
    * An interrupted search is thrown away. When a request arrives the job is
      aborted -- by moving its deadline into the past, which the search already
      checks every 2048 nodes -- and its result is dropped rather than stored,
      so the cache can never hold an analysis cut shorter than the one a request
      would have produced. The bookkeeping is a generation counter rather than
      the search's own ``aborted`` flag, because that flag is also set by a
      search that simply used up its clock, which is the ordinary outcome and
      exactly what the request path returns too.
    * Solver mode is never warmed. That engine deliberately keeps its
      transposition table between searches, so what it can prove depends on what
      it has already been asked -- exactly the property that would make a
      background answer differ from a foreground one.

    One thread, not a pool. Two would double the background throughput and halve
    the foreground's, because the search is pure Python and holds the GIL; the
    thing being optimised here is the human's wait, not the CPU's utilisation.
    """

    def __init__(self, engine: Engine, cache: AnalysisCache, enabled: bool = True) -> None:
        # Public because the thing worth inspecting from outside is exactly
        # this: which engine, and with what budget, produced a warmed answer.
        self.engine = engine
        self._cache = cache
        self.enabled = enabled
        self.warmed = 0
        self._jobs: list[Position] = []
        self._generation = 0
        self._lock = threading.Lock()
        self._work = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="warmer", daemon=True)
        self._thread.start()

    # -------------------------------------------------------------- control

    def suspend(self) -> None:
        """Stop thinking, now. Called before the request path starts a search.

        Cheap enough to call on every request, including the ones that hit the
        cache and never search at all.
        """
        with self._lock:
            self._jobs.clear()
            self._generation += 1
        self.engine.abort()

    def schedule(self, positions: Iterable[Position]) -> None:
        """Queue what to think about next, replacing whatever was queued."""
        if not self.enabled:
            return
        queued = list(positions)
        with self._lock:
            self._jobs = queued
            self._generation += 1
            # Under the same lock the worker uses to declare itself idle, so a
            # caller that schedules and then waits cannot be told "finished"
            # about the batch before this one.
            if queued:
                self._idle.clear()
        if queued:
            self._work.set()

    def disable(self) -> None:
        """Turn it off and stop anything in flight.

        The test suite runs with the warmer off: a test that counts searches is
        measuring the request path, and a helpful background thread filling the
        cache underneath it would make that count depend on timing.
        """
        self.enabled = False
        self.suspend()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until the queue is empty. For tests; nothing in the app waits."""
        return self._idle.wait(timeout)

    def close(self) -> None:
        self._closed = True
        self.suspend()
        self._work.set()
        self._thread.join(timeout=2.0)

    # --------------------------------------------------------------- worker

    def _run(self) -> None:
        while not self._closed:
            self._work.wait()
            self._work.clear()
            while not self._closed:
                with self._lock:
                    if not self._jobs:
                        self._idle.set()
                        break
                    position = self._jobs.pop(0)
                    generation = self._generation
                self._warm(position, generation)

    def _warm(self, position: Position, generation: int) -> None:
        key = (position.key(), False)
        if self._cache.peek(key) is not None:
            return
        analysis = self.engine.analyse(position)
        with self._lock:
            # Anything that happened while this was running -- an abort, a
            # fresh schedule, a shutdown -- bumped the generation, and means
            # the result is either cut short or about a game nobody is
            # playing any more. A search that merely ran out of its own clock
            # is not cut short: that is what the request path produces too.
            if generation != self._generation:
                return
            self._cache.put(key, analysis)
            self.warmed += 1


def _engine_for(skill: int) -> Engine:
    """Solver mode gets the long-budget engine; every other skill gets the
    normal one. Difficulty below 5 is chosen from the *same* honest analysis,
    so there is no reason to think less about it."""
    return app.state.solver if skill >= SOLVER_SKILL else app.state.engine


def _analyse(pos: Position, skill: int) -> Analysis:
    """The only place a search is started. Memoised -- see :class:`AnalysisCache`.

    Keyed on the position *and* on whether this is solver mode, because the two
    engines have different budgets and would answer differently about the same
    board.
    """
    # Whatever the background thread is chewing on, this request matters
    # more: the search is pure Python, so a second one running alongside it
    # would hold the GIL half the time and double the wait being measured.
    app.state.warmer.suspend()

    key = (pos.key(), skill >= SOLVER_SKILL)
    cached = app.state.analysis_cache.get(key)
    if cached is not None:
        return cached
    analysis = _engine_for(skill).analyse(pos)
    app.state.analysis_cache.put(key, analysis)
    return analysis


def _analysis_payload(pos: Position, skill: int = 5) -> dict | None:
    """Analyse ``pos``, or return ``None`` if the game is already over."""
    if pos.has_won() or pos.is_draw():
        return None
    # With no trained model the engine still has plenty to say; the dataset
    # panel simply reports nothing rather than the server refusing to answer.
    return describe_analysis(
        _analyse(pos, skill), pos, app.state.evaluator or _NULL_EVALUATOR
    )


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
    payload = {
        "game": describe_game(game),
        "analysis": _analysis_payload(game.position, game.skill),
        **extra,
    }
    # Only now, with the answer in hand, is the server free to think ahead.
    _schedule_warmup(game)
    return payload


def _schedule_warmup(game: Game) -> None:
    """Queue the positions the *next* request is going to ask about.

    Only while the human is the one being waited on. When it is the bot's turn
    the client is already asking for the move, so there is no idle time to use
    and a background search would only compete with the real one.

    The ordering is free: the analysis this response carries ranks the moves of
    whoever is to move, and that is the human, so its own best-first list is
    also the order in which their replies are worth precomputing.
    """
    position = game.position
    if (
        game.skill >= SOLVER_SKILL
        or position.current_player() == game.bot_player
        or position.has_won()
        or position.is_draw()
    ):
        return

    known = app.state.analysis_cache.peek((position.key(), False))
    columns = (
        [evaluation.column for evaluation in known.evaluations]
        if known
        else position.legal_moves()
    )
    app.state.warmer.schedule(position.played(column) for column in columns)


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
    game = app.state.store.create(
        skill=request.skill,
        bot_player=HUMAN if request.bot_first else BOT,
    )
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

    if request.skill is not None:
        game.skill = request.skill

    # Analysed once and reused: ``choose_move`` runs its own search, so asking
    # it for a move and then separately analysing the same position would pay
    # for the search twice and could -- if anything were nondeterministic --
    # report an opinion that did not match the move played. Going through
    # ``_analyse`` also means the search the *client's* own move already paid
    # for is reused here rather than repeated, which is most of a turn's cost.
    analysis = _analyse(pos, game.skill)
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
