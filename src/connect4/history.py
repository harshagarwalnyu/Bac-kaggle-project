"""Persistent record of finished games.

Why a file and not a database
-----------------------------
A game is a list of at most 42 small integers. The entire history of a very
heavy demo session is a few kilobytes. SQLite would mean a schema, a migration
story and a connection lifecycle in exchange for queries nobody runs -- the only
question ever asked here is "the last N games, newest first".

So: append-only JSON Lines. One game per line, written the moment the game
ends. That format has three properties that matter more than query power:

* **Append-only writes cannot corrupt earlier games.** A crash mid-write costs
  the last line and nothing else, and a truncated last line is skipped on load
  rather than taking the app down.
* **It is readable without this program.** `tail -1 data/games.jsonl` is a
  legitimate debugging step.
* **Replay is exact.** Because a record stores the move list rather than a
  rendered board, any past game can be re-analysed by the current engine --
  including with a stronger search than the one that played it.

The in-memory list is the read path; the file is the durability path. They are
kept in sync by only ever appending through :meth:`GameHistory.record`.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Newest-first, bounded. A demo does not need the 5000th game back, and an
# unbounded list in a long-running process is just a slow leak.
DEFAULT_LIMIT = 200


@dataclass(frozen=True)
class GameRecord:
    """One finished game, in the form that survives a restart."""

    id: str
    moves: list[int]
    # 0 draw, otherwise the player number (1 or 2) that made four in a row.
    winner: int
    bot_player: int
    skill: int
    started: float
    ended: float
    # Denormalised on purpose: the client renders this string directly, and
    # deriving it in three places (API, history list, tests) invites the three
    # of them to disagree.
    outcome: str = field(default="")

    @property
    def plies(self) -> int:
        return len(self.moves)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_dict(cls, raw: dict) -> GameRecord:
        """Build a record from a decoded line, tolerating unknown extra keys.

        Forward compatibility is cheap here and the alternative is nasty: a
        history file written by a newer version would otherwise crash startup
        on a field this version has never heard of.
        """
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


def describe_outcome(winner: int, bot_player: int) -> str:
    """Phrase the result from the human's point of view, which is the only one
    the player cares about."""
    if winner == 0:
        return "draw"
    return "bot won" if winner == bot_player else "you won"


class GameHistory:
    """Append-only game log, backed by a JSONL file when given a path."""

    def __init__(self, path: Path | None = None, limit: int = DEFAULT_LIMIT) -> None:
        self.path = path
        self.limit = limit
        # `deque(maxlen=)` does the bounding for us; appendleft keeps the list
        # newest-first so the common read needs no sorting.
        self._games: deque[GameRecord] = deque(maxlen=limit)
        if path is not None:
            self._load()

    # ---------------------------------------------------------------- reading

    def __len__(self) -> int:
        return len(self._games)

    def recent(self, limit: int | None = None) -> list[GameRecord]:
        """Newest first."""
        games = list(self._games)
        return games if limit is None else games[:limit]

    def get(self, game_id: str) -> GameRecord | None:
        return next((g for g in self._games if g.id == game_id), None)

    def summary(self) -> dict:
        """Aggregate the score line, because "am I winning overall" is the
        question a player actually asks of a history list."""
        wins = sum(1 for g in self._games if g.winner and g.winner != g.bot_player)
        losses = sum(1 for g in self._games if g.winner and g.winner == g.bot_player)
        draws = sum(1 for g in self._games if not g.winner)
        return {
            "games": len(self._games),
            "human_wins": wins,
            "bot_wins": losses,
            "draws": draws,
        }

    # ---------------------------------------------------------------- writing

    def record(
        self,
        game_id: str,
        moves: list[int],
        winner: int,
        bot_player: int,
        skill: int,
        started: float,
    ) -> GameRecord:
        """Store a finished game. Idempotent by id.

        Idempotence matters because a finished game can be re-described more
        than once -- the client is free to GET a game after it ended, and that
        must not append a duplicate line every time.
        """
        existing = self.get(game_id)
        if existing is not None:
            return existing

        record = GameRecord(
            id=game_id,
            moves=list(moves),
            winner=winner,
            bot_player=bot_player,
            skill=skill,
            started=started,
            ended=time.time(),
            outcome=describe_outcome(winner, bot_player),
        )
        self._games.appendleft(record)
        self._append_to_disk(record)
        return record

    def clear(self) -> None:
        """Forget everything, on disk too. The user asked; do it completely
        rather than leaving a file that reappears on the next restart."""
        self._games.clear()
        if self.path is not None and self.path.exists():
            self.path.unlink()

    # ------------------------------------------------------------------ disk

    def _append_to_disk(self, record: GameRecord) -> None:
        if self.path is None:
            return
        # Never let a logging failure lose a game the user just played: the
        # in-memory copy is already stored by the time we get here, so a disk
        # problem degrades durability and nothing else. The mkdir is inside the
        # guard too -- an unwritable parent is exactly as survivable as an
        # unwritable file.
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(record.to_json() + "\n")
        except OSError:
            pass

    def _load(self) -> None:
        assert self.path is not None
        if not self.path.exists():
            return
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return

        # The file is oldest-first; the deque is newest-first. Taking the tail
        # before reversing means a huge file costs one pass, not `limit`
        # pointless appendlefts that immediately fall off the end.
        for line in reversed(lines[-self.limit:]):
            line = line.strip()
            if not line:
                continue
            try:
                self._games.append(GameRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError):
                # A half-written final line from a crash. Skipping it is the
                # whole reason this format was chosen.
                continue
