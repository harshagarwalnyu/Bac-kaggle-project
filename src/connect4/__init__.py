"""A glass-box Connect 4 bot.

Two independent brains reason about every position and both publish their
opinion: an exact alpha-beta search over bitboards, and a neural evaluator
trained from scratch on the UCI connect-4 opening database.
"""

from .bitboard import HEIGHT, WIDTH, Position
from .engine import Analysis, Engine, MoveEvaluation, SearchStats, heuristic_evaluator

# Grouped to mirror the two imports above -- board vocabulary first, then the
# search -- which is the order a reader meets them in. Alphabetical would
# interleave the two and say nothing.
__all__ = [  # noqa: RUF022
    "HEIGHT",
    "WIDTH",
    "Position",
    "Engine",
    "Analysis",
    "MoveEvaluation",
    "SearchStats",
    "heuristic_evaluator",
]
