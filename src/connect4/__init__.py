"""A glass-box Connect 4 bot.

Two independent brains reason about every position and both publish their
opinion: an exact alpha-beta search over bitboards, and a neural evaluator
trained from scratch on the Kaggle/UCI connect-4 dataset.
"""

from .bitboard import HEIGHT, WIDTH, Position
from .engine import Analysis, Engine, MoveEvaluation, SearchStats, heuristic_evaluator

__all__ = [
    "HEIGHT",
    "WIDTH",
    "Position",
    "Engine",
    "Analysis",
    "MoveEvaluation",
    "SearchStats",
    "heuristic_evaluator",
]
