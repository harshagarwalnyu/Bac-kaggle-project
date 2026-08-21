"""PUCT tree search -- the "search" half of AlphaZero.

Two things separate this from :mod:`connect4.engine`'s alpha-beta:

1. It is **best-first, not depth-first**. Alpha-beta proves bounds and returns
   exact game-theoretic scores when it can; this returns a *distribution over
   moves* built from how often each was visited. Alpha-beta is right; this is
   merely well-informed. The compensation is that its output doubles as a
   training target, which a proof does not.
2. Its leaf value comes from a network, not from a hand-written heuristic or a
   rollout. AlphaZero's actual contribution was noticing you can drop the
   rollout entirely.

**The evaluator is injected, not imported.** :class:`Search` never mentions
torch, or any network. It asks a callable for ``(priors, value)`` and does not
care where they came from -- which is what lets the whole thing be tested
against a stub with no framework installed.

**Why the odd two-step API.** :meth:`Search.descend` walks to a leaf and hands
it back; :meth:`Search.expand` feeds the answer in. A single ``run`` method
would be nicer to read, but it forces one network call per leaf, and on a CPU a
batch of 64 tiny forward passes costs barely more than one. Splitting the step
lets :func:`run_batch` drive many games at once and evaluate all their leaves
together. That is the difference between self-play being feasible on this
machine and not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from connect4.bitboard import WIDTH

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from connect4.bitboard import Position

#: What an evaluator must return for one position: a prior over all ``WIDTH``
#: columns (illegal ones included -- the search masks them) and a value in
#: ``[-1, 1]`` from the point of view of the side to move.
Evaluation = tuple[np.ndarray, float]


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    """Knobs, with the reasoning for each default.

    ``simulations`` is the strength dial and the cost dial at once.

    ``c_puct`` trades exploiting the current best child against trusting the
    prior. 1.5 is the usual starting point for a game this small; too low and an
    early lucky value pins the search on one column, too high and the visits
    spread so thin that none of them mean anything.

    ``dirichlet_alpha`` is noise added at the root so self-play does not play
    the same game every time -- with a deterministic network and no noise, a
    self-play run of 10,000 games is one game recorded 10,000 times. The usual
    rule of thumb is about ``10 / legal_moves``; Connect 4 has seven columns, so
    a little over 1.

    ``fpu_reduction`` is what an unvisited child is assumed to be worth,
    subtracted from the parent's own value. Zero means "assume it is a draw",
    which is optimistic for a losing position and makes the search re-check
    moves it has already found to be bad. A small positive value tells the
    search to trust what it has seen. Kept at 0 by default because it is a real
    behavioural change and this project measures things before adopting them.
    """

    simulations: int = 160
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.1
    dirichlet_fraction: float = 0.25
    fpu_reduction: float = 0.0
    add_noise: bool = True


class Node:
    """One position in the tree, with per-edge statistics held as arrays.

    Statistics live on the *parent*, indexed by column, rather than on the
    child. It saves an object per unvisited move -- and at the root, six of the
    seven children may never be built at all.

    Every stored value is from **this node's** mover's point of view, so a
    backup flips sign at every level on the way up.
    """

    __slots__ = (
        "children",
        "expanded",
        "legal",
        "position",
        "prior",
        "terminal_value",
        "total_visits",
        "value_sum",
        "visits",
    )

    def __init__(self, position: Position) -> None:
        self.position = position
        self.children: list[Node | None] = [None] * WIDTH
        self.prior = np.zeros(WIDTH, dtype=np.float32)
        self.visits = np.zeros(WIDTH, dtype=np.int32)
        self.value_sum = np.zeros(WIDTH, dtype=np.float32)
        self.total_visits = 0
        self.expanded = False
        self.legal = position.legal_moves()
        # None means "not an ending"; otherwise the exact value of this position
        # to the side to move, which is -1 for a loss and 0 for a draw. A node
        # can never be a *win* for the side to move: you do not get a turn after
        # your opponent has already made four in a row.
        self.terminal_value: float | None = None
        if position.has_won():
            self.terminal_value = -1.0
        elif position.is_draw():
            self.terminal_value = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.terminal_value is not None

    def value(self) -> float:
        """This node's own value estimate: the visit-weighted mean of its edges."""
        if self.total_visits == 0:
            return 0.0
        return float(self.value_sum.sum() / self.total_visits)

    def select(self, c_puct: float, fpu_reduction: float) -> int:
        """The PUCT rule: pick the column maximising ``Q + U``.

        ``Q`` is what we have measured, ``U`` is what the prior promises decayed
        by how often we have already checked. Early on ``U`` dominates and the
        search follows the network; as visits accumulate ``Q`` takes over and
        the search follows the evidence.
        """
        visited = self.visits > 0
        q = np.where(visited, self.value_sum / np.maximum(self.visits, 1), 0.0)
        if fpu_reduction:
            q = np.where(visited, q, self.value() - fpu_reduction)
        # sqrt of the parent's visit count -- the standard PUCT numerator. The
        # max(1) keeps the very first selection from zeroing every U at once,
        # which would leave the choice to a tie-break rather than to the prior.
        u = c_puct * self.prior * np.sqrt(max(self.total_visits, 1)) / (1 + self.visits)
        score = q + u
        mask = np.full(WIDTH, -np.inf, dtype=np.float32)
        mask[self.legal] = 0.0
        return int(np.argmax(score + mask))


class Search:
    """One tree, one root, driven a simulation at a time.

    Usage is a loop of :meth:`descend` / :meth:`expand` until :attr:`done`.
    ``descend`` returns ``None`` when the simulation ended at a terminal
    position and needed no network call, so a caller must tolerate that rather
    than treating it as "finished".
    """

    __slots__ = ("config", "pending", "rng", "root", "simulations_done")

    def __init__(
        self,
        position: Position,
        config: MCTSConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config or MCTSConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.root = Node(position)
        self.simulations_done = 0
        # The leaf handed out by the last descend, with the path back to the
        # root, waiting for expand to supply its evaluation.
        self.pending: tuple[Node, list[tuple[Node, int]]] | None = None

    @property
    def done(self) -> bool:
        if self.root.is_terminal or not self.root.legal:
            return True
        return self.simulations_done >= self.config.simulations and self.pending is None

    def descend(self) -> Position | None:
        """Walk from the root to a leaf.

        Returns the leaf's position when it needs evaluating, or ``None`` when
        the walk ended on a decided position -- in which case the exact value
        has already been backed up and the simulation is spent.
        """
        if self.pending is not None:
            msg = "descend called twice without an intervening expand"
            raise RuntimeError(msg)
        if self.done:
            return None

        node = self.root
        path: list[tuple[Node, int]] = []
        while node.expanded and not node.is_terminal:
            col = node.select(self.config.c_puct, self.config.fpu_reduction)
            child = node.children[col]
            if child is None:
                child = Node(node.position.played(col))
                node.children[col] = child
            path.append((node, col))
            node = child

        if node.is_terminal:
            self._backup(path, node.terminal_value or 0.0)
            self.simulations_done += 1
            return None

        self.pending = (node, path)
        return node.position

    def expand(self, priors: np.ndarray, value: float) -> None:
        """Attach an evaluation to the leaf that :meth:`descend` handed out."""
        if self.pending is None:
            msg = "expand called with no pending leaf"
            raise RuntimeError(msg)
        node, path = self.pending
        self.pending = None

        node.prior = self._masked_priors(priors, node)
        node.expanded = True
        if node is self.root and self.config.add_noise:
            self._add_root_noise()

        self._backup(path, float(value))
        self.simulations_done += 1

    def _masked_priors(self, priors: np.ndarray, node: Node) -> np.ndarray:
        """Zero out full columns and renormalise over what is left.

        A network that puts mass on an illegal column is not wrong exactly --
        nothing taught it the column was full -- but a search that followed it
        there would be. Renormalising rather than merely masking keeps the U
        term's scale the same whether one column is full or five are.
        """
        masked = np.zeros(WIDTH, dtype=np.float32)
        masked[node.legal] = np.asarray(priors, dtype=np.float32)[node.legal]
        total = masked.sum()
        if total <= 0.0:
            # A degenerate or adversarial evaluator. Uniform over legal moves is
            # the honest fallback: it says nothing rather than something false.
            masked[node.legal] = 1.0 / len(node.legal)
            return masked
        return masked / total

    def _add_root_noise(self) -> None:
        legal = self.root.legal
        noise = self.rng.dirichlet([self.config.dirichlet_alpha] * len(legal))
        frac = self.config.dirichlet_fraction
        self.root.prior[legal] = (1 - frac) * self.root.prior[legal] + frac * noise

    @staticmethod
    def _backup(path: list[tuple[Node, int]], value: float) -> None:
        """Push a leaf value up the path, flipping sign at every ply.

        The leaf's value is what the leaf's *mover* thinks. One ply up it is the
        opponent's turn, and their gain is the negative of it. Getting this
        wrong produces a search that confidently walks into losses, and it is
        the single most common bug in an MCTS implementation -- hence
        ``test_backup_alternates_sign``.
        """
        v = value
        for node, col in reversed(path):
            v = -v
            node.visits[col] += 1
            node.value_sum[col] += v
            node.total_visits += 1

    # ------------------------------------------------------------- results

    def visit_counts(self) -> np.ndarray:
        return self.root.visits.copy()

    def root_value(self) -> float:
        return self.root.value()

    def best_move(self) -> int:
        """The most-visited legal column.

        The legality mask is not decoration. The first simulation expands the
        root and credits no edge, so at ``simulations=1`` every count is still
        zero and a bare ``argmax`` returns column 0 -- whether or not column 0
        has room. The caller then plays it, the board silently fails to change,
        and a game loop that waits for a win or a draw waits forever. Ties
        among equal counts break toward the centre, which is the order
        ``legal`` is already in.
        """
        if not self.root.legal:
            raise ValueError("no legal move: the root position is already over")
        counts = self.root.visits
        return max(self.root.legal, key=lambda column: counts[column])

    def policy(self, temperature: float = 1.0) -> np.ndarray:
        """Visit counts as a probability distribution.

        Temperature 0 means "play the most-visited move"; that is what you want
        once a self-play game is past its opening, and what you always want when
        the network is actually playing an opponent. Above 0 the counts are
        raised to ``1 / temperature`` and normalised, which is how the opening
        gets its variety.
        """
        counts = self.root.visits.astype(np.float64)
        if counts.sum() == 0:
            policy = np.zeros(WIDTH)
            if self.root.legal:
                policy[self.root.legal] = 1.0 / len(self.root.legal)
            return policy
        if temperature <= 0:
            policy = np.zeros(WIDTH)
            policy[int(np.argmax(counts))] = 1.0
            return policy
        # Divide by the max before the power: counts reach the hundreds and a
        # low temperature turns that into an overflow rather than a decision.
        scaled = (counts / counts.max()) ** (1.0 / temperature)
        return scaled / scaled.sum()


def run(
    position: Position,
    evaluate: Callable[[Position], Evaluation],
    config: MCTSConfig | None = None,
    rng: np.random.Generator | None = None,
) -> Search:
    """Run a complete search with a one-position-at-a-time evaluator."""
    search = Search(position, config, rng)
    while not search.done:
        leaf = search.descend()
        if leaf is None:
            continue
        priors, value = evaluate(leaf)
        search.expand(priors, value)
    return search


def run_batch(
    searches: Sequence[Search],
    evaluate: Callable[[list[Position]], tuple[np.ndarray, np.ndarray]],
) -> None:
    """Drive many searches together, evaluating all their leaves in one call.

    Each pass asks every unfinished search for a leaf, evaluates the whole
    collection at once, and feeds each answer back. The searches stay
    independent -- no shared tree, no virtual loss -- so this is parallelism in
    the network call only. That is the part that benefits: the tree walk is
    Python either way, but a batched forward pass over 64 boards costs barely
    more than one board, and self-play spends most of its time in the network.
    """
    while True:
        active: list[Search] = []
        leaves: list[Position] = []
        for search in searches:
            if search.done:
                continue
            leaf = search.descend()
            if leaf is not None:
                active.append(search)
                leaves.append(leaf)
        if not leaves:
            # Either everything finished, or every remaining simulation ended on
            # a terminal node and consumed itself. Both mean: check again, and
            # stop when nothing is left to do.
            if all(s.done for s in searches):
                return
            continue
        priors, values = evaluate(leaves)
        for i, search in enumerate(active):
            search.expand(priors[i], float(values[i]))
