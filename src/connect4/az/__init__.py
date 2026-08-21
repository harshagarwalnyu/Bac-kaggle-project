"""AlphaZero-style self-play for Connect 4.

The rest of this project searches with a *hand-written* evaluator, and the
network only publishes an opinion beside it. This package is where that changes:
a network with a policy head and a value head, a PUCT tree search that uses the
policy as its prior and the value in place of a rollout, and a self-play loop
that generates its own training data.

Three deliberate boundaries:

* :mod:`connect4.az.features` and :mod:`connect4.az.mcts` import **numpy only**.
  The search takes its evaluator as a callable, so it can be tested against a
  hand-written stub with no framework installed at all.
* :mod:`connect4.az.net` is the only module that imports torch, and torch is an
  optional extra (``uv sync --extra az``). Installing this project to play a
  game still costs nothing.
* Nothing here touches :mod:`connect4.engine`. The alpha-beta search remains
  exactly what it was; the network becomes an *alternative* player, and the
  two get graded against each other rather than tangled together.
"""
