# Connect 4 — a glass-box bot

A Connect 4 bot you can play in the browser, built on the
[UCI/Kaggle connect-4 dataset](https://www.kaggle.com/datasets/tbrewer/connect-4)
(67,557 solved 8-ply positions).

The twist: **two brains publish an opinion on every single move, and the UI shows
both.** A bitboard alpha-beta search that sometimes returns *proofs* ("loss in 6"),
and a from-scratch NumPy neural network trained on the dataset that only ever has
an *opinion*. You watch, live, where the learned model agrees with perfect play and
where it is confidently wrong.

```bash
uv run python -m connect4.api      # → http://127.0.0.1:8000
```

That is the whole install. No build step, no CDN, no framework.

---

## Why this design

The obvious use of a labelled dataset is to train a classifier and ship it as the
bot. That bot would be weak, and — worse — you could not tell *how* weak, because
nothing would be checking it. Every position in this dataset is exactly 8 plies
deep. A real game is 42.

So the dataset does two honest jobs instead:

1. **It trains the evaluator** — the leaf heuristic a search can call.
2. **It is a ground-truth test set.** The labels are perfect-play outcomes from
   John Tromp's solver, so we can grade the bot's verdicts against *truth*, which
   is a luxury you almost never get.

And the search is what actually plays.

## What the measurements said — including the inconvenient part

`scripts/validate.py` runs both engines at **fixed depth**, so any difference is
attributable to the evaluator rather than to one engine getting more nodes for its
money. (The network costs 1.54× the wall clock per search at equal depth; caching
evaluations on the position key absorbs most of the cost.)

**A. Verdict agreement at ply 8**, 300 held-out positions, graded against the solver:

| predictor | 3-class | decisive (draws excluded) |
|---|---|---|
| network alone, **no search** | **0.840** | **0.923** |
| search + hand-written heuristic | 0.670 | 0.734 |
| search + network | 0.710 | 0.782 |

**B. Head to head**, 14 games, alternating colours, forced distinct openings:

> search+heuristic **8** — **4** search+network (2 draws)

**C. Does the search earn its keep?**

> search+network **13** — **0** network alone, greedy (1 draw)

Read together, those three tables say something specific and slightly awkward:

- The network is **excellent on the distribution it was trained on** and beats the
  hand-written heuristic badly there (0.923 vs 0.734 decisive).
- It is **unreliable off that distribution**. A depth-7 search from a ply-8 board
  asks the network about ply-15 leaves — boards it has never seen — and it answers
  with enormous confidence anyway. That is why it loses the head-to-head.
- **Search is worth far more than either evaluator.** 13–0 is not a close call.

So the shipped bot searches with the *hand-written* evaluator, and the network keeps
the job it is genuinely good at: publishing a per-column opinion next to the search's,
where you can see it. That decision is a result, not a default —
[`src/connect4/api.py`](src/connect4/api.py) says so at the line where it is made.

## What you see in the UI

- **Per-column bars, two brains.** Signed and centred — left of the midline is bad
  for the mover. Proven verdicts turn green (win) or red (loss); the search's
  guesses stay blue.
- **The disagreement callout.** Names the column where the network most disagrees
  with a *proof*. Only proofs count: comparing the network against another heuristic
  would be two opinions, and neither is evidence about the other.
- **Ghost pieces** showing the principal variation, numbered in play order.
- **Live search stats** — depth, nodes, transposition-table hits, milliseconds.
- **Difficulty as *n*-th best true move**, never random noise. Even at 0 the bot
  takes a win that is on the board and refuses to walk into a mate in two — a bot
  that plays well and then randomly throws a piece away reads as a bug, not as
  easy mode.

## Layout

```
src/connect4/
  bitboard.py   7-bits-per-column bitboard; wins detected in 4 shift-and-mask ops
  engine.py     negamax + alpha-beta, transposition table, iterative deepening
  dataset.py    parsing, validation, 98 features (84 raw planes + 14 engineered)
  model.py      MLP written from scratch in NumPy — forward, backward, Adam
  api.py        FastAPI; a game is stored as its move list, not as a board
web/            vanilla HTML/CSS/JS; the DOM is a pure function of one state object
scripts/train.py     trains the evaluator, against two baselines
scripts/validate.py  the experiments above
tests/          181 tests
```

### The two claims worth checking first

**The bitboard is right.** `tests/test_bitboard.py` uses differential testing: a
slow, obviously-correct nested-loop reference is asserted to agree with the fast
shift-and-mask version over thousands of random positions. A wrong shift constant
works for a hundred positions and then silently miscounts a diagonal.

**The backward pass is right.** `tests/test_model.py` gradient-checks every layer
against central differences in float64, with and without weight decay. That is the
only honest proof a hand-written backprop is correct, and it is the reason I can
claim to understand every line of the training code.

## The model

`(98 → 128 → 64 → 3)`, ≈21k parameters. He init, ReLU, numerically stable softmax,
cross-entropy, hand-written Adam with bias correction, early stopping with
best-weight restore. Test set, against controls:

| model | accuracy | draw recall |
|---|---|---|
| always predict "win" (65.8% of rows) | 0.6583 | 0.000 |
| logistic regression, same features | 0.7619 | 0.007 |
| **from-scratch MLP** | **0.8507** | **0.282** |

Draw recall is the number that matters. Draws are 9.6% of the data, and the linear
control essentially never predicts one — it buys its 76% by ignoring the class
entirely. Accuracy alone would have hidden that completely.

## Reproducing

```bash
uv sync
uv run python -m pytest              # 181 tests
uv run python -m scripts.train       # downloads the data, trains, prints baselines
uv run python -m scripts.validate    # the three experiments above
uv run python -m connect4.api        # play
```

The dataset is fetched from the UCI mirror on first run; no Kaggle credentials needed.
