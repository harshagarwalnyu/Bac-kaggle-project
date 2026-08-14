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

The layout borrows its idiom from lichess's analysis board, because that page
solves exactly this problem: show a lot of engine output without burying the game.

- **A vertical eval bar** beside the board, from the mover's point of view. It
  shows `#n` for a proven mate, `=` for a proven draw, and a number otherwise.
- **The engine lines list**, sorted best-first, one row per legal column. Click a
  row to play it. Only the top line shows a principal variation; the rest show
  just their own move, because inventing continuations for them would be
  fabrication.
- **Two brains per line.** The search's score, and next to it a signed,
  centre-anchored bar for the network's opinion of the same move. Proven verdicts
  turn green (win) or red (loss); guesses stay blue.
- **The disagreement callout.** Names the column where the network most disagrees
  with a *proof*. Only proofs count: comparing the network against another heuristic
  would be two opinions, and neither is evidence about the other.
- **Ghost pieces** showing the principal variation, numbered in play order.
- **Live search stats** — depth, nodes, kn/s, transposition-table hits, milliseconds.
- **Assist toggle.** Ticking it outlines the column the search would play *for you*
  and offers a one-key button, so the engine works as a coach rather than only as
  an opponent.
- **Difficulty as *n*-th best true move**, never random noise. Even at 0 the bot
  takes a win that is on the board and refuses to walk into a mate in two — a bot
  that plays well and then randomly throws a piece away reads as a bug, not as
  easy mode.
- **Tabs** for Analysis, the move list, and past games.

## Solver mode (difficulty 6)

Connect 4 is solved — the first player wins by move 41 with perfect play. Skill 6
is a separate mode rather than another notch on the dial, because what changes is
the *budget*, not the move choice: six times the clock, and a transposition table
that persists between moves. That second part matters more than the first.
Consecutive searches in one game overlap enormously, so keeping the table turns
each move into a continuation of the last rather than a fresh start. Entries are
keyed by position, not by search, so the reuse is sound.

It does **not** claim a solve from the empty board — proving that takes billions of
nodes, which CPython is not going to do inside a web request. From roughly the
eighth stone onward the search does resolve whole lines exactly, and the UI lights
the `proven` badge only where it genuinely did. The mode also carries a banner
saying plainly that the dataset network is out of the driving seat here. Claiming
a solve we did not compute would be the one dishonest thing this project could
ship.

## Past games

Finished games are appended to `data/games.jsonl` and shown under the History tab,
with a win/loss/draw record. The archive stores the **move list**, not a board, so
a game replays exactly and can be re-analysed later by a stronger search.

JSONL rather than SQLite, deliberately: a crash mid-append costs the line being
written and leaves every earlier game intact, a truncated final line is simply
skipped on load, unknown fields from a newer version are ignored, and the file is
readable without the program.

The durability guarantee is worth stating exactly, because the honest version is
narrower than "cannot lose a game". A game is held in memory the moment it ends
and written immediately after, so a failed write — full disk, unwritable path —
leaves it visible and replayable for the rest of that process, and is logged
rather than swallowed. It is **durable only once the write succeeds**: a game
whose write failed is gone after a restart. What the format buys is that its loss
is confined to itself.

Because a game *is* its move list, two things fall out for free:

- `POST /api/games` takes an optional `moves` opening, so a position is
  shareable and a saved game is resumable. Illegal sequences are rejected, not
  clamped — a typo must not quietly produce a different position.
- `POST /api/history/{id}/rematch` replays an archived game into a new one,
  one ply short of the end by default: the position worth thinking about again
  is the one before the mistake. Colours and difficulty are inherited, because
  a rematch you win by quietly switching sides is not a rematch.

## Layout

```
src/connect4/
  bitboard.py   7-bits-per-column bitboard; wins detected in 4 shift-and-mask ops
  engine.py     negamax + alpha-beta, transposition table, iterative deepening
  dataset.py    parsing, validation, 98 features (84 raw planes + 14 engineered)
  model.py      MLP written from scratch in NumPy — forward, backward, Adam
  history.py    append-only JSONL archive of finished games
  api.py        FastAPI; a game is stored as its move list, not as a board
web/            vanilla HTML/CSS/JS; the DOM is a pure function of one state object
scripts/train.py     trains the evaluator, against two baselines
scripts/validate.py  the experiments above
tests/          279 tests
```

### Configuration

Every knob is an environment variable, so nothing needs editing to run it
differently:

| variable | default | meaning |
|---|---|---|
| `CONNECT4_HOST` / `CONNECT4_PORT` | `127.0.0.1` / `8000` | where to serve |
| `CONNECT4_TIME_LIMIT` | `2.0` | search budget per move, seconds |
| `CONNECT4_SOLVER_TIME_LIMIT` | `12.0` | budget in solver mode |
| `CONNECT4_HISTORY` | `data/games.jsonl` | where the game archive is written |

`CONNECT4_HISTORY` has three states, not two: unset uses the default file, a
path writes there, and an **empty** value keeps games in memory only and never
touches the disk — which is what a shared machine or a throwaway container
wants.

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
uv run python -m pytest              # 279 tests
uv run python -m scripts.train       # downloads the data, trains, prints baselines
uv run python -m scripts.validate    # the three experiments above
uv run python -m connect4.api        # play
```

The dataset is fetched from the UCI mirror on first run; no Kaggle credentials needed.
