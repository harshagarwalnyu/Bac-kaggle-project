# Connect 4 — a glass-box bot

[![tests](https://github.com/harshagarwalnyu/Bac-kaggle-project/actions/workflows/tests.yml/badge.svg)](https://github.com/harshagarwalnyu/Bac-kaggle-project/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Connect 4 bot you can play in the browser, built on the
[UCI `connect-4` opening database](https://archive.ics.uci.edu/dataset/26/connect+4)
(67,557 legal 8-ply positions, each labelled with its perfect-play outcome by John
Tromp's solver). See [which dataset, and why](#which-connect-4-dataset-and-why) --
there are two very different files under this name.

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

### Which connect-4 dataset, and why

Two well-known datasets share the name, and they are not interchangeable:

| | UCI `connect-4` (**used here**) | Kaggle "Connect-4 Game Dataset" |
|---|---|---|
| rows | 67,557 | 376,641 |
| a row is | one **position**, at exactly 8 plies | one **finished game**'s final board |
| cell order | column-major, bottom-up (`a1..a6, b1..b6`) | left-to-right, top-to-bottom |
| encoding | `x` / `o` / `b` | `1` / `-1` / `0` |
| label | perfect-play outcome from **Tromp's solver** | who actually won that game |
| labels produced by | an exact solver | self-play *while a network was being trained* |

The label column is the whole reason for the choice. This project's premise is
grading a bot's verdicts against **truth**, and only one of these files contains
truth: UCI's labels are game-theoretic values, so a disagreement between my search
and the label is unambiguously my search being wrong. The Kaggle file's labels are
the observed results of games between two weak, still-learning agents, so a
disagreement means nothing in particular.

Two further problems with the Kaggle file for *this* design. Its rows are **final**
boards, and a finished board is terminal -- exactly the node type a search scores
exactly and never asks an evaluator about, so it is close to useless as leaf
training data. And predicting the winner from a final board is near-trivial, because
the winning four-in-a-row is sitting right there in the input: a model can score
very well on it while learning nothing about evaluating a live position.

The honest cost of the choice is the one the measurements below expose: UCI is
**single-depth**, so the network never sees an opening or an endgame, and that is
precisely why it fails off-distribution. The Kaggle file has the opposite trade --
far more coverage, far weaker labels. Fixing this properly means neither file: it
means solver-labelled positions sampled across *many* depths.

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

### The difficulty dial is not evenly spaced

`scripts/tournament.py` plays every difficulty against every other, both seats,
from a shared book of openings — the seat and the opening are the only things
that vary, because the engine is deterministic and replaying a pairing would
just replay one game. 168 games at a 0.1s base clock, fitted to a Bradley-Terry
rating and put on the Elo scale:

| level | elo | step over the one below | expected score vs it |
|-------|-----|-------------------------|----------------------|
| 6 | 562 | 114 | 66% |
| 5 | 448 | **206** | 77% |
| 4 | 242 | 128 | 68% |
| 3 | 114 | **6** | 51% |
| 2 | 107 | 72 | 60% |
| 1 | 35 | 35 | 55% |
| 0 | 0 | — | — |

The dial reads like seven even notches and is not one. **Levels 2 and 3 are the
same opponent** — six Elo apart, a coin flip — and the largest jump on the whole
ladder is 4 to 5, nearly double the celebrated 5-to-6 step.

That falls out of how the levels are defined. Difficulties 0–4 play the
`(5 - skill)`-th best move from one search; 5 plays the best. So the dial is
really an index into a ranking, and the gap between the 3rd- and 4th-best of
seven columns is much smaller than the gap between the 1st and 2nd. The low end
compresses because the moves themselves are close together, not because the
search is weaker. Anyone rebalancing the dial should start there, not at the
solver.

The Elo fit credits every pair with one virtual draw. Without it an undefeated
level has no finite rating, and level 6 is undefeated against everything below
it. That is a deliberate thumb on the scale: an honest 562 with a stated prior
beats an infinity.

## Solver mode (difficulty 6)

Connect 4 is solved — the first player wins by move 41 with perfect play. Skill 6
is a separate mode rather than another notch on the dial, because what changes is
the *budget*, not the move choice: twelve times the clock (`CONNECT4_TIME_LIMIT`
is 1.0s, `CONNECT4_SOLVER_TIME_LIMIT` is 12.0s), and a transposition table that
persists between moves. Consecutive searches in one game overlap enormously, so
keeping the table turns each move into a continuation of the last rather than a
fresh start. Entries are keyed by position, not by search, so the reuse is sound.

Which of the two knobs actually buys the strength is a fair question and
`scripts/ablate.py` exists to answer it — it crosses the two and plays the four
resulting configurations against each other. At a scaled-down clock the clock
wins that comparison and the table contributes nothing measurable, but that run
is a hostile test for the table: search overlap between consecutive moves is
precisely what a persistent table sells, and overlap grows with the budget. Run
it at `--time 1.0` before believing either answer.

It does **not** claim a solve from the empty board — proving that takes billions of
nodes, which CPython is not going to do inside a web request. Walking a full game
under the solver's own settings, the first position it resolves completely —
every legal column carrying a proven score — arrives at **fourteen stones**,
and from there to the end every ply but one comes back fully proven, most of
them in well under a second, because the persistent table has already seen the
sub-positions. Before fourteen it is partial and not monotonic: four of seven
columns proven at seven stones, none at eight. The UI lights the `proven` badge
per column, only where the search genuinely resolved that column. The mode also
carries a banner saying plainly that the dataset network is out of the driving
seat here. Claiming
a solve we did not compute would be the one dishonest thing this project could
ship.

## What a move costs

A turn is two searches and neither is optional: one picks the bot's reply, the
other describes the position that reply leaves you in -- and that second one *is*
the analysis panel. What the server gets to choose is *when* they happen.

Measured against a freshly started server at the default one-second budget, mean
over the turns played:

| | your move | bot's reply | turn |
|---|---|---|---|
| a search per request | 1.06s | 2.08s | **3.13s** |
| analysis memoised per position | 1.02s | 1.03s | **2.05s** |
| plus thinking while you decide | 0.05s | 1.07s | **1.13s** |

The first row wasted an entire search per turn. Playing a stone analysed the
position it created, and a moment later the bot analysed that same position again
to choose its move -- the identical search, twice. `AnalysisCache` makes the
second one a lookup. It is sound because `analyse` clears its transposition table
on every call, which makes an analysis a pure function of the position rather
than of the searches that came before it.

The third row is the `Warmer`. Between the bot's reply and your next stone the
server has nothing to do, so it analyses the positions you could create -- best
first, in the order the panel is already showing you. It runs on its own engine
with the same budget and the same evaluator, so a warmed answer is the answer a
request would have got; a search interrupted by an arriving request is thrown
away rather than stored, so a cache hit can never be thinner than a miss. Solver
mode is never warmed, because that engine keeps its table between searches and
what it can prove therefore depends on what it was asked before.

The costs, stated plainly. Someone who clicks the instant the bot moves gains
nothing and pays about 4% (2.14s against 2.05s) for background work that gets
discarded, and a game in progress keeps a second core busy for a few seconds a
turn; `CONNECT4_WARM=0` turns it off. Measured in a real browser, from the click
to the bot's stone appearing: **1.18-1.30s** on a cold server, **0.14-0.22s**
once an opening has been seen before.

### And a node costs less than it did

The budget buys a fixed number of seconds; what those seconds are worth depends
on how much searching fits inside them. Profiling one fixed-depth search said the
time was going somewhere unglamorous -- a third of it inside the threat-map
function, called nearly six times per node for a board that was not changing
between the calls.

Three fixes, all bookkeeping rather than cleverness:

- **Remember the threat maps.** Every node asks for them two or three times over
  -- once to filter out losing replies, once to rank the moves, once more at a
  leaf by the evaluator -- and the answer cannot change while the board does not.
- **Rank the moves only when a second one is actually wanted.** Ranking costs a
  board and a threat map per column, and the transposition table's move usually
  causes a cutoff on its own, so ranking the other six was work whose result was
  thrown away. It now happens on demand, and one ply above the leaves not at all:
  there, searching a child is cheaper than deciding which child to search first.
- **Build each child position once.** The ranking built all of them and discarded
  them; the search then rebuilt the one it wanted.

Together, on ten benchmark positions: **27,000 to 69,000 nodes per second**, which
buys one extra ply of search per second on nine of the ten. The play is not
merely faster but stronger, and the answers did not drift: every column's score
at every depth from 1 to 7, across those ten positions, is identical before and
after. (One of the seventy ghost-piece lines now ends on a different but
equally-valued move.)

Two things were measured and rejected, which is the more useful half of the
exercise: principal variation search cut the tree by 5.6% without moving the
clock, and an unrolled threat map came in inside the noise. Neither was worth
what it cost to read.

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
  az/           AlphaZero: PUCT search, policy-value net, self-play, arena
                (the only torch code in the project, and an optional extra)
web/            vanilla HTML/CSS/JS; the DOM is a pure function of one state object
web/tests/      32 front-end tests; no dependencies, no build step
scripts/train.py     trains the evaluator, against two baselines
scripts/validate.py  the experiments above
scripts/az_pretrain.py  warm-starts the value head on solver-exact labels
scripts/train_az.py     the self-play loop: play, learn, gate, promote
scripts/az_arena.py     grades a checkpoint against fixed opponents
tests/          471 tests (425 without the optional `az` extra)
.github/        the workflow that runs both suites on every push
```

### Configuration

Every knob is an environment variable, so nothing needs editing to run it
differently:

| variable | default | meaning |
|---|---|---|
| `CONNECT4_HOST` / `CONNECT4_PORT` | `127.0.0.1` / `8000` | where to serve |
| `CONNECT4_TIME_LIMIT` | `1.0` | search budget per move, seconds |
| `CONNECT4_SOLVER_TIME_LIMIT` | `12.0` | budget in solver mode |
| `CONNECT4_HISTORY` | `data/games.jsonl` | where the game archive is written |
| `CONNECT4_WARM` | `1` | think ahead while waiting for the human; `0` turns it off |

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

**The page says what the board says.** `web/tests/` runs `web/app.js` itself —
boot path included — against a stub DOM and a fake server that answers in the
shapes `api.py` really sends, `turn: 0` on a finished game included. It needs no
npm install and no browser: `node --test web/tests/app.test.js`. Three defects
were found this way and are now regression tests: stones coloured by seat rather
than by role (so a bot-first game handed the bot the human's yellow), a legend
whose colour keys contradicted the marks they explained, and a refused move that
still let the bot reply — costing a tempo and wiping the error message that
explained the refusal.

## AlphaZero mode (`connect4.az`)

The MLP above is a *classifier*. It reads a position and predicts the
game-theoretic result, and the front end draws it as blue bars — but it does not
choose moves. Every move the shipped bot plays comes from alpha-beta with a
hand-written evaluator. That is the honest state of the project, and it is the
gap this package closes.

`src/connect4/az/` is a self-contained AlphaZero implementation: a policy-value
network, PUCT search that uses the policy as a prior and the value in place of a
rollout, and a self-play loop that trains on the search's own visit counts.

**The framework rule changed here, deliberately.** `connect4.model` is a
hand-written NumPy MLP because a small dense classifier is genuinely legible
when you write the backward pass yourself. A residual convolutional tower
trained by self-play is not that; hand-rolling conv backprop would teach nobody
anything and would very likely be subtly wrong. So this package uses PyTorch,
as an **optional extra**. Playing a game against the shipped bot still needs
neither torch nor any of these files.

```bash
uv sync --extra az
uv run python -m scripts.az_pretrain                     # warm-start the value head
uv run python -m scripts.train_az --iterations 20        # self-play
uv run python -m scripts.az_arena checkpoints/az/champion.pt
```

**Sized for the machine, not for the paper.** AlphaGo Zero used 20 residual
blocks of 256 filters. The default here is 3 blocks of 32 — about 60k
parameters — because on an 8-core CPU the bottleneck is not capacity, it is how
many self-play games per hour the network can generate. Measured on a batch of
128 boards, best of 15: 64×4 gives 5,479 boards/s, 32×3 gives 17,943. Going
wider costs 3.3× the time for 5× the parameters, and that time is games not
played.

**The gate is the whole safety mechanism.** Each iteration trains a challenger,
plays it against the reigning champion over a book of forced openings from both
seats, and promotes only on a score of 0.55 or better. Training loss is *not*
evidence of strength here — it is measured against targets the network's own
search produced, so it says how self-consistent the network is and nothing
about whether it plays better. `scripts/az_arena.py` reports the number that
does mean something: the score against the shipped alpha-beta engine at each
skill level, plus a *uniform-search* control that runs the same MCTS with a
network that knows nothing. That control is what separates what the network
contributed from what the search contributed.

**The warm start uses real labels, and only where they apply.** The UCI file is
67,557 positions with solver-exact outcomes, so `scripts/az_pretrain.py` fits
the value head to them before self-play begins. It does **not** touch the policy
head: the file says who wins, not which move to play, so there is no policy
target in it and inventing one would defeat the point of using real data. On the
run recorded here it took validation MSE from 0.731 (predict the mean) to 0.188,
with 94% sign agreement on decisive positions. The trunk is shared, so the
policy head still starts from a representation that has seen exact labels.

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
uv sync --extra dev
uv run ruff check .                        # lint
uv run python -m pytest                    # 471 tests (425 without --extra az)
node --test web/tests/app.test.js          # 32 front-end tests, no npm install
uv run python -m scripts.train             # downloads the data, trains, prints baselines
uv run python -m scripts.validate          # the three experiments above
uv run python -m connect4.api              # play
```

The dataset is fetched from the UCI archive on first run; nothing needs a Kaggle account.

Those first three lines are exactly what CI runs on every push and pull request
([`.github/workflows/tests.yml`](.github/workflows/tests.yml)) — the python suite on
3.14, the front-end suite on node 20 and 24. There is no step in CI that you cannot
run yourself, and no step here that CI skips.

The interpreter is pinned in [`.python-version`](.python-version), so `uv` picks 3.14
without being told and downloads it if the machine does not have it.

## License

The code is [MIT](LICENSE).

The dataset is not mine to license: the UCI `connect-4` database is distributed by the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/26/connect+4) under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and the perfect-play labels in
it are John Tromp's work. Nothing in this repository redistributes it — `scripts/train.py`
fetches it from the archive at first run, which is also why `data/raw/` is gitignored.
