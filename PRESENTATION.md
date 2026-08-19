# Presenting this project — the full script

Read top to bottom. Every **bold** number is real and reproducible from this repo.

---

## 0. The 30-second version (open with this)

> "It's a Connect 4 bot you play in the browser. The twist is that it's a
> **glass-box** bot: two different brains give an opinion on every single move,
> and the UI shows both side by side. One is a bitboard alpha-beta search that
> sometimes returns an actual *proof* — 'you lose in 6, here's the line'. The
> other is a neural network I wrote from scratch in NumPy, trained on the UCI
> Connect-4 dataset, which only ever has an *opinion*. You can watch, live,
> where the learned model agrees with perfect play and where it's confidently
> wrong. And the interesting finding is that the dataset model **lost** the job
> of playing — I have the measurements that took it out of the driver's seat."

That last sentence is what separates this from a normal ML project. Most
projects show you the thing that worked. This one shows you the experiment that
killed the obvious approach, and what replaced it.

---

## 1. What I used

**Stack — deliberately tiny.**

| layer | what | why this and not the obvious thing |
|---|---|---|
| board | bitboards (plain Python ints) | A 6x7 list-of-lists needs dozens of comparisons per node to check a win; the bitboard needs 4 shift-and-mask ops. That is the difference between thousands and hundreds of thousands of nodes per second. |
| search | negamax + alpha-beta, transposition table, iterative deepening | Hand-written, no game library. |
| ML | **NumPy only** — no PyTorch, TensorFlow or scikit-learn | The deliverable was being able to defend every line. Forward pass, backprop, Adam with bias correction, He init, early stopping — all hand-written and gradient-checked. |
| data | **UCI `connect-4` opening database**, 67,557 rows | Downloaded automatically on first run from the UCI archive. Not the 376k-row Kaggle file of a similar name — see section 2. |
| API | FastAPI + Pydantic | Handlers are sync; validation lives in the request models, so a bad column is a 422 and never an IndexError deeper down. |
| UI | **vanilla HTML/CSS/JS** — no React, no build step, no CDN | The DOM is a pure function of one state object. The whole install is one command. |
| storage | append-only JSONL | Not SQLite — see section 6. |
| tests | pytest, 342 tests, including differential tests and gradient checks | |
| tooling | `uv` | `uv sync`, then `uv run python -m connect4.api`. |

**If they ask "did you use AI?"** — answer straight: yes, as a pair programmer
and as a reviewer, and say what that concretely bought. The code was put through
independent AI code review (GitHub Copilot CLI, an external Antigravity/Gemini
agent, and CodeRabbit on the pull requests). That surfaced two real defects I had
missed: a race where two simultaneous requests on one game could both pass the
turn check and play two plies for one side, and a rematch endpoint that created a
game *before* validating it, so a rejected rematch could still evict a live game
to make room for a dead one. Both are fixed, and both have a regression test that
fails without the fix. That is a better answer than "no", and it is true.

---

## 2. The dataset — explain it exactly like this

This is the part most people get wrong, so be precise here.

### Vocabulary — define these three words before you use them

**Ply = one single move by one player.** One stone dropped. It is *not* a
"move" in the everyday sense of "I go, then you go" — that would be two plies.
So the ply count of a board is just **how many stones are on it**.

**The board is 7 columns x 6 rows = 42 cells.** So a game that fills the board
completely is **42 plies** long. 42 is the maximum a Connect 4 game can be; most
end sooner because somebody connects four.

Putting those together, "ply *n*" and "*n* stones on the board" are the same
statement:

```text
ply 0    board empty                    .......   game start
ply 8    8 stones,  4 each     ~19% full  ##.....   <-- EVERY row of my dataset
ply 15   15 stones             ~36% full  ####...
ply 30   30 stones             ~71% full  ######.   late midgame
ply 42   42 stones            100% full  #######   board full = draw
```

**Why 8 plies means "4 each", and why that matters.** Players alternate starting
with `x`, so after an even number of stones both have played the same number: 8
plies = 4 `x` + 4 `o`, and it is **`x`'s turn again**. That is why every label in
the file is from `x`'s point of view *and* from the point of view of the player
about to move — the two coincide, so no sign flip is ever needed in my code.

**And why "ply 30" keeps coming up.** A search looks *ahead*. If the bot is
standing on a ply-8 board and searches 7 moves deep, the boards at the bottom of
its search tree — the **leaves**, the ones it actually hands to the evaluator —
are ply-15 boards. Later in a real game, standing at ply 23 and searching 7 deep,
the leaves are ply-30 boards. **The evaluator is never asked about the board you
are looking at. It is asked about boards several moves into the future.** My
network only ever saw ply-8 boards in training, so from the very first move it is
being asked about positions unlike anything in its training set. That single fact
is the explanation for the whole result in section 3.

### Say this first: there are two datasets called "connect-4"

Get in front of it, because if someone Googles "Kaggle connect 4 dataset" mid-talk
they will land on the *other* one and think your numbers are wrong.

| | UCI `connect-4` — **the one I used** | Kaggle "Connect-4 Game Dataset" |
|---|---|---|
| rows | **67,557** | **376,641** |
| a row is | one **position**, at exactly 8 plies | one **finished game**'s final board |
| cell order | column-major bottom-up (`a1..a6, b1..b6`) | left-to-right, top-to-bottom |
| encoding | `x` / `o` / `b` | `1` / `-1` / `0` |
| label | perfect-play outcome, from **Tromp's solver** | who actually won that game |
| labels came from | an exact solver | self-play *while a network was still training* |

**Why I picked the smaller one — three reasons, say them in this order:**

1. **The label is the entire point.** This project grades a bot's verdicts against
   **truth**. UCI's labels are game-theoretic values, so when my search disagrees
   with a label, my search is unambiguously wrong. The Kaggle file's labels are the
   results of games between two weak, still-learning agents — a disagreement there
   tells you nothing.
2. **Its rows are *final* boards.** A finished board is terminal, and terminal is
   exactly the node a search scores exactly and never asks an evaluator about. So
   it is close to useless as leaf-evaluator training data. To use it you would have
   to reconstruct the intermediate positions, and the file does not contain the
   move order — only the final grid.
3. **The task on it is near-trivial.** The winning four-in-a-row is sitting right
   there in the input. A model can score very well on that while learning nothing
   about evaluating a *live* position.

**Then volunteer the cost, before they find it** — this is the strongest thing you
can do here: "The price of that choice is that UCI is **single-depth**. My network
never sees an opening or an endgame, and that is precisely why it fails
off-distribution in table B. The Kaggle file has the opposite trade: far more
coverage, far weaker labels. The real fix is neither file — it is solver-labelled
positions sampled across many depths, which is exactly what I said I'd do next."

That turns a gotcha into evidence you understood the trade-off rather than
stumbled into it.

### What the file actually contains

- **67,557 rows.** Each row is one Connect 4 **position**, not a game.
- Format: **42 fields + 1 label**. The 42 fields are the 42 cells, each one of
  `x`, `o` or `b` (blank). Field order is **column-major, bottom-up** —
  `a1,a2,...,a6, b1,...,b6, ...` — where `a` is the leftmost column and row 1 is
  the bottom.
- The label is `win`, `loss` or `draw`, at **65.83% / 24.62% / 9.55%**.

### The three things that actually matter about it

**(1) It is not a dataset of games. It is a dataset of one specific depth.**
Every position has **exactly 8 stones** — 4 each. The dataset is "all legal
8-ply positions where nobody has won yet and the next move is not forced." So a
model trained on it has seen **nothing** of the opening and **nothing** of the
endgame. A real game is 42 plies. Asking this model about a 30-ply board is
**extrapolation, not inference** — and I measured how badly that goes rather
than assuming it was fine.

**(2) The labels are perfect-play outcomes, not observed results.**
They come from **John Tromp's solver**. The label is not "somebody won this
game"; it is "with perfect play by both sides from here, `x` wins / loses /
draws." That is **ground truth**, and it has two consequences: a model fitting
these labels is literally learning to approximate a solver — which is why it
makes sense to compare it *against* one — and I get to grade my bot's verdicts
against truth, which is a luxury you almost never have in ML.

**(3) 8 plies means 4 stones each, so `x` is always the player to move.**
So the label is always **from the point of view of the side to move** — the same
convention my search engine uses internally. No sign flip is needed anywhere in
the codebase, which removes the single most likely source of a silent bug in a
project like this.

### How a row becomes something the model can eat

`src/connect4/dataset.py`, in order:

1. **Parse** (`parse_line`) — field `i` maps to column `i // 6`, row `i % 6`.
   The dataset's layout and my bit layout agree exactly, so there is no
   transposition or flipping. The output is a `Position` with `moves=8`.
2. **Validate** (`validate_sample`) — point at this one. For *every* row I assert
   the invariants the documentation promises: exactly 4 stones each, nobody has
   already won, and **no floating stones** (every occupied cell has support
   beneath it). Why: a wrong stride constant produces plausible-looking boards
   for a hundred rows and then silently miscounts. This catches an off-by-one
   instantly, instead of three days later as "the model just isn't learning".
   - Honest footnote worth saying out loud: the docs also promise "the next move
     is not forced", but *forced* there is a game-theoretic property, not a board
     property — you cannot check it without a solver. So I assert only the half
     that is decidable, and I say so in the code.
3. **Encode** (`encode`) — **98 features**, in two blocks:
   - **84 raw occupancy** — one binary plane for the mover's stones, one for the
     opponent's, 42 cells each. I deliberately **left out the "empty" plane**: it
     is exactly `1 - mover - opponent`, so it carries no information the network
     cannot derive, and it would have inflated the input by 50%.
   - **14 engineered** — domain knowledge that is hard to learn from only 67k
     examples: mover and opponent threat counts, **odd/even threat counts**, the
     7 column heights, and centre control.
   - **The odd/even split is the one to explain.** Connect 4 has a parity
     theorem: because both players fill columns from the bottom, the first player
     tends to win on *odd* rows and the second on *even* rows. So a threat's
     **row parity matters more than the raw threat count**. Handing the network
     that directly is worth more than another hidden layer.
   - Everything is scaled to roughly [0, 1], so no single feature dominates the
     first layer's gradients.
4. **Split** (`stratified_split`) — 70/15/15, **stratified**, seeded. Stratified
   because the classes are 66/25/10 skewed: a plain random split can hand the
   test set a visibly different draw rate by pure chance, which then shows up as
   a mysterious train/test gap that has nothing to do with the model.
5. **Cache** — encoded arrays are written to `.npz`, so the encode cost is paid
   once.

### What the dataset is *used for* — the design decision

The obvious move is: train a classifier on it, ship the classifier as the bot.
That bot would be weak, and — worse — **you could not tell how weak**, because
nothing would be checking it.

So the dataset does **two honest jobs** instead:

1. It **trains the evaluator** — the leaf scoring function a search can call.
2. It **is a ground-truth test set** — the labels are perfect play, so verdicts
   can be graded against truth.

And **the search is what actually plays.**

---

## 3. The measurements — the heart of the talk

`scripts/validate.py` runs both engines at **fixed depth**, so any difference is
attributable to the *evaluator* rather than to one engine getting more nodes for
its money. Say that out loud; it is the control that makes the comparison mean
anything.

**A. Verdict agreement at ply 8** — 300 held-out positions, graded against the
solver:

| predictor | 3-class | decisive (draws excluded) |
|---|---|---|
| network alone, **no search** | **0.840** | **0.923** |
| search + hand-written heuristic | 0.670 | 0.734 |
| search + network | 0.710 | 0.782 |

**B. Head to head** — 14 games, alternating colours, forced distinct openings:

> search+heuristic **8 — 4** search+network (2 draws)

**C. Does the search earn its keep?**

> search+network **13 — 0** greedy network alone (1 draw)

**Then say what they mean together, because A and B look contradictory:**

- The network is **excellent on the distribution it was trained on** — it beats
  the hand-written heuristic badly there, **0.923 vs 0.734** decisive.
- It is **unreliable off that distribution**. A depth-7 search from a ply-8 board
  asks the network about **ply-15 leaves** — boards it has never seen — and it
  answers with enormous confidence anyway. Confidently wrong at the leaves is
  worse for a search than roughly right, because alpha-beta **propagates** leaf
  error. That is why it loses the head-to-head despite being the better
  standalone predictor.
- **Search is worth far more than either evaluator.** 13–0 is not a close call.

**So the shipped bot searches with the hand-written evaluator**, and the network
keeps the job it is genuinely good at: publishing a per-column opinion next to
the search's, where you can see it. Point at the line in `api.py` where that
decision is made — the reasoning is written there. **That decision is a result,
not a default.**

### The model itself

`(98 -> 128 -> 64 -> 3)`, **about 21k parameters**. He init, ReLU, numerically
stable softmax, cross-entropy, hand-written Adam with bias correction, early
stopping with best-weight restore.

| model | accuracy | draw recall |
|---|---|---|
| always predict "win" (65.8% of rows) | 0.6583 | 0.000 |
| logistic regression, same features | 0.7619 | 0.007 |
| **from-scratch MLP** | **0.8507** | **0.282** |

**Say explicitly: draw recall is the number that matters.** Draws are 9.6% of the
data, and the linear control essentially *never* predicts one — it buys its 76%
by ignoring the class entirely. Accuracy alone would have hidden that completely.
That is why both controls are in the table: a majority-class baseline and a
linear baseline on the *same features*, so the MLP has to earn its complexity
rather than just be reported.

The scalar the search consumes is **P(win) − P(loss)**. Draw contributes zero,
which is exactly right, and it is bounded by construction — so an opinion can
never stray into the proven-score range and outrank a real proof.

---

## 4. Live demo — the running order

Have it running before you start: `uv run python -m connect4.api`.

1. **Play one move.** Point at the **eval bar** beside the board — from the
   mover's point of view, `#n` for a proven mate, `=` for a proven draw, a number
   otherwise.
2. **The engine lines list.** One row per legal column, sorted best first. Click
   a row to play it. Note that **only the top line shows a principal variation** —
   inventing continuations for the others would be fabrication, so they show just
   their own move.
3. **Two brains per line.** The search's score, and next to it a signed,
   centre-anchored bar for the network's opinion of the same move. **Proven
   verdicts turn green or red; guesses stay blue.** That colour rule is the whole
   honesty story in one glance.
4. **The disagreement callout.** It names the column where the network most
   disagrees with a **proof**. Say why only proofs count: comparing the network
   against another heuristic would be two opinions, and neither is evidence about
   the other.
5. **Ghost pieces** — the principal variation drawn on the board, numbered in
   play order.
6. **Live search stats** — depth, nodes, kn/s, transposition-table hits, ms.
7. **Assist toggle** — outlines the column the search would play *for you*, so
   the engine works as a coach rather than only as an opponent.
8. **Difficulty.** Emphasise: it is the ***n*-th best true move**, never random
   noise. Even at 0 the bot takes a win that is on the board and refuses to walk
   into a mate in two — because a bot that plays well and then randomly throws a
   piece away reads as a **bug**, not as easy mode.
9. **Solver mode (difficulty 6).** A separate mode, not another notch, because
   what changes is the **budget**, not the move choice: six times the clock, and
   a **transposition table that persists between moves**. The second part matters
   more — consecutive searches in one game overlap enormously, so keeping the
   table makes each move a continuation of the last rather than a fresh start.
   Entries are keyed by *position*, not by search, so the reuse is sound.
   - **The honest caveat lands well:** it does **not** claim a solve from the
     empty board. Connect 4 is solved — the first player wins by move 41 — but
     proving that takes billions of nodes, and CPython is not doing that inside a
     web request. Measured over a full game at the solver's own settings,
     the first fully resolved position — every legal column proven — comes
     at **fourteen stones**, and from there on nearly every ply is exact, most
     in under a second once the persistent table holds the sub-positions.
     Earlier it is partial and erratic: 4/7 columns at seven stones, 0/7 at
     eight. The UI lights the `proven` badge per column, **only** where it
     genuinely did. Overclaiming a solve would be the one dishonest thing this
     project could ship.
10. **History tab.** Finished games with a win/loss/draw record, replayable.

---

## 5. Walking the code — file by file, with the line to point at

Go **bottom-up**. Each layer only makes sense once the one below it exists.

### `src/connect4/bitboard.py` — "why the board is an integer"

- **The pitch:** the obvious representation is a 6x7 list of lists, and then a
  win check means scanning rows, columns and both diagonals — dozens of
  comparisons *per node*. A search visits millions of nodes, so that cost
  dominates everything. Instead the board is packed into a single Python integer
  and a win is detected with **four shift-and-mask steps, no loops at all**.
- **Show the ASCII diagram in the module docstring.** 7 bits per column: 6
  playable cells plus one always-empty **sentinel** bit on top.
- **The sentinel is the trick — explain it, it is a great detail.** The column
  stride is 7, so a vertical run is 4 *adjacent* bits. Without the sentinel, the
  top cell of column 0 (bit 5) and the bottom of column 1 (bit 6) would be
  adjacent too, and a vertical win check would happily match four bits that
  **wrap from one column into the next**. The permanently empty sentinel sits
  between them and breaks every such wrap.
- **Two integers, not one board of colours:** `position` holds the stones of the
  player *whose turn it is*, `mask` holds every stone of either colour, and the
  opponent is `position ^ mask`. This makes the code **colour-agnostic** — the
  search never asks "am I red or yellow", it only reasons from the side to move.
  **That is what lets you write negamax instead of a mirrored minimax with two
  branches.**
- **`non_losing_moves` / `safe_moves`** — moves that do not hand the opponent an
  immediate win, computed with masks rather than by trying each move in turn.

### `src/connect4/engine.py` — "what actually plays"

- `heuristic_evaluator` — deliberately simple, **because its job is to be the
  baseline the neural evaluator gets measured against**. Two cheap signals:
  threat count, and centre control (a stone in column 3 belongs to 13 possible
  fours, one in column 0 belongs to 3). Clamped to +/-0.95.
- `WIN_SCORE = 10_000` — terminal scores sit far above anything an evaluator can
  return, so **a proof always outranks a promising-looking position**. The board
  holds 42 stones, so subtracting the ply count to prefer faster wins can never
  cross into heuristic range.
- **`EXACT / LOWER_BOUND / UPPER_BOUND`** — point at this. Alpha-beta does not
  always compute a node's true score; when a cutoff happens it only learns a
  *bound*. Storing **which of the three you have** is what makes transposition
  table reuse sound rather than subtly wrong. This is the easiest place in a
  search engine to be quietly incorrect.
- **`analyse()`** — the glass-box compromise. Each child is searched with a
  **full window** instead of the usual null-window re-search. That is slower, but
  it yields a real score for **every** column instead of just "worse than the
  best one" — and showing every column is the entire premise of the UI. Say that
  you knowingly paid performance for explainability.
- **Iterative deepening** — solve shallow, then deeper, reusing the table. Cheap
  insurance: when the clock runs out you still hold a complete, coherent answer
  from the last finished depth, never a half-updated one.
- **Move ordering** — centre-first, then the transposition table's best move.
  Ordering is what makes alpha-beta actually cut.
- **The 2.5x came from a profiler, not from cleverness** — this is the one to
  tell as a story. The profile of a single fixed-depth search said a third of
  the time was inside the threat-map function, called ~6 times per node for a
  board that never changed between calls. Three fixes: **remember the threat
  maps** on the position (every node asks two or three times: the move filter,
  the ordering, the leaf evaluator); **rank moves lazily**, because the table
  move usually cuts off on its own and ranking the other six columns costs a
  board and a threat map each; and **build each child once** instead of once
  for ranking and again for the search. **27k -> 69k nodes/second, one extra
  ply per second on nine of ten benchmark positions.**
  - **The claim that makes it safe:** every column's score at every depth from
    1 to 7 on those ten positions is **identical before and after**. Root
    children are searched with a full window, so those numbers are true values
    -- an ordering change cannot move one without it being a bug.
  - **What was rejected, with numbers** (worth volunteering): principal
    variation search cut the tree 5.6% and the clock not at all; an unrolled
    threat map landed inside the measurement noise. Also worth saying: the
    first micro-benchmarks were wrong, because whichever candidate ran last
    came out ~60% slower regardless of which one it was. Interleaving the
    candidates and taking the minimum fixed it.

### `src/connect4/dataset.py` — section 2 above.

### `src/connect4/model.py` — "the network, by hand"

- `forward` returns activations **and** pre-activations, because backprop needs
  both — that is why it is not a one-liner.
- `backward` and `apply_adam` are written out: Adam **with bias correction**, and
  weight decay handled as decoupled rather than folded into the gradient.
- **The claim to make, and it is checkable:** `tests/test_model.py`
  **gradient-checks every layer against central differences in float64**, with
  and without weight decay. That is the only honest proof a hand-written backward
  pass is correct, and it is the reason you can say you understand every line of
  the training code.
- `NeuralEvaluator` memoises on the position key, because the search revisits
  transpositions constantly and a forward pass costs far more than a dict lookup.
  The network costs **1.54x** the wall clock per search at equal depth; the cache
  absorbs most of it.

### `src/connect4/api.py` — "a game is its move list"

Two design decisions are stated at the top of the file — quote them:

- **A game is stored as its move list, not as a board.** Replaying seven columns
  is free, and it makes undo, rematch, sharing and debugging trivial: a game's
  entire state fits in a URL. Storing a bitboard instead would save nothing and
  cost you the history.
- **Analysis is a first-class part of every response, not a debug endpoint.**
  Every bot move comes back with both brains' full opinion of every column,
  because showing the disagreement **is** the product. The UI never has to ask a
  second time, and therefore **cannot** render an opinion about a different
  position than the one on screen — the analysis is attached to the position it
  describes.
- **A turn is two searches, and one of them happens while you are thinking.**
  Choosing the bot's move and describing the position it creates are both
  searches; the second one is the analysis panel. Caching the analysis per
  position removed a duplicate search (3.13s -> 2.05s a turn), and analysing
  your likely replies while the board waits for you removed most of what was
  left (**2.05s -> 1.13s**). Section 6 has the numbers and the caveats.
- `legal_moves` is **sorted** on the wire even though the engine produces it
  centre-first: that ordering is a *search optimisation* and has no business
  leaking into the wire format.
- Every knob is an **environment variable** — time limit, solver time limit,
  host and port, history path — so nothing needs editing to run it differently.
  `CONNECT4_HISTORY` has **three** states, not two: unset uses the default file,
  a path writes there, and **empty keeps games in memory only and never touches
  the disk** — which is what a shared machine or a throwaway container wants.

### `web/` — "no framework, on purpose"

The DOM is a **pure function of one state object**: every response replaces the
state, then one render function draws it. No incremental patching means no class
of bug where the board and the analysis disagree about which position they are
showing. The layout idiom is borrowed from **lichess's analysis board**, because
that page solves exactly this problem — showing a lot of engine output without
burying the game.

---

## 6. Engineering decisions worth defending (they will ask)

- **JSONL instead of SQLite for the archive.** A crash mid-append costs only the
  line being written and leaves every earlier game intact; a truncated final line
  is simply skipped on load; unknown fields from a newer version are ignored; and
  the file is readable without the program.
  - **State the durability guarantee precisely, because the honest version is
    narrower than "cannot lose a game":** a game is held in memory the moment it
    ends and written immediately after, so a failed write — full disk, unwritable
    path — leaves it visible and replayable for the rest of that process, and is
    **logged rather than swallowed**. It is durable **only once the write
    succeeds**. What the format buys is that a loss is confined to itself.
- **Concurrency.** FastAPI runs synchronous handlers in a **threadpool**, so two
  requests for the same game genuinely run at the same time. Turn validation and
  the move append have to be **one critical section** — otherwise both requests
  pass the turn check on the same position and both append, which plays two plies
  for one side. A double-click on a column is enough to do it. Fixed with one
  lock per game, so unrelated games never wait on each other. The test fires six
  simultaneous requests through a barrier and asserts **exactly one** 200.
- **Latency was measured, not tuned by feel.** A turn costs two searches --
  one to choose the bot's move, one to describe the position it creates, which
  is the analysis panel. Timed end to end against a running server: a search
  per request **3.13s**, memoising the analysis per position **2.05s** (the
  same position was being searched twice a turn), and analysing the human's
  likely replies while the board waits for them **1.13s**. In a browser, click
  to the bot's stone: 1.18-1.30s cold, 0.14-0.22s on a familiar opening.
  - **The caveats belong in the same breath as the numbers.** Someone who
    clicks instantly gains nothing and pays ~4% for background work that is
    discarded; a game in progress keeps a second core busy; `CONNECT4_WARM=0`
    turns it off. The cache is only sound because `analyse` clears its table
    each call, so an analysis is a pure function of the position -- and solver
    mode, which deliberately *keeps* its table, is excluded from warming for
    exactly that reason.
- **Bounded memory.** Games live in a dict with a capacity and oldest-first
  eviction. A database would be ceremony around a dict for a local app, but
  memory is still finite.

---

## 7. The two claims worth checking first (say this near the end)

> "If you only have time to check two things in this repo, check these."

**The bitboard is right** — `tests/test_bitboard.py` uses **differential
testing**: a slow, obviously correct nested-loop reference is asserted to agree
with the fast shift-and-mask version over thousands of random positions. A wrong
shift constant works for a hundred positions and then silently miscounts a
diagonal, and no amount of staring at it will tell you.

**The backward pass is right** — `tests/test_model.py` gradient-checks against
central differences, as above.

Those two are the load-bearing correctness claims. Everything else is built on
them.

---

## 8. Likely questions, with answers

**"Why not just use PyTorch?"**
Because the deliverable was understanding, not the last 2% of accuracy. Every
line of the forward pass, backprop and Adam is mine and gradient-checked. On a
21k-parameter MLP over 67k rows, a framework buys convenience and nothing else.

**"Why doesn't the neural net play, if it is more accurate?"**
It *is* more accurate — at ply 8, the only depth it has ever seen (0.923 vs
0.734 decisive). But a depth-7 search asks it about ply-15 leaves, and it is
confidently wrong there. Alpha-beta propagates leaf error, so a confidently wrong
evaluator is worse than a roughly right one. Head to head, the hand-written
heuristic beat it 8–4. I kept the measurement rather than the assumption.

**"Isn't Connect 4 already solved?"**
Yes — the first player wins by move 41. That is precisely why this dataset is
useful: the labels are perfect-play ground truth, so I can grade against truth
instead of against a proxy. And I explicitly do **not** claim a solve from the
empty board; the UI only lights `proven` where the search genuinely resolved the
line.

**"The Kaggle connect-4 dataset has 370,000 rows — why do you say 67,557?"**
Because those are two different files. The one on Kaggle under that name is the
"Connect-4 Game Dataset": 376,641 rows, one per finished game, cells left-to-right
top-to-bottom as 1/-1/0, labelled with who actually won a self-play game. I used
the UCI `connect-4` opening database — 67,557 positions at exactly 8 plies, labelled
with the **perfect-play** outcome by Tromp's solver. I need solver labels, because
grading against ground truth is the whole premise. See the table in section 2 for
the full comparison, and the cost of that choice.

**"How do you know the labels are right?"**
They are Tromp's solver output, and independently my own search agrees with them
on held-out positions at the rate reported in table A — which is also how I
measure the bot.

**"What would you do next?"**
Train on positions at *many* depths (self-play or solver-labelled), because the
single-depth distribution is the root cause of the network's failure — that is a
data problem, not a model problem. Then a compiled-language search for depth, and
self-play fine-tuning so the evaluator sees the boards the search actually
visits.

**"What went wrong, and what did you learn?"**
The honest one: I assumed the trained model would be the bot. The measurement
said otherwise, and the useful part of the project came from believing the
measurement. Also, the fixed-depth control mattered more than the comparison
itself — without it I would have been measuring the clock.

---

## 9. Closing line

> "The dataset didn't give me a bot. It gave me a **referee**, and a second
> opinion worth putting on screen. The bot is the search. Everything in that UI
> is either something the engine proved or something the network guessed, and the
> interface never lets you confuse the two."
