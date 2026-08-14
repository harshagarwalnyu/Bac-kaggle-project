/* Front end for the glass-box bot.
 *
 * No framework on purpose. The interesting part of this project is the search
 * and the network, and a build step would add a step between "clone" and
 * "play" without adding anything to either.
 *
 * One rule runs through the whole file: the DOM is a pure function of `state`.
 * Every handler mutates state and calls `render()`. Nothing patches the board
 * in place, so the board can never drift out of sync with the server's idea of
 * the position -- a class of bug that is miserable to chase and free to avoid.
 */

const COLUMNS = 7;
const ROWS = 6;

const state = {
  game: null,          // server's description of the position
  analysis: null,      // both brains, about the position now on screen
  busy: false,         // a request is in flight; input is locked
  message: "",
  hoverColumn: null,
};

const el = {
  board: document.getElementById("board"),
  status: document.getElementById("status"),
  columns: document.getElementById("columns"),
  stats: document.getElementById("stats"),
  pv: document.getElementById("pv-line"),
  disagreement: document.getElementById("disagreement"),
  newGame: document.getElementById("new-game"),
  undo: document.getElementById("undo"),
  botFirst: document.getElementById("bot-first"),
  skill: document.getElementById("skill"),
  skillValue: document.getElementById("skill-value"),
};

const SKILL_NAMES = [
  "0 — sixth best", "1 — fifth best", "2 — fourth best",
  "3 — third best", "4 — second best", "5 — full strength",
];

/* ----------------------------------------------------------------- network */

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    // The server sends a human-readable `detail` for every deliberate refusal
    // (full column, finished game). Surfacing it beats a generic failure.
    let detail = `HTTP ${response.status}`;
    try { detail = (await response.json()).detail ?? detail; } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return response.json();
}

const post = (path, body) =>
  api(path, { method: "POST", body: JSON.stringify(body ?? {}) });

/* ------------------------------------------------------------------- flow */

async function newGame() {
  const botFirst = el.botFirst.checked;
  await withBusy(async () => {
    const data = await post("/api/games", {
      skill: Number(el.skill.value),
      bot_first: botFirst,
    });
    absorb(data);
  });
  if (botFirst) await botMove();
}

async function playColumn(column) {
  if (!state.game || state.busy) return;
  if (state.game.status !== "playing") return;
  if (state.game.turn !== humanPlayer()) return;
  if (!state.game.legal_moves.includes(column)) return;

  await withBusy(async () => {
    absorb(await post(`/api/games/${state.game.id}/moves`, { column }));
  });

  if (state.game.status === "playing") await botMove();
}

async function botMove() {
  if (!state.game || state.game.status !== "playing") return;

  state.message = "Thinking…";
  render();

  await withBusy(async () => {
    const data = await post(`/api/games/${state.game.id}/bot-move`, {
      skill: Number(el.skill.value),
    });
    absorb(data);
  });
}

async function undo() {
  if (!state.game || state.busy) return;
  await withBusy(async () => {
    absorb(await post(`/api/games/${state.game.id}/undo`));
  });
}

/** Run `work` with input locked, and turn any failure into a visible message. */
async function withBusy(work) {
  state.busy = true;
  render();
  try {
    await work();
    state.message = "";
  } catch (error) {
    state.message = String(error.message ?? error);
  } finally {
    state.busy = false;
    render();
  }
}

function absorb(data) {
  state.game = data.game;
  state.analysis = data.analysis;
}

/** The bot owns one colour; the human owns the other. */
const humanPlayer = () => (state.game.bot_player === 1 ? 2 : 1);

/* ----------------------------------------------------------------- drawing */

function render() {
  if (!state.game) return;
  renderStatus();
  renderBoard();
  renderPanel();
  el.undo.disabled = state.busy || state.game.moves.length === 0;
  el.newGame.disabled = state.busy;
}

function renderStatus() {
  const game = state.game;
  let text, cls = "";

  if (state.message) {
    text = state.message;
    cls = state.message === "Thinking…" ? "thinking" : "";
  } else if (game.status === "won") {
    const humanWon = game.winner === humanPlayer();
    text = humanWon ? "You win." : "The bot wins.";
    cls = humanWon ? "win" : "lose";
  } else if (game.status === "draw") {
    text = "Draw — the board is full.";
  } else if (game.turn === humanPlayer()) {
    text = `Your move (ply ${game.ply + 1}).`;
  } else {
    text = "Bot to move.";
  }

  el.status.textContent = text;
  el.status.className = `status ${cls}`;
}

function renderBoard() {
  const game = state.game;
  const ghosts = principalVariationCells();
  const lastCell = lastMoveCell();

  el.board.replaceChildren();

  for (let row = 0; row < ROWS; row++) {
    for (let column = 0; column < COLUMNS; column++) {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.dataset.column = String(column);

      const value = game.grid[row][column];
      const playable = game.legal_moves.includes(column);
      if (!playable) cell.dataset.full = "1";
      if (playable && state.hoverColumn === column) cell.classList.add("hint");

      if (value !== 0) {
        const disc = document.createElement("div");
        disc.className = `disc p${value}`;
        if (lastCell && lastCell.row === row && lastCell.column === column) {
          disc.classList.add("last");
        }
        cell.appendChild(disc);
      } else {
        const ghost = ghosts.get(`${row},${column}`);
        if (ghost) {
          const mark = document.createElement("div");
          mark.className = `ghost p${ghost.player}`;
          mark.textContent = String(ghost.order);
          cell.appendChild(mark);
        }
      }

      el.board.appendChild(cell);
    }
  }
}

/** Where the last stone landed, in display coordinates. */
function lastMoveCell() {
  const game = state.game;
  if (game.last_move === null) return null;
  const column = game.last_move;
  // The topmost occupied cell of that column is the one just played.
  for (let row = 0; row < ROWS; row++) {
    if (game.grid[row][column] !== 0) return { row, column };
  }
  return null;
}

/**
 * Walk the principal variation, working out where each stone would land.
 *
 * The server sends the variation as bare columns; the landing row depends on
 * the stones already placed *and* on the earlier moves of the variation
 * itself, so the heights have to be carried forward as we go.
 */
function principalVariationCells() {
  const cells = new Map();
  const pv = state.analysis?.principal_variation ?? [];
  if (!pv.length) return cells;

  const heights = new Array(COLUMNS).fill(0);
  for (let column = 0; column < COLUMNS; column++) {
    for (let row = 0; row < ROWS; row++) {
      if (state.game.grid[row][column] !== 0) heights[column]++;
    }
  }

  let player = state.game.turn;
  pv.forEach((column, index) => {
    const height = heights[column];
    if (height >= ROWS) return;
    cells.set(`${ROWS - 1 - height},${column}`, { order: index + 1, player });
    heights[column]++;
    player = player === 1 ? 2 : 1;
  });
  return cells;
}

function renderPanel() {
  const analysis = state.analysis;
  el.columns.replaceChildren();

  if (!analysis) {
    el.stats.textContent = "";
    el.pv.textContent = "—";
    el.disagreement.textContent = "Game over — nothing left to search.";
    return;
  }

  const stats = analysis.stats;
  el.stats.innerHTML =
    `depth <b>${stats.depth}</b> · nodes <b>${stats.nodes.toLocaleString()}</b> · ` +
    `tt hits <b>${stats.table_hits.toLocaleString()}</b> · <b>${stats.elapsed_ms}</b> ms` +
    (stats.exact ? ' · <b style="color:var(--proof)">solved</b>' : "");

  // Columns arrive best-first from the engine. Showing them in board order
  // instead lets you compare the panel against the board without hunting.
  const byColumn = [...analysis.columns].sort((a, b) => a.column - b.column);
  for (const entry of byColumn) el.columns.appendChild(columnRow(entry, analysis.best_move));

  el.pv.textContent = analysis.principal_variation.length
    ? analysis.principal_variation.map((c, i) => `${i + 1}. col ${c}`).join("   ")
    : "—";

  el.disagreement.innerHTML = disagreementNote(analysis);
}

function columnRow(entry, bestMove) {
  const row = document.createElement("div");
  row.className = "row";
  if (entry.column === bestMove) row.classList.add("best");

  const engineProof = entry.engine.exact;
  const losing = engineProof && entry.engine.score < 0;
  if (losing) row.classList.add("dead");

  const id = document.createElement("div");
  id.className = "col-id";
  id.textContent = entry.column;

  const bars = document.createElement("div");
  bars.className = "bars";
  bars.appendChild(bar("engine", engineWidth(entry.engine),
    engineProof ? (losing ? "proof-loss" : "proof") : ""));
  bars.appendChild(bar("network", entry.network.preference, ""));

  const verdict = document.createElement("div");
  verdict.className = "verdict";
  const engineText = document.createElement("span");
  engineText.className = `engine-v ${engineProof ? (losing ? "proof-loss" : "proof") : ""}`;
  engineText.textContent = entry.engine.label;
  const networkText = document.createElement("span");
  networkText.className = "network-v";
  networkText.textContent = formatSigned(entry.network.preference);
  verdict.append(engineText, networkText);

  row.append(id, bars, verdict);
  row.title = networkTooltip(entry);
  return row;
}

/**
 * Map an engine score onto [-1, 1] for drawing.
 *
 * A proof is drawn full-width regardless of how many plies it takes, because
 * a forced win is a forced win — scaling it by distance would suggest a
 * "stronger" and "weaker" win, which is not a thing. Heuristic scores are
 * already bounded in [-1, 1] by construction, so they pass through.
 */
function engineWidth(engine) {
  if (engine.exact) return engine.score > 0 ? 1 : engine.score < 0 ? -1 : 0;
  return Math.max(-1, Math.min(1, engine.score));
}

function bar(kind, value, extraClass) {
  const track = document.createElement("div");
  track.className = "bar";
  const fill = document.createElement("div");
  fill.className = `fill ${kind} ${extraClass}`.trim();

  const magnitude = Math.abs(value) * 50;   // half the track is 100%
  fill.style.width = `${magnitude}%`;
  fill.style.left = value >= 0 ? "50%" : `${50 - magnitude}%`;

  track.appendChild(fill);
  return track;
}

const formatSigned = (value) => (value >= 0 ? "+" : "") + value.toFixed(2);

function networkTooltip(entry) {
  const p = entry.network.probabilities;
  return `network: win ${(p.win * 100).toFixed(1)}%  ` +
         `draw ${(p.draw * 100).toFixed(1)}%  loss ${(p.loss * 100).toFixed(1)}%`;
}

/**
 * The headline of the whole project: name the column where the learned model
 * most disagrees with the search.
 *
 * Only *proven* engine verdicts count here. Comparing the network against
 * another heuristic guess would be comparing two opinions, and neither would
 * be evidence about the other. A proof is the only thing that settles it.
 */
function disagreementNote(analysis) {
  const proven = analysis.columns.filter((c) => c.engine.exact);
  if (!proven.length) {
    return "No proven verdicts yet — the search is still in heuristic territory, " +
           "so both panels are opinions.";
  }

  let worst = null;
  for (const entry of proven) {
    const truth = Math.sign(entry.engine.score);
    const guess = entry.network.preference;
    const gap = Math.abs(truth - Math.max(-1, Math.min(1, guess)));
    if (!worst || gap > worst.gap) worst = { entry, gap, truth };
  }

  if (worst.gap < 0.5) {
    return `The network agrees with every proof on the board (worst gap ` +
           `${worst.gap.toFixed(2)}). It was trained on 8-stone positions only, ` +
           `so this will not last.`;
  }

  const { entry, truth } = worst;
  const truthWord = truth > 0 ? "a proven win" : truth < 0 ? "a proven loss" : "a proven draw";
  return `Column <b>${entry.column}</b> is ${truthWord} (${entry.engine.label}), but the ` +
         `network scores it ${formatSigned(entry.network.preference)}. ` +
         `That is the gap between fitting 67k labelled positions and actually solving one.`;
}

/* ---------------------------------------------------------------- wiring */

el.board.addEventListener("click", (event) => {
  const cell = event.target.closest(".cell");
  if (cell) playColumn(Number(cell.dataset.column));
});

el.board.addEventListener("mousemove", (event) => {
  const cell = event.target.closest(".cell");
  const column = cell ? Number(cell.dataset.column) : null;
  if (column !== state.hoverColumn) {
    state.hoverColumn = column;
    renderBoard();
  }
});

el.board.addEventListener("mouseleave", () => {
  state.hoverColumn = null;
  renderBoard();
});

// Keyboard play: 1-7 drop, u undo, n new game. Faster than the mouse once you
// know the board, and it makes the app usable without one.
document.addEventListener("keydown", (event) => {
  if (event.key >= "1" && event.key <= "7") playColumn(Number(event.key) - 1);
  else if (event.key === "u") undo();
  else if (event.key === "n") newGame();
});

el.newGame.addEventListener("click", newGame);
el.undo.addEventListener("click", undo);
el.skill.addEventListener("input", () => {
  el.skillValue.textContent = SKILL_NAMES[Number(el.skill.value)];
});

el.skillValue.textContent = SKILL_NAMES[Number(el.skill.value)];
newGame();
