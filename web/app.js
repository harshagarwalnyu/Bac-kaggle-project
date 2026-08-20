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
const SOLVER_SKILL = 6;

const state = {
  game: null,          // server's description of the position
  analysis: null,      // both brains, about the position now on screen
  busy: false,         // a request is in flight; input is locked
  message: "",
  tab: "analysis",
  history: { summary: null, games: [] },
};

const el = {
  board: document.getElementById("board"),
  status: document.getElementById("status"),
  evalbar: document.getElementById("evalbar"),
  evalbarFill: document.getElementById("evalbar-fill"),
  evalbarText: document.getElementById("evalbar-text"),
  stats: document.getElementById("stats"),
  lines: document.getElementById("lines"),
  disagreement: document.getElementById("disagreement"),
  movelist: document.getElementById("movelist"),
  movelistEmpty: document.getElementById("movelist-empty"),
  history: document.getElementById("history"),
  record: document.getElementById("record"),
  clearHistory: document.getElementById("clear-history"),
  newGame: document.getElementById("new-game"),
  undo: document.getElementById("undo"),
  playBest: document.getElementById("play-best"),
  botFirst: document.getElementById("bot-first"),
  assist: document.getElementById("assist"),
  skill: document.getElementById("skill"),
  skillValue: document.getElementById("skill-value"),
  solverBanner: document.getElementById("solver-banner"),
  tabs: document.getElementById("tabs"),
};

const SKILL_NAMES = [
  "0 — sixth best", "1 — fifth best", "2 — fourth best",
  "3 — third best", "4 — second best", "5 — full strength",
  "6 — SOLVER (not the ML model)",
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
  else await loadHistory();
}

/** The assist toggle's button: play whatever the search says is best for me. */
function playRecommended() {
  const column = recommendedColumn();
  if (column !== null) playColumn(column);
}

async function botMove() {
  if (!state.game || state.game.status !== "playing") return;

  state.message = solverMode() ? "Solving…" : "Thinking…";
  render();

  await withBusy(async () => {
    const data = await post(`/api/games/${state.game.id}/bot-move`, {
      skill: Number(el.skill.value),
    });
    absorb(data);
  });
  if (state.game.status !== "playing") await loadHistory();
}

async function undo() {
  if (!state.game || state.busy) return;
  await withBusy(async () => {
    absorb(await post(`/api/games/${state.game.id}/undo`));
  });
}

async function loadHistory() {
  try {
    state.history = await api("/api/history?limit=25");
  } catch {
    // History is a nicety. Failing to load it must not break the game.
    state.history = { summary: null, games: [] };
  }
  renderHistory();
}

async function clearHistory() {
  await api("/api/history", { method: "DELETE" });
  await loadHistory();
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

/**
 * The class that paints a stone: yellow for the human, red for the bot.
 *
 * Choosing it from the player *number* looks right only while the human opens.
 * Let the bot go first and it becomes player 1, so the board hands the bot the
 * yellow stones and the human the red ones -- while the legend underneath still
 * says "yellow outlines are yours". Every other number on this page is stated
 * from the human's side; the colours are a statement about roles too, so they
 * have to be chosen from the role rather than from the seat.
 */
const sideClass = (player) => (player === humanPlayer() ? "you" : "bot");
const solverMode = () => Number(el.skill.value) >= SOLVER_SKILL;
const myTurn = () =>
  state.game && state.game.status === "playing" && state.game.turn === humanPlayer();

/**
 * The column the search would play for me, or null.
 *
 * `analysis` always describes the position currently on the board, from the
 * point of view of whoever is to move. When that is me, the engine's best move
 * *is* my best move -- no transformation needed, and none should be invented.
 */
function recommendedColumn() {
  if (!myTurn() || !state.analysis) return null;
  const best = state.analysis.best_move;
  return state.game.legal_moves.includes(best) ? best : null;
}

/* ----------------------------------------------------------------- drawing */

function render() {
  if (!state.game) return;
  renderStatus();
  renderEvalBar();
  renderBoard();
  renderLines();
  renderMoveList();

  const assistColumn = el.assist.checked ? recommendedColumn() : null;
  el.playBest.hidden = assistColumn === null;
  el.playBest.disabled = state.busy;
  el.playBest.textContent =
    assistColumn === null ? "Play best" : `Play best — column ${assistColumn}`;

  // Undo takes back the human's turn, so it needs one to exist: a bot-first
  // game showing only the bot's opening has nothing to give back.
  const humanPlies = state.game.moves.filter(
    (_, i) => 1 + (i % 2) === humanPlayer(),
  ).length;
  el.undo.disabled = state.busy || humanPlies === 0;
  el.newGame.disabled = state.busy;
  el.solverBanner.hidden = !solverMode();
  el.skill.classList.toggle("solver", solverMode());
}

function renderStatus() {
  const game = state.game;
  let text, cls = "";

  if (state.message) {
    text = state.message;
    cls = /…$/.test(state.message) ? "thinking" : "";
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

/**
 * The headline number, always stated from the human's side.
 *
 * `analysis` speaks for whoever is to move, so when the bot is to move the
 * sign has to flip. Getting that backwards would make the bar say the opposite
 * of the truth exactly half the time, which is worse than having no bar.
 */
function renderEvalBar() {
  const best = state.analysis?.columns
    ? [...state.analysis.columns].sort((a, b) => b.engine.score - a.engine.score)[0]
    : null;

  if (!best) {
    el.evalbar.className = "evalbar";
    el.evalbarFill.style.height = "50%";
    el.evalbarText.textContent = "—";
    return;
  }

  const mine = state.game.turn === humanPlayer() ? 1 : -1;
  const signed = engineWidth(best.engine) * mine;

  el.evalbar.className =
    "evalbar" + (best.engine.exact ? (signed > 0 ? " proof" : signed < 0 ? " proof-loss" : "") : "");
  el.evalbarFill.style.height = `${(signed + 1) * 50}%`;

  if (best.engine.exact && best.engine.mate_in) {
    el.evalbarText.textContent = `#${best.engine.mate_in}`;
  } else if (best.engine.exact) {
    el.evalbarText.textContent = "=";
  } else {
    el.evalbarText.textContent = Math.abs(signed).toFixed(1);
  }
}

function renderBoard() {
  const game = state.game;
  const ghosts = principalVariationCells();
  const lastCell = lastMoveCell();
  const assistColumn = el.assist.checked ? recommendedColumn() : null;
  const heights = columnHeights();

  const board = document.createDocumentFragment();

  for (let column = 0; column < COLUMNS; column++) {
    const stack = document.createElement("div");
    stack.className = "column";
    stack.dataset.column = String(column);
    // Two different reasons a column cannot be clicked, kept distinct so the
    // stylesheet can say so: the column is full, or the game is over.
    if (!game.legal_moves.includes(column)) stack.classList.add("full");
    if (game.status !== "playing") stack.classList.add("over");
    if (column === assistColumn) stack.classList.add("recommended");

    const landingRow = ROWS - 1 - heights[column];

    for (let row = 0; row < ROWS; row++) {
      const cell = document.createElement("div");
      cell.className = "cell";
      if (column === assistColumn && row === landingRow) cell.classList.add("landing");

      const value = game.grid[row][column];
      if (value !== 0) {
        const disc = document.createElement("div");
        disc.className = `disc ${sideClass(value)}`;
        if (lastCell && lastCell.row === row && lastCell.column === column) {
          disc.classList.add("last");
        }
        cell.appendChild(disc);
      } else {
        const ghost = ghosts.get(`${row},${column}`);
        if (ghost) {
          const mark = document.createElement("div");
          mark.className = `ghost ${sideClass(ghost.player)}`;
          mark.textContent = String(ghost.order);
          cell.appendChild(mark);
        }
      }

      stack.appendChild(cell);
    }
    board.appendChild(stack);
  }

  // One swap rather than 42 appends, so the browser lays the board out once.
  el.board.replaceChildren(board);
}

/** How many stones are already in each column. */
function columnHeights() {
  const heights = new Array(COLUMNS).fill(0);
  for (let column = 0; column < COLUMNS; column++) {
    for (let row = 0; row < ROWS; row++) {
      if (state.game.grid[row][column] !== 0) heights[column]++;
    }
  }
  return heights;
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

  const heights = columnHeights();
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

/* ------------------------------------------------------------ engine lines */

function renderLines() {
  const analysis = state.analysis;
  el.lines.replaceChildren();

  if (!analysis) {
    el.stats.textContent = "";
    el.disagreement.textContent = "Game over — nothing left to search.";
    return;
  }

  const stats = analysis.stats;
  const nps = stats.elapsed_ms > 0
    ? Math.round(stats.nodes / (stats.elapsed_ms / 1000))
    : 0;
  el.stats.innerHTML =
    `depth <b>${stats.depth}</b> · <b>${stats.nodes.toLocaleString()}</b> nodes · ` +
    `<b>${(nps / 1000).toFixed(0)}</b> kn/s · tt <b>${stats.table_hits.toLocaleString()}</b> · ` +
    `<b>${stats.elapsed_ms}</b> ms` +
    (stats.exact ? ' · <span class="pill proof">proven</span>' : "");

  // Best first, like an engine-lines list: the ordering *is* the information.
  // (The board itself is the place to look things up by column.)
  const ranked = [...analysis.columns].sort((a, b) => b.engine.score - a.engine.score);
  ranked.forEach((entry, index) => {
    el.lines.appendChild(lineRow(entry, index === 0, analysis));
  });

  el.disagreement.innerHTML = disagreementNote(analysis);
}

function lineRow(entry, isBest, analysis) {
  const row = document.createElement("div");
  row.className = "line";
  row.dataset.column = String(entry.column);
  if (isBest) row.classList.add("best");

  const proof = entry.engine.exact;
  const losing = proof && entry.engine.score < 0;
  if (losing) row.classList.add("dead");

  const evalBox = document.createElement("div");
  evalBox.className = `line-eval ${proof ? (losing ? "proof-loss" : "proof") : ""}`.trim();
  evalBox.textContent = shortEval(entry.engine);

  const main = document.createElement("div");
  main.className = "line-main";

  // Only the top line has a stored variation; the rest show their own move as
  // the head of a line we did not search deeply. Inventing a continuation for
  // them would be fabrication, so they simply show one move.
  const pv = document.createElement("div");
  pv.className = "line-pv";
  const line = isBest && analysis.principal_variation.length
    ? analysis.principal_variation
    : [entry.column];
  line.forEach((column, index) => {
    const span = document.createElement("span");
    span.className = index === 0 ? "drop head" : "drop";
    span.textContent = `${index + 1}.c${column} `;
    pv.appendChild(span);
  });

  const net = document.createElement("div");
  net.className = "line-net";
  const label = document.createElement("span");
  label.textContent = "net";
  const track = document.createElement("div");
  track.className = "netbar";
  const fill = document.createElement("div");
  fill.className = "netbar-fill";
  const magnitude = Math.abs(entry.network.preference) * 50;
  fill.style.width = `${magnitude}%`;
  fill.style.left = entry.network.preference >= 0 ? "50%" : `${50 - magnitude}%`;
  track.appendChild(fill);
  const value = document.createElement("span");
  value.textContent = formatSigned(entry.network.preference);
  net.append(label, track, value);

  main.append(pv, net);
  row.append(evalBox, main);
  row.title = networkTooltip(entry);
  return row;
}

/** Engine verdict, short enough for a 60px box. `#n` is a proven mate in n. */
function shortEval(engine) {
  if (!engine.exact) return formatSigned(engine.score);
  if (!engine.mate_in || engine.score === 0) return "=";
  return `${engine.score > 0 ? "#" : "-#"}${engine.mate_in}`;
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

/* -------------------------------------------------------------- move list */

function renderMoveList() {
  const moves = state.game.moves;
  el.movelistEmpty.hidden = moves.length > 0;
  el.movelist.replaceChildren();

  // Paired per turn, the way a chess score sheet reads: one row is "player 1
  // did this, player 2 answered that".
  const rows = document.createDocumentFragment();
  for (let i = 0; i < moves.length; i += 2) {
    const tr = document.createElement("tr");
    if (i + 2 >= moves.length) tr.className = "current";
    tr.append(
      cellText("num", `${i / 2 + 1}.`),
      cellText("m1", `col ${moves[i]}`),
      cellText("m2", moves[i + 1] === undefined ? "" : `col ${moves[i + 1]}`),
    );
    rows.appendChild(tr);
  }
  el.movelist.appendChild(rows);
}

function cellText(className, text) {
  const td = document.createElement("td");
  td.className = className;
  td.textContent = text;
  return td;
}

/* ---------------------------------------------------------------- history */

function renderHistory() {
  const { summary, games } = state.history;

  el.record.innerHTML = summary && summary.games
    ? `<b>${summary.games}</b> games · <span class="w">${summary.human_wins}W</span> ` +
      `<span class="l">${summary.bot_wins}L</span> ${summary.draws}D`
    : "No games finished yet.";

  el.history.replaceChildren();
  const list = document.createDocumentFragment();
  for (const game of games) {
    const row = document.createElement("div");
    const kind = game.winner === 0 ? "drew" : game.outcome === "you won" ? "won" : "lost";
    row.className = `hgame ${kind}`;

    const outcome = document.createElement("div");
    outcome.className = "outcome";
    outcome.textContent = game.outcome;

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${game.plies} plies · ${game.duration_s}s · skill ${game.skill}`;
    if (game.solver) {
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = "SOLVER";
      meta.appendChild(badge);
    }

    const when = document.createElement("div");
    when.className = "when";
    when.textContent = relativeTime(game.ended);

    // The move list is the record's whole point; hovering shows it.
    row.title = `moves: ${game.moves.join(" ")}`;
    row.append(outcome, meta, when);
    list.appendChild(row);
  }
  el.history.appendChild(list);
}

function relativeTime(epochSeconds) {
  const seconds = Math.max(0, Date.now() / 1000 - epochSeconds);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

/* ---------------------------------------------------------------- wiring */

// Column highlighting is `.column:hover` in the stylesheet. There is no hover
// handler here on purpose: re-rendering on mousemove rebuilt every disc and
// restarted its animation, which flickered the whole board.
el.board.addEventListener("click", (event) => {
  const stack = event.target.closest(".column");
  if (stack) playColumn(Number(stack.dataset.column));
});

// Clicking an engine line plays that column. It is the fastest way to explore
// "what if I took the second-best move", and it costs one line of code.
el.lines.addEventListener("click", (event) => {
  const line = event.target.closest(".line");
  if (line) playColumn(Number(line.dataset.column));
});

el.tabs.addEventListener("click", (event) => {
  const tab = event.target.closest(".tab");
  if (!tab) return;
  state.tab = tab.dataset.tab;
  for (const button of el.tabs.querySelectorAll(".tab")) {
    button.classList.toggle("is-active", button.dataset.tab === state.tab);
  }
  for (const body of document.querySelectorAll(".tab-body")) {
    body.hidden = body.dataset.body !== state.tab;
  }
  if (state.tab === "history") loadHistory();
});

// Keyboard play: 1-7 drop, u undo, n new game, Enter takes the recommendation.
// Faster than the mouse once you know the board, and it makes the app usable
// without one.
document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, button")) return;
  if (event.key >= "1" && event.key <= "7") playColumn(Number(event.key) - 1);
  else if (event.key === "u") undo();
  else if (event.key === "n") newGame();
  else if (event.key === "Enter" && el.assist.checked) playRecommended();
});

el.newGame.addEventListener("click", newGame);
el.undo.addEventListener("click", undo);
el.playBest.addEventListener("click", playRecommended);
el.assist.addEventListener("change", render);
el.clearHistory.addEventListener("click", clearHistory);
el.skill.addEventListener("input", () => {
  el.skillValue.textContent = SKILL_NAMES[Number(el.skill.value)];
  if (state.game) render();
});

el.skillValue.textContent = SKILL_NAMES[Number(el.skill.value)];
newGame();
loadHistory();
