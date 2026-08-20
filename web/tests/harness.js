/* A DOM and a server, small enough to read in one sitting.
 *
 * `web/app.js` is a plain script: no modules, no build step, and that is a
 * deliberate property of this project rather than an accident to work around.
 * So the harness does what a browser does -- puts a document and a `fetch` in
 * front of the file and runs it -- instead of asking the file to become
 * importable for the tests' convenience.
 *
 * Two rules keep this honest:
 *
 *   1. The element ids come out of `web/index.html`, not out of a list kept
 *      here. Delete an id the script needs and the tests fail, which is the
 *      whole point of having them.
 *   2. The fake server answers with the shapes `src/connect4/api.py` really
 *      sends -- including the two that are easy to forget, `turn: 0` and an
 *      empty `legal_moves` once a game is over.
 */

const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const WEB = path.join(__dirname, "..");
const COLUMNS = 7;
const ROWS = 6;
const SOLVER_SKILL = 6;

/* -------------------------------------------------------------------- DOM */

/** Selector support is deliberately thin: tags, ids, stacked classes and comma
 *  lists, which is everything `app.js` and its tests ask for. Anything richer
 *  would be a second CSS engine to keep correct. */
function matchesSelector(node, selector) {
  return selector.split(",").some((part) => {
    const one = part.trim();
    if (!one) return false;
    const [head, ...classes] = one.split(".");
    if (head.startsWith("#")) {
      if (node.id !== head.slice(1)) return false;
    } else if (head && node.tagName !== head.toUpperCase()) {
      return false;
    }
    return classes.every((name) => node.classList.contains(name));
  });
}

class Node {
  constructor(tagName = "div") {
    this.tagName = tagName.toUpperCase();
    this.id = "";
    this.parentNode = null;
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.listeners = new Map();
    this.title = "";
    this.hidden = false;
    this.disabled = false;
    this.checked = false;
    this.value = "";
    this._classes = [];
    this._text = "";
    this._html = "";
  }

  get className() {
    return this._classes.join(" ");
  }

  set className(value) {
    this._classes = String(value).split(/\s+/).filter(Boolean);
  }

  get classList() {
    const classes = this._classes;
    return {
      add: (...names) => names.forEach((n) => classes.includes(n) || classes.push(n)),
      remove: (...names) => names.forEach((n) => {
        const at = classes.indexOf(n);
        if (at >= 0) classes.splice(at, 1);
      }),
      contains: (name) => classes.includes(name),
      toggle: (name, force) => {
        const on = force === undefined ? !classes.includes(name) : Boolean(force);
        if (on) this.classList.add(name);
        else this.classList.remove(name);
        return on;
      },
    };
  }

  get textContent() {
    return this._text + this.children.map((c) => c.textContent).join("");
  }

  set textContent(value) {
    this.children = [];
    this._text = value === null || value === undefined ? "" : String(value);
  }

  /** No parser here on purpose. Tests that need structure build it with
   *  `createElement`; tests that need the words read them back as text. */
  get innerHTML() {
    return this._html;
  }

  set innerHTML(value) {
    this.children = [];
    this._html = String(value);
    this._text = String(value).replace(/<[^>]*>/g, "");
  }

  appendChild(node) {
    if (node.isFragment) {
      node.children.splice(0).forEach((child) => this.appendChild(child));
      return node;
    }
    if (node.parentNode) node.parentNode.removeChild(node);
    node.parentNode = this;
    this.children.push(node);
    return node;
  }

  append(...nodes) {
    nodes.forEach((node) => this.appendChild(node));
  }

  removeChild(node) {
    const at = this.children.indexOf(node);
    if (at >= 0) this.children.splice(at, 1);
    node.parentNode = null;
    return node;
  }

  replaceChildren(...nodes) {
    this.children.splice(0).forEach((child) => (child.parentNode = null));
    this._text = "";
    this._html = "";
    nodes.forEach((node) => this.appendChild(node));
  }

  querySelectorAll(selector) {
    const found = [];
    const walk = (node) => {
      for (const child of node.children) {
        if (matchesSelector(child, selector)) found.push(child);
        walk(child);
      }
    };
    walk(this);
    return found;
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] ?? null;
  }

  matches(selector) {
    return matchesSelector(this, selector);
  }

  closest(selector) {
    for (let node = this; node; node = node.parentNode) {
      if (node.matches && node.matches(selector)) return node;
    }
    return null;
  }

  addEventListener(type, handler) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(handler);
  }

  /** Fire the handlers `app.js` registered, with `this` node as the target. */
  dispatch(type, event = {}) {
    const handlers = this.listeners.get(type) ?? [];
    const full = { target: this, ...event };
    return Promise.all(handlers.map((handler) => handler(full)));
  }
}

class Fragment extends Node {
  constructor() {
    super("#fragment");
    this.isFragment = true;
  }
}

/** Every id the page declares, so a missing one surfaces as a test failure
 *  rather than as a `null` the script happens not to touch that run. */
function idsInMarkup() {
  const html = fs.readFileSync(path.join(WEB, "index.html"), "utf8");
  return [...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]);
}

function makeDocument(overrides = {}) {
  const byId = new Map();
  for (const id of idsInMarkup()) {
    const node = new Node("div");
    node.id = id;
    byId.set(id, node);
  }

  // The controls carry the defaults the markup gives them; the script reads
  // these before the first render, so wrong defaults would quietly change
  // which skill every test runs at.
  const defaults = { skill: "5", assist: false, "bot-first": false, ...overrides };
  for (const [id, value] of Object.entries(defaults)) {
    const node = byId.get(id);
    if (!node) continue;
    if (typeof value === "boolean") node.checked = value;
    else node.value = value;
  }

  const root = new Node("body");
  for (const node of byId.values()) root.appendChild(node);

  const document = {
    body: root,
    getElementById: (id) => byId.get(id) ?? null,
    createElement: (tag) => new Node(tag),
    createDocumentFragment: () => new Fragment(),
    querySelectorAll: (selector) => root.querySelectorAll(selector),
    querySelector: (selector) => root.querySelector(selector),
    addEventListener: (type, handler) => root.addEventListener(type, handler),
    dispatch: (type, event) => root.dispatch(type, event),
    byId,
  };
  return document;
}

/* ------------------------------------------------------------------ rules */

const emptyGrid = () =>
  Array.from({ length: ROWS }, () => new Array(COLUMNS).fill(0));

function gridOf(moves) {
  const grid = emptyGrid();
  const heights = new Array(COLUMNS).fill(0);
  moves.forEach((column, index) => {
    const row = ROWS - 1 - heights[column];
    grid[row][column] = 1 + (index % 2);
    heights[column]++;
  });
  return { grid, heights };
}

function winnerOf(grid) {
  const at = (row, column) =>
    row >= 0 && row < ROWS && column >= 0 && column < COLUMNS ? grid[row][column] : 0;
  const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];
  for (let row = 0; row < ROWS; row++) {
    for (let column = 0; column < COLUMNS; column++) {
      const player = at(row, column);
      if (!player) continue;
      for (const [dr, dc] of directions) {
        let run = 1;
        while (run < 4 && at(row + dr * run, column + dc * run) === player) run++;
        if (run === 4) return player;
      }
    }
  }
  return 0;
}

/* ----------------------------------------------------------------- server */

/** The endpoints `app.js` calls, answering in the shapes `api.py` sends.
 *  Deterministic on purpose: the "bot" always takes the lowest legal column,
 *  so a test can say what the board looks like two plies from now. */
class FakeServer {
  constructor() {
    this.games = new Map();
    this.history = { summary: null, games: [] };
    this.requests = [];
    this.nextId = 1;
    this.fail = null; // set to { path, status, detail } to make one call fail
  }

  describe(game) {
    const { grid, heights } = gridOf(game.moves);
    const winner = winnerOf(grid);
    const full = game.moves.length === ROWS * COLUMNS;
    const status = winner ? "won" : full ? "draw" : "playing";
    const legal = [];
    for (let column = 0; column < COLUMNS; column++) {
      if (heights[column] < ROWS) legal.push(column);
    }
    return {
      id: game.id,
      moves: [...game.moves],
      grid,
      ply: game.moves.length,
      // Both of these go flat once the game is over. The front end has to
      // cope with that, so the harness must not soften it.
      turn: status !== "playing" ? 0 : 1 + (game.moves.length % 2),
      legal_moves: status !== "playing" ? [] : legal,
      status,
      winner: status === "won" ? winner : 0,
      bot_player: game.botPlayer,
      skill: game.skill,
      solver: game.skill >= SOLVER_SKILL,
      last_move: game.moves.length ? game.moves[game.moves.length - 1] : null,
    };
  }

  analysis(described) {
    if (described.status !== "playing") return null;
    const legal = described.legal_moves;
    const columns = legal.map((column, index) => ({
      column,
      engine: {
        score: Number((0.5 - index * 0.2).toFixed(4)),
        label: index === 0 ? "best" : "playable",
        exact: index === 0,
        mate_in: index === 0 ? 5 : null,
      },
      network: {
        preference: Number((0.4 - index * 0.3).toFixed(4)),
        probabilities: { win: 0.5, draw: 0.3, loss: 0.2 },
      },
    }));
    return {
      best_move: legal[0],
      columns,
      principal_variation: legal.slice(0, 3),
      stats: { nodes: 1234, depth: 9, table_hits: 56, elapsed_ms: 42.5, exact: true, aborted: false },
    };
  }

  respond(game) {
    const described = this.describe(game);
    return { game: described, analysis: this.analysis(described) };
  }

  botColumn(game) {
    const described = this.describe(game);
    return described.legal_moves[0] ?? null;
  }

  handle(url, options = {}) {
    const method = options.method ?? "GET";
    const body = options.body ? JSON.parse(options.body) : {};
    const route = url.split("?")[0];
    this.requests.push({ method, url, body });

    if (this.fail && this.fail.path === route) {
      const failure = this.fail;
      this.fail = null;
      throw Object.assign(new Error(failure.detail ?? `HTTP ${failure.status}`), {
        status: failure.status,
        detail: failure.detail,
        notJson: failure.notJson,
      });
    }

    if (route === "/api/games" && method === "POST") {
      const game = {
        id: `g${this.nextId++}`,
        moves: [],
        skill: body.skill ?? 5,
        botPlayer: body.bot_first ? 1 : 2,
      };
      this.games.set(game.id, game);
      return this.respond(game);
    }

    if (route === "/api/history" && method === "GET") return this.history;
    if (route === "/api/history" && method === "DELETE") {
      this.history = { summary: null, games: [] };
      return this.history;
    }

    const match = /^\/api\/games\/([^/]+)\/(moves|bot-move|undo)$/.exec(route);
    if (match && method === "POST") {
      const game = this.games.get(match[1]);
      if (!game) throw Object.assign(new Error("No such game."), { status: 404, detail: "No such game." });
      const described = this.describe(game);

      if (match[2] === "moves") {
        if (described.status !== "playing") {
          throw Object.assign(new Error("That game is over."), { status: 409, detail: "That game is over." });
        }
        if (!described.legal_moves.includes(body.column)) {
          throw Object.assign(new Error("That column is full."), { status: 409, detail: "That column is full." });
        }
        game.moves.push(body.column);
      } else if (match[2] === "bot-move") {
        if (described.status !== "playing") {
          throw Object.assign(new Error("That game is over."), { status: 409, detail: "That game is over." });
        }
        if (body.skill !== null && body.skill !== undefined) game.skill = body.skill;
        game.moves.push(this.botColumn(game));
      } else {
        // Undo hands the board back to the human, so it takes back whole
        // turns rather than single plies.
        const humanPlayer = game.botPlayer === 1 ? 2 : 1;
        const humanPlies = game.moves.filter((_, i) => 1 + (i % 2) === humanPlayer).length;
        if (!humanPlies) {
          throw Object.assign(new Error("Nothing to undo."), { status: 409, detail: "Nothing to undo." });
        }
        while (game.moves.length) {
          const removed = 1 + ((game.moves.length - 1) % 2);
          game.moves.pop();
          if (removed === humanPlayer) break;
        }
      }
      return this.respond(game);
    }

    throw Object.assign(new Error(`HTTP 404`), { status: 404, detail: `No route ${method} ${route}` });
  }

  fetch(url, options) {
    let payload;
    try {
      payload = this.handle(url, options);
    } catch (error) {
      const status = error.status ?? 500;
      return Promise.resolve({
        ok: false,
        status,
        // A proxy or a crash can answer with HTML. The client has to survive
        // a body it cannot parse, so the harness can produce one.
        json: async () => {
          if (error.notJson) throw new SyntaxError("Unexpected token < in JSON");
          return { detail: error.detail ?? String(error.message) };
        },
      });
    }
    return Promise.resolve({ ok: true, status: 200, json: async () => payload });
  }
}

/* ----------------------------------------------------------------- loader */

const SOURCE = fs.readFileSync(path.join(WEB, "app.js"), "utf8");

// Everything the script declares lives in its own scope, which is exactly
// where it belongs. The epilogue is the seam: it hands the tests the same
// names the browser gives the page, without the file exporting anything.
const EPILOGUE = `
;globalThis.__app = {
  state, el, render, renderBoard, renderStatus, renderEvalBar, renderLines,
  renderMoveList, renderHistory, sideClass, humanPlayer, myTurn, solverMode,
  recommendedColumn, principalVariationCells, columnHeights, lastMoveCell,
  engineWidth, shortEval, formatSigned, relativeTime, disagreementNote,
  newGame, playColumn, undo, botMove, loadHistory, clearHistory, api,
};
`;

/** Let every already-resolved promise in the script settle. */
async function settle(turns = 200) {
  for (let i = 0; i < turns; i++) await Promise.resolve();
  await new Promise((resolve) => setImmediate(resolve));
  for (let i = 0; i < turns; i++) await Promise.resolve();
}

/**
 * Run `web/app.js` against a fresh document and server, exactly as the page
 * does -- boot included, since the script starts a game the moment it loads.
 */
async function loadApp({ controls = {}, server = new FakeServer() } = {}) {
  const document = makeDocument(controls);
  const context = vm.createContext({
    document,
    fetch: (url, options) => server.fetch(url, options),
    console,
    setTimeout,
    clearTimeout,
    setImmediate,
  });
  vm.runInContext(SOURCE + EPILOGUE, context, { filename: "web/app.js" });
  const app = context.__app;
  await settle();
  return { app, document, server, el: app.el, state: app.state, settle };
}

/** Play a column the way a click does, then wait for the bot's reply. */
async function play(loaded, column) {
  await loaded.app.playColumn(column);
  await loaded.settle();
}

/** Board cells as a grid of class names, for asserting on what is drawn. */
function drawnBoard(loaded) {
  return loaded.el.board.querySelectorAll(".cell").map((cell) => {
    const child = cell.children[0];
    return child ? child.className : "";
  });
}

module.exports = {
  COLUMNS,
  ROWS,
  FakeServer,
  Node,
  loadApp,
  play,
  drawnBoard,
  gridOf,
  winnerOf,
  idsInMarkup,
  matchesSelector,
};
