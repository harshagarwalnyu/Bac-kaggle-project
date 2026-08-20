/* What the front end promises, written down.
 *
 * The 348 python tests cover the search, the network and the API. They cannot
 * see the page, and the two defects found by hand on 2026-08-20 both lived
 * here: stones coloured by seat instead of by role, and a legend that
 * contradicted the marks it was explaining. Neither was subtle once seen, and
 * neither was reachable from python.
 *
 * Every test below drives the real `web/app.js` through the real boot path.
 * Nothing is stubbed except the browser and the server.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const { loadApp, play, drawnBoard, idsInMarkup, FakeServer, ROWS, COLUMNS } = require("./harness.js");

const SOURCE = fs.readFileSync(path.join(__dirname, "..", "app.js"), "utf8");

/** Drive the game to a finished state: four stones in one column beats a bot
 *  that always answers in column 0. */
async function humanWins(loaded) {
  for (const column of [3, 3, 3, 3]) await play(loaded, column);
  return loaded;
}

const classesOf = (loaded, selector) =>
  loaded.el.board.querySelectorAll(selector).map((node) => node.className);

/* ------------------------------------------------------------- contract */

test("every element the script reaches for exists in the markup", () => {
  const wanted = [...SOURCE.matchAll(/getElementById\("([^"]+)"\)/g)].map((m) => m[1]);
  const declared = new Set(idsInMarkup());
  assert.ok(wanted.length > 10, "expected the script to look up many ids");
  for (const id of wanted) {
    assert.ok(declared.has(id), `app.js reads #${id}, which index.html does not declare`);
  }
});

test("booting starts a game at the settings the page shows and draws a full board", async () => {
  const loaded = await loadApp({ controls: { skill: "4" } });
  const [first] = loaded.server.requests;
  assert.equal(first.url, "/api/games");
  assert.deepEqual(first.body, { skill: 4, bot_first: false });
  assert.equal(loaded.el.board.querySelectorAll(".column").length, COLUMNS);
  assert.equal(loaded.el.board.querySelectorAll(".cell").length, COLUMNS * ROWS);
  assert.match(loaded.el.status.textContent, /^Your move \(ply 1\)/);
});

/* ------------------------------------------------- colour follows the role */

test("the human owns the yellow stones when the human opens", async () => {
  const loaded = await loadApp();
  await play(loaded, 3);
  const discs = classesOf(loaded, ".disc");
  assert.deepEqual(discs.sort(), ["disc bot last", "disc you"].sort());
});

test("the human still owns the yellow stones when the bot opens", async () => {
  const loaded = await loadApp({ controls: { "bot-first": true } });
  assert.equal(loaded.state.game.bot_player, 1, "the bot took the first seat");
  await play(loaded, 2);

  const discs = classesOf(loaded, ".disc");
  const yours = discs.filter((c) => c.includes("you")).length;
  const theirs = discs.filter((c) => c.includes("bot")).length;
  // Player 1 is the bot in this game. Colouring by seat would hand it the
  // human's yellow and contradict the footnote under the board.
  assert.equal(yours, 1, `expected one human stone, got classes ${JSON.stringify(discs)}`);
  assert.equal(theirs, 2, `expected two bot stones, got classes ${JSON.stringify(discs)}`);
});

test("no seat-numbered classes survive anywhere in the page", async () => {
  const loaded = await loadApp();
  await play(loaded, 3);
  const painted = loaded.el.board.querySelectorAll(".disc").concat(
    loaded.el.board.querySelectorAll(".ghost"),
  );
  for (const node of painted) {
    assert.doesNotMatch(node.className, /\bp[12]\b/, `stale seat class on ${node.className}`);
  }
  const css = fs.readFileSync(path.join(__dirname, "..", "style.css"), "utf8");
  assert.doesNotMatch(css, /\.(disc|ghost|ghost-key)\.p[12]\b/);
});

test("the predicted line is drawn from the side to move, alternating", async () => {
  const loaded = await loadApp();
  const ghosts = loaded.el.board
    .querySelectorAll(".ghost")
    .sort((a, b) => Number(a.textContent) - Number(b.textContent));

  assert.ok(ghosts.length >= 2, "expected the opening line to be drawn");
  assert.match(ghosts[0].className, /\byou\b/, "mark 1 belongs to whoever is to move");
  assert.match(ghosts[1].className, /\bbot\b/, "mark 2 is the reply");
  ghosts.forEach((ghost, index) => assert.equal(ghost.textContent, String(index + 1)));
});

/* -------------------------------------------------------------- the legend */

test("the legend's keys follow the side to move", async () => {
  const loaded = await loadApp();
  assert.equal(loaded.el.ghostKeyNext.className, "ghost-key you");
  assert.equal(loaded.el.ghostKeyReply.className, "ghost-key bot");

  // The bot's think is the whole reason these keys are not hardcoded: the
  // analysis on screen then describes the bot's move, so mark 1 is the bot's.
  loaded.state.game.turn = loaded.state.game.bot_player;
  loaded.app.render();
  assert.equal(loaded.el.ghostKeyNext.className, "ghost-key bot");
  assert.equal(loaded.el.ghostKeyReply.className, "ghost-key you");
});

test("the legend keeps two colours once the game is over", async () => {
  const loaded = await loadApp();
  await humanWins(loaded);

  assert.equal(loaded.state.game.status, "won");
  // The server reports `turn: 0` on a finished game -- there is no side to
  // move. Asking for the colour of player 0 (and of player 3) hands both keys
  // the same colour, and the legend stops distinguishing the two marks it is
  // there to explain.
  assert.notEqual(
    loaded.el.ghostKeyNext.className,
    loaded.el.ghostKeyReply.className,
    "both legend keys ended up the same colour",
  );
  assert.equal(loaded.el.ghostKeyNext.className, "ghost-key you");
  assert.equal(loaded.el.ghostKeyReply.className, "ghost-key bot");
});

/* ------------------------------------------------------------ the eval bar */

test("the eval bar is stated from the human's side whoever is to move", async () => {
  const loaded = await loadApp();
  const mine = Number(loaded.el.evalbarFill.style.height.replace("%", ""));
  assert.ok(mine > 50, `a winning position should fill past half, got ${mine}%`);

  loaded.state.game.turn = loaded.state.game.bot_player;
  loaded.app.render();
  const theirs = Number(loaded.el.evalbarFill.style.height.replace("%", ""));
  assert.ok(theirs < 50, `the same score for the bot should read below half, got ${theirs}%`);
  assert.equal(mine + theirs, 100, "the two readings should mirror each other");
});

test("a proven win is drawn full height and labelled with its distance", async () => {
  const loaded = await loadApp();
  assert.equal(loaded.el.evalbar.className, "evalbar proof");
  assert.equal(loaded.el.evalbarText.textContent, "#5");
  assert.equal(loaded.el.evalbarFill.style.height, "100%");
});

test("with nothing to analyse the bar says so instead of guessing", async () => {
  const loaded = await loadApp();
  await humanWins(loaded);
  assert.equal(loaded.state.analysis, null);
  assert.equal(loaded.el.evalbarText.textContent, "—");
  assert.equal(loaded.el.evalbarFill.style.height, "50%");
  assert.equal(loaded.el.evalbar.className, "evalbar");
});

/* ---------------------------------------------------------------- assist */

test("the assist marks one column and names it on the button", async () => {
  const loaded = await loadApp({ controls: { assist: true } });
  loaded.app.render();

  const recommended = loaded.el.board.querySelectorAll(".column.recommended");
  assert.equal(recommended.length, 1);
  assert.equal(recommended[0].dataset.column, String(loaded.state.analysis.best_move));
  assert.equal(loaded.el.board.querySelectorAll(".cell.landing").length, 1);
  assert.equal(loaded.el.playBest.hidden, false);
  assert.match(loaded.el.playBest.textContent, /column \d$/);
});

test("the assist stays quiet when the recommendation is not mine to take", async () => {
  const loaded = await loadApp({ controls: { assist: true } });

  loaded.state.game.turn = loaded.state.game.bot_player;
  loaded.app.render();
  assert.equal(loaded.app.recommendedColumn(), null, "the bot's best move is not advice for me");
  assert.equal(loaded.el.playBest.hidden, true);
  assert.equal(loaded.el.board.querySelectorAll(".column.recommended").length, 0);
});

test("an illegal recommendation is refused rather than drawn", async () => {
  const loaded = await loadApp({ controls: { assist: true } });
  loaded.state.game.legal_moves = loaded.state.game.legal_moves.filter(
    (column) => column !== loaded.state.analysis.best_move,
  );
  loaded.app.render();
  assert.equal(loaded.app.recommendedColumn(), null);
  assert.equal(loaded.el.playBest.hidden, true);
});

/* ------------------------------------------------------------------ undo */

test("undo is offered only once the human has a turn to take back", async () => {
  const loaded = await loadApp({ controls: { "bot-first": true } });
  assert.equal(loaded.state.game.moves.length, 1, "the bot opened");
  assert.equal(loaded.el.undo.disabled, true, "there is no human turn to undo yet");

  await play(loaded, 4);
  assert.equal(loaded.el.undo.disabled, false);

  await loaded.app.undo();
  await loaded.settle();
  assert.equal(loaded.state.game.moves.length, 1, "back to the bot's opening");
  assert.equal(loaded.el.undo.disabled, true);
});

test("undo asks the server rather than editing the board in place", async () => {
  const loaded = await loadApp();
  await play(loaded, 3);
  loaded.server.requests.length = 0;

  await loaded.app.undo();
  await loaded.settle();
  assert.deepEqual(
    loaded.server.requests.map((r) => r.url),
    [`/api/games/${loaded.state.game.id}/undo`],
  );
  assert.equal(loaded.state.game.moves.length, 0, "undo left a stone on the board");
});

/* ---------------------------------------------------------------- guards */

test("clicks the rules do not allow are dropped, not sent", async () => {
  const loaded = await loadApp();
  const sent = () => loaded.server.requests.filter((r) => r.url.endsWith("/moves")).length;

  loaded.state.game.turn = loaded.state.game.bot_player;
  await play(loaded, 3);
  assert.equal(sent(), 0, "a click during the bot's turn reached the server");

  loaded.state.game.turn = loaded.app.humanPlayer();
  loaded.state.game.legal_moves = [0, 1, 2];
  await play(loaded, 6);
  assert.equal(sent(), 0, "a click on a full column reached the server");

  loaded.state.game.status = "won";
  await play(loaded, 0);
  assert.equal(sent(), 0, "a click after the game ended reached the server");
});

test("a request in flight locks the board and the buttons", async () => {
  const loaded = await loadApp();
  loaded.state.busy = true;
  loaded.app.render();
  assert.equal(loaded.el.undo.disabled, true);
  assert.equal(loaded.el.newGame.disabled, true);

  await play(loaded, 3);
  assert.equal(
    loaded.server.requests.filter((r) => r.url.endsWith("/moves")).length,
    0,
    "a click during a request in flight reached the server",
  );
});

/* --------------------------------------------------------- board geometry */

test("stones stack from the bottom and the newest one is marked", async () => {
  const loaded = await loadApp();
  await play(loaded, 3);
  await play(loaded, 3);

  assert.equal(loaded.app.columnHeights()[3], 2);
  // Spread first: the script runs in its own realm, so its objects are not
  // reference-comparable with ours even when they are identical.
  assert.deepEqual({ ...loaded.app.lastMoveCell() }, { row: ROWS - 2, column: 0 });
  assert.equal(loaded.el.board.querySelectorAll(".disc.last").length, 1);
});

test("a full column and a finished game are marked differently", async () => {
  const loaded = await loadApp();
  loaded.state.game.legal_moves = [0, 1, 2, 4, 5, 6];
  loaded.app.render();
  assert.deepEqual(
    loaded.el.board.querySelectorAll(".column.full").map((c) => c.dataset.column),
    ["3"],
  );
  assert.equal(loaded.el.board.querySelectorAll(".column.over").length, 0);

  const finished = await humanWins(await loadApp());
  assert.equal(finished.el.board.querySelectorAll(".column.over").length, COLUMNS);
});

test("the predicted line stacks on itself and stops at a full column", async () => {
  const loaded = await loadApp();
  loaded.state.analysis.principal_variation = [3, 3, 3];
  loaded.app.render();

  const marks = loaded.el.board
    .querySelectorAll(".ghost")
    .map((g) => ({ order: Number(g.textContent), side: g.className.split(" ")[1] }))
    .sort((a, b) => a.order - b.order);
  assert.equal(marks.length, 3, "three moves in one column need three distinct cells");
  assert.deepEqual(marks.map((m) => m.order), [1, 2, 3]);
  assert.deepEqual(marks.map((m) => m.side), ["you", "bot", "you"]);

  // A column with no room left has nowhere to draw the mark, and inventing a
  // cell for it would put a stone outside the board.
  const grid = loaded.state.game.grid;
  for (let row = 0; row < ROWS; row++) grid[row][3] = 1;
  loaded.app.render();
  assert.equal(loaded.el.board.querySelectorAll(".ghost").length, 0);
});

/* ------------------------------------------------------------- move list */

test("the move list reads as paired turns", async () => {
  const loaded = await loadApp();
  assert.equal(loaded.el.movelistEmpty.hidden, false);

  await play(loaded, 3);
  await play(loaded, 5);
  assert.equal(loaded.el.movelistEmpty.hidden, true);

  const rows = loaded.el.movelist.querySelectorAll("tr");
  assert.equal(rows.length, 2, "four plies make two rows");
  assert.equal(rows[0].textContent, "1.col 3col 0");
  assert.equal(rows[1].className, "current", "the last row is the live one");
});

/* --------------------------------------------------------------- history */

test("the record survives a history endpoint that fails", async () => {
  const server = new FakeServer();
  server.fail = { path: "/api/history", status: 500, detail: "boom" };
  const loaded = await loadApp({ server });
  assert.equal(loaded.state.history.summary, null);
  assert.equal(loaded.state.history.games.length, 0);
  assert.match(loaded.el.record.textContent, /No games finished yet/);
  assert.equal(loaded.state.game.status, "playing", "a missing record must not touch the game");
});

test("finished games are listed with their outcome and clock", async () => {
  const server = new FakeServer();
  server.history = {
    summary: { games: 3, human_wins: 2, bot_wins: 1, draws: 0 },
    games: [
      {
        outcome: "you won", winner: 2, plies: 11, duration_s: 42, skill: 6,
        solver: true, ended: Date.now() / 1000 - 120, moves: [3, 0, 3],
      },
    ],
  };
  const loaded = await loadApp({ server });
  await loaded.app.loadHistory();

  assert.match(loaded.el.record.textContent, /3.*games.*2W.*1L.*0D/);
  const [row] = loaded.el.history.querySelectorAll(".hgame");
  assert.match(row.className, /\bwon\b/);
  assert.match(row.textContent, /you won/);
  assert.match(row.textContent, /11 plies · 42s · skill 6SOLVER/);
  assert.match(row.textContent, /2m ago/);
  assert.equal(row.title, "moves: 3 0 3");
});

test("relative times round down into readable buckets", async () => {
  const { app } = await loadApp();
  const ago = (seconds) => app.relativeTime(Date.now() / 1000 - seconds);
  assert.equal(ago(0), "just now");
  assert.equal(ago(59), "just now");
  assert.equal(ago(60), "1m ago");
  assert.equal(ago(3599), "59m ago");
  assert.equal(ago(3600), "1h ago");
  assert.equal(ago(86400), "1d ago");
  assert.equal(ago(-10), "just now", "a clock skewed into the future must not read as negative");
});

/* -------------------------------------------------------------- wording */

test("the status line is written from the human's point of view", async () => {
  const loaded = await loadApp();
  assert.match(loaded.el.status.textContent, /^Your move/);

  loaded.state.game.turn = loaded.state.game.bot_player;
  loaded.app.render();
  assert.equal(loaded.el.status.textContent, "Bot to move.");

  loaded.state.game.turn = loaded.app.humanPlayer();
  await humanWins(loaded);
  assert.equal(loaded.el.status.textContent, "You win.");
  assert.equal(loaded.el.status.className, "status win");

  loaded.state.game.winner = loaded.state.game.bot_player;
  loaded.app.render();
  assert.equal(loaded.el.status.textContent, "The bot wins.");
  assert.equal(loaded.el.status.className, "status lose");
});

test("engine verdicts are shortened without losing their meaning", async () => {
  const { app } = await loadApp();
  assert.equal(app.shortEval({ exact: false, score: 0.375 }), "+0.38");
  assert.equal(app.shortEval({ exact: false, score: -0.5 }), "-0.50");
  assert.equal(app.shortEval({ exact: true, score: 1, mate_in: 7 }), "#7");
  assert.equal(app.shortEval({ exact: true, score: -1, mate_in: 3 }), "-#3");
  assert.equal(app.shortEval({ exact: true, score: 0, mate_in: 0 }), "=");

  // A proof is a proof: drawing a mate in 11 shorter than a mate in 3 would
  // suggest a weaker win, which is not a thing.
  assert.equal(app.engineWidth({ exact: true, score: 1 }), 1);
  assert.equal(app.engineWidth({ exact: true, score: -1 }), -1);
  assert.equal(app.engineWidth({ exact: true, score: 0 }), 0);
  assert.equal(app.engineWidth({ exact: false, score: 4 }), 1, "heuristics are clamped");
  assert.equal(app.engineWidth({ exact: false, score: -0.25 }), -0.25);
});

test("the disagreement note only speaks where there is a proof to speak from", async () => {
  const { app } = await loadApp();
  const column = (exact, score, preference) => ({
    column: 4, engine: { exact, score, label: "won in 5" }, network: { preference },
  });

  assert.match(
    app.disagreementNote({ columns: [column(false, 0.9, -0.9)] }),
    /No proven verdicts yet/,
    "an unproven column is an opinion, not evidence against the network",
  );
  assert.match(app.disagreementNote({ columns: [column(true, 1, 0.95)] }), /agrees with every proof/);
  assert.match(app.disagreementNote({ columns: [column(true, 1, -0.9)] }), /proven win.*network scores it -0.90/s);
  assert.match(app.disagreementNote({ columns: [column(true, -1, 0.9)] }), /proven loss/);
});

/* --------------------------------------------------------------- errors */

test("the server's own words are what the player is shown", async () => {
  const server = new FakeServer();
  const loaded = await loadApp({ server });
  server.fail = { path: `/api/games/${loaded.state.game.id}/moves`, status: 409, detail: "That column is full." };

  await play(loaded, 3);
  assert.equal(loaded.state.message, "That column is full.");
  assert.equal(loaded.el.status.textContent, "That column is full.");
  assert.equal(loaded.state.busy, false, "input stayed locked after a failure");
});

test("a refused move does not hand the bot a free one", async () => {
  const server = new FakeServer();
  const loaded = await loadApp({ server });
  server.fail = {
    path: `/api/games/${loaded.state.game.id}/moves`,
    status: 409,
    detail: "That column is full.",
  };

  await play(loaded, 3);
  // The human's stone never landed, so it is still the human's turn. Asking
  // the bot to move here lets a rejected click cost a tempo -- and the bot's
  // successful reply wipes the message that explained the refusal.
  assert.equal(loaded.state.game.moves.length, 0, "the bot moved after a refused human move");
  assert.equal(loaded.state.game.turn, loaded.app.humanPlayer());
});

test("a failure with no readable body still names the status", async () => {
  const server = new FakeServer();
  const loaded = await loadApp({ server });
  server.fail = { path: `/api/games/${loaded.state.game.id}/moves`, status: 503, notJson: true };

  await play(loaded, 3);
  assert.equal(loaded.state.message, "HTTP 503");
  assert.equal(loaded.el.status.className, "status", "a plain failure is not a thinking state");
});
