// Thin wrapper around Stockfish 17.1 (lite, single-threaded WASM build from the
// `stockfish` npm package, GPLv3 — see engine/COPYING.txt) running in a Web Worker.

let worker = null;
let ready = null;
const listeners = new Set();

function send(cmd) { worker.postMessage(cmd); }

function waitFor(prefix) {
  return new Promise(resolve => {
    const fn = line => {
      if (line.startsWith(prefix)) { listeners.delete(fn); resolve(line); }
    };
    listeners.add(fn);
  });
}

export function initEngine() {
  if (ready) return ready;
  worker = new Worker('engine/stockfish.js');
  worker.onmessage = e => { for (const fn of [...listeners]) fn(String(e.data)); };
  ready = (async () => {
    const ok = waitFor('uciok');
    send('uci');
    await ok;
    const rd = waitFor('readyok');
    send('isready');
    await rd;
  })();
  return ready;
}

let queue = Promise.resolve();

/**
 * Ask Stockfish for a move.
 * opts: { elo } uses UCI_LimitStrength/UCI_Elo (1320–3190),
 *       { skill, depth } uses Skill Level (0–20) with a depth cap for sub-1320 play.
 * Returns a UCI string like "e2e4" or "e7e8q".
 */
export function engineMove(fen, opts) {
  const job = queue.then(async () => {
    await initEngine();
    send('ucinewgame');
    if (opts.elo) {
      send('setoption name UCI_LimitStrength value true');
      send(`setoption name UCI_Elo value ${opts.elo}`);
      send('setoption name Skill Level value 20');
    } else {
      send('setoption name UCI_LimitStrength value false');
      send(`setoption name Skill Level value ${opts.skill ?? 20}`);
    }
    const rd = waitFor('readyok');
    send('isready');
    await rd;
    const best = waitFor('bestmove');
    send(`position fen ${fen}`);
    send(opts.depth ? `go depth ${opts.depth} movetime ${opts.movetime ?? 800}` : `go movetime ${opts.movetime ?? 800}`);
    return (await best).split(' ')[1];
  });
  queue = job.catch(() => {});
  return job;
}
