import { predict } from './model.js';

const VALUE = { p: 1, n: 3, b: 3.2, r: 5, q: 9, k: 0 };

function material(game) {
  // score from the perspective of the side to move
  let s = 0;
  for (const row of game.board()) for (const p of row) if (p) s += (p.color === 'w' ? 1 : -1) * VALUE[p.type];
  return game.turn() === 'w' ? s : -s;
}

function negamax(game, depth) {
  if (game.isCheckmate()) return -1000 - depth;
  if (game.isDraw()) return 0;
  if (depth === 0) return material(game);
  let best = -Infinity;
  for (const m of game.moves({ verbose: true })) {
    game.move(m);
    best = Math.max(best, -negamax(game, depth - 1));
    game.undo();
  }
  return best;
}

export const BOTS = {
  random: {
    label: 'Random mover',
    async pick(game) {
      const ms = game.moves({ verbose: true });
      return ms[Math.floor(Math.random() * ms.length)];
    },
  },
  material: {
    label: 'Material grabber (2-ply)',
    async pick(game) {
      let best = -Infinity, choices = [];
      for (const m of game.moves({ verbose: true })) {
        game.move(m);
        const score = -negamax(game, 1);
        game.undo();
        if (score > best + 1e-9) { best = score; choices = [m]; }
        else if (Math.abs(score - best) < 1e-9) choices.push(m);
      }
      return choices[Math.floor(Math.random() * choices.length)];
    },
  },
};

for (const elo of [800, 1200, 1600, 2000]) {
  BOTS[`model${elo}`] = {
    label: `Model as a ${elo} player`,
    elo,
    async pick(game, whiteElo, blackElo) {
      // Sample like a human would: sharpen a bit, drop long-tail moves.
      const preds = await predict(game, whiteElo, blackElo);
      const pool = preds.filter(x => x.p >= 0.03);
      const weights = pool.map(x => Math.pow(x.p, 1 / 0.7));
      const total = weights.reduce((a, b) => a + b, 0);
      let r = Math.random() * total;
      for (let i = 0; i < pool.length; i++) { r -= weights[i]; if (r <= 0) return pool[i].move; }
      return preds[0].move;
    },
  };
}
