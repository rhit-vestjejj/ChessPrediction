import { engineMove } from './engine.js';

// Opponents are Stockfish at different strengths. `elo` is what we tell the
// prediction model the opponent is rated (it's one of the model's inputs).
// UCI_Elo is Stockfish's own calibration (anchored at 120s+1s games), so the
// numbered levels are approximate.

function fromUci(game, uci) {
  return game.moves({ verbose: true }).find(m => m.from + m.to + (m.promotion || '') === uci);
}

function stockfish(label, elo, opts) {
  return {
    label,
    elo,
    async pick(game) {
      const uci = await engineMove(game.fen(), opts);
      return fromUci(game, uci);
    },
  };
}

export const BOTS = {
  sf_beginner: stockfish('Beginner (~800)', 800, { skill: 0, depth: 1, movetime: 300 }),
  sf_casual: stockfish('Casual (~1100)', 1100, { skill: 2, depth: 3, movetime: 400 }),
  sf1400: stockfish('Stockfish 1400', 1400, { elo: 1400 }),
  sf1700: stockfish('Stockfish 1700', 1700, { elo: 1700 }),
  sf2000: stockfish('Stockfish 2000', 2000, { elo: 2000 }),
  sf2400: stockfish('Stockfish 2400', 2400, { elo: 2400 }),
  random: {
    label: 'Random mover',
    async pick(game) {
      const ms = game.moves({ verbose: true });
      return ms[Math.floor(Math.random() * ms.length)];
    },
  },
};
