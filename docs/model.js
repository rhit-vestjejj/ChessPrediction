// Runs the exported ChessModel (see export_onnx.py) in the browser with onnxruntime-web.
// Input encoding mirrors chess_lite_demo.ipynb:
//   boards[64]: a1=0 ... h8=63, value = piece_type (P1 N2 B3 R4 Q5 K6), +6 for black, 0 empty
//   side      : turn (0 white / 1 black) + 2 * castling bits (K=1 Q=2 k=4 q=8) + 32 * ep file (8 = none)
//   elo[2]    : [white_elo, black_elo]

const ort = window.ort;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
ort.env.wasm.numThreads = 1;

const PIECE_TYPE = { p: 1, n: 2, b: 3, r: 4, q: 5, k: 6 };

let session = null;
let vocab = null;

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
  // content-length may be the gzip size (GitHub Pages compresses), so it's only a progress hint.
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) onProgress?.(Math.min(1, got / total));
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf;
}

export async function loadModel(onProgress) {
  const [bytes, v] = await Promise.all([
    fetchWithProgress('model.onnx', onProgress),
    fetch('vocab.json').then(r => r.json()),
  ]);
  vocab = v;
  session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
}

function encode(game) {
  const boards = new BigInt64Array(64);
  const rows = game.board(); // rows[0] is rank 8
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const p = rows[r][f];
      if (p) boards[(7 - r) * 8 + f] = BigInt(PIECE_TYPE[p.type] + (p.color === 'w' ? 0 : 6));
    }
  }
  const castleField = game.fen().split(' ')[2];
  const castle = (castleField.includes('K') ? 1 : 0) + (castleField.includes('Q') ? 2 : 0)
               + (castleField.includes('k') ? 4 : 0) + (castleField.includes('q') ? 8 : 0);
  // python-chess's has_legal_en_passant(): only count ep if a legal ep capture exists
  const epMove = game.moves({ verbose: true }).find(m => m.flags.includes('e'));
  const ep = epMove ? epMove.to.charCodeAt(0) - 97 : 8;
  const side = (game.turn() === 'w' ? 0 : 1) + 2 * castle + 32 * ep;
  return { boards, side };
}

export const uciOf = m => m.from + m.to + (m.promotion || '');

/**
 * Predict the move distribution for the side to move.
 * Returns legal moves sorted by probability: [{ move, uci, san, p }]
 */
export async function predict(game, whiteElo, blackElo) {
  const { boards, side } = encode(game);
  const feeds = {
    boards: new ort.Tensor('int64', boards, [1, 64]),
    side: new ort.Tensor('int64', BigInt64Array.from([BigInt(side)]), [1]),
    elo: new ort.Tensor('int64', BigInt64Array.from([BigInt(whiteElo), BigInt(blackElo)]), [1, 2]),
  };
  const { logits } = await session.run(feeds);
  const out = logits.data;

  const legal = game.moves({ verbose: true });
  const scored = legal.map(m => {
    const uci = uciOf(m);
    const idx = vocab[uci];
    return { move: m, uci, san: m.san, logit: idx === undefined ? -Infinity : out[idx] };
  });
  const max = Math.max(...scored.map(s => s.logit));
  let sum = 0;
  for (const s of scored) { s.p = Math.exp(s.logit - max); sum += s.p; }
  for (const s of scored) s.p /= sum;
  return scored.sort((a, b) => b.p - a.p);
}
