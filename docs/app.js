import { Chess } from 'https://cdn.jsdelivr.net/npm/chess.js@1.0.0/dist/esm/chess.js';
import { loadModel, predict, uciOf } from './model.js';
import { BOTS } from './bots.js';
import { initEngine } from './engine.js';

const $ = id => document.getElementById(id);
const FILES = 'abcdefgh';

const state = {
  game: new Chess(),
  userColor: 'w',
  userElo: 1500,
  botKey: 'sf1400',
  pending: null,        // Promise<predictions> for the user's current turn
  selected: null,
  busy: true,
  lastMove: null,
  log: [],              // { color, san, mark }
  stats: { n: 0, top1: 0, top3: 0, pSum: 0 },
  gameId: 0,
};

/* ---------- setup ---------- */

for (const [key, bot] of Object.entries(BOTS)) {
  const opt = document.createElement('option');
  opt.value = key;
  opt.textContent = bot.label;
  $('bot').append(opt);
}
$('bot').value = state.botKey;

$('elo').addEventListener('input', e => { $('elo-out').textContent = e.target.value; });
$('new-game').addEventListener('click', newGame);

function elos() {
  const bot = BOTS[state.botKey];
  const botElo = bot.elo ?? state.userElo;
  return state.userColor === 'w' ? [state.userElo, botElo] : [botElo, state.userElo];
}

/* ---------- board rendering ---------- */

function pieceSvg(p) {
  return `<svg class="piece" viewBox="0 0 40 40"><use href="pieces.svg#${p.color}${p.type}"/></svg>`;
}

function squareAt(r, f) {
  // r, f are visual row/col (0 = top-left)
  const flip = state.userColor === 'b';
  const file = flip ? 7 - f : f;
  const rank = flip ? r + 1 : 8 - r;
  return FILES[file] + rank;
}

function render() {
  const { game } = state;
  const board = $('board');
  const targets = new Map();
  if (state.selected) {
    for (const m of game.moves({ square: state.selected, verbose: true })) targets.set(m.to, m);
  }
  let checkSq = null;
  if (game.inCheck()) {
    for (const row of game.board()) for (const p of row) if (p && p.type === 'k' && p.color === game.turn()) checkSq = p.square;
  }
  const userTurn = !state.busy && game.turn() === state.userColor && !game.isGameOver();

  let html = '';
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const sq = squareAt(r, f);
      const p = game.get(sq);
      const fileIdx = sq.charCodeAt(0) - 97, rank = +sq[1];
      const cls = ['sq', (fileIdx + rank) % 2 === 0 ? 'light' : 'dark'];
      if (state.lastMove && (sq === state.lastMove.from || sq === state.lastMove.to)) cls.push('last');
      if (sq === state.selected) cls.push('sel');
      if (sq === checkSq) cls.push('check');
      if (targets.has(sq)) cls.push('target', p || targets.get(sq).flags.includes('e') ? 'capture' : '');
      if (userTurn && p && p.color === state.userColor) cls.push('mine');
      let inner = p ? pieceSvg(p) : '';
      if (r === 7) inner += `<span class="coord file">${sq[0]}</span>`;
      if (f === 0) inner += `<span class="coord rank">${sq[1]}</span>`;
      html += `<div class="${cls.join(' ')}" data-sq="${sq}">${inner}</div>`;
    }
  }
  board.innerHTML = html;
}

/* ---------- input: click + drag ---------- */

let drag = null;

function sqFromPoint(x, y) {
  const el = document.elementFromPoint(x, y)?.closest('.sq');
  return el && $('board').contains(el) ? el.dataset.sq : null;
}

$('board').addEventListener('pointerdown', e => {
  const sqEl = e.target.closest('.sq');
  if (!sqEl || state.busy || state.game.turn() !== state.userColor || state.game.isGameOver()) return;
  const sq = sqEl.dataset.sq;
  const p = state.game.get(sq);

  if (state.selected && state.selected !== sq && isTarget(state.selected, sq)) {
    tryMove(state.selected, sq);
    return;
  }
  if (p && p.color === state.userColor) {
    state.selected = sq;
    render();
    const pieceEl = $('board').querySelector(`[data-sq="${sq}"] .piece`);
    const size = sqEl.getBoundingClientRect().width;
    const ghost = document.createElement('div');
    ghost.className = 'drag-ghost';
    ghost.style.width = ghost.style.height = size + 'px';
    ghost.innerHTML = pieceSvg(p);
    ghost.style.left = e.clientX + 'px';
    ghost.style.top = e.clientY + 'px';
    document.body.append(ghost);
    pieceEl?.classList.add('dragging');
    drag = { from: sq, ghost };
    $('board').setPointerCapture(e.pointerId);
  } else {
    state.selected = null;
    render();
  }
});

$('board').addEventListener('pointermove', e => {
  if (!drag) return;
  drag.ghost.style.left = e.clientX + 'px';
  drag.ghost.style.top = e.clientY + 'px';
});

function endDrag(e) {
  if (!drag) return;
  const { from, ghost } = drag;
  drag = null;
  ghost.remove();
  const to = e ? sqFromPoint(e.clientX, e.clientY) : null;
  if (to && to !== from && isTarget(from, to)) tryMove(from, to);
  else render();
}
$('board').addEventListener('pointerup', endDrag);
$('board').addEventListener('pointercancel', () => endDrag(null));

function isTarget(from, to) {
  return state.game.moves({ square: from, verbose: true }).some(m => m.to === to);
}

function choosePromotion(color) {
  return new Promise(resolve => {
    const box = $('promo');
    box.innerHTML = `<div class="promo-row">${['q', 'r', 'b', 'n'].map(t =>
      `<button data-t="${t}" aria-label="${t}"><svg viewBox="0 0 40 40"><use href="pieces.svg#${color}${t}"/></svg></button>`).join('')}</div>`;
    box.hidden = false;
    box.onclick = e => {
      const b = e.target.closest('button');
      box.hidden = true;
      resolve(b ? b.dataset.t : null);
    };
  });
}

async function tryMove(from, to) {
  const { game } = state;
  const candidates = game.moves({ square: from, verbose: true }).filter(m => m.to === to);
  if (!candidates.length) return;
  let move = candidates[0];
  if (candidates.length > 1) {
    const t = await choosePromotion(state.userColor);
    if (!t) { state.selected = null; render(); return; }
    move = candidates.find(m => m.promotion === t);
  }
  state.selected = null;
  state.busy = true;
  const id = state.gameId;
  const preds = await state.pending;
  if (id !== state.gameId) return;

  const played = game.move(move);
  state.lastMove = played;
  const uci = uciOf(played);
  const rank = preds.findIndex(x => x.uci === uci);
  const p = preds[rank]?.p ?? 0;
  const mark = rank === 0 ? 'hit' : rank < 3 ? 'near' : 'miss';
  const s = state.stats;
  s.n++; s.pSum += p;
  if (rank === 0) s.top1++;
  if (rank < 3) s.top3++;
  state.log.push({ color: played.color, san: played.san, mark });

  $('lock').hidden = true;
  reveal(preds, rank, played.san);
  updateStats();
  renderMoves();
  render();
  if (checkGameOver()) return;
  setTimeout(() => botTurn(id), 250);
}

/* ---------- turn flow ---------- */

async function newGame() {
  state.gameId++;
  state.userElo = +$('elo').value;
  state.botKey = $('bot').value;
  const c = document.querySelector('input[name=color]:checked').value;
  state.userColor = c === 'r' ? (Math.random() < 0.5 ? 'w' : 'b') : c;
  state.game = new Chess();
  state.selected = null;
  state.lastMove = null;
  state.log = [];
  state.stats = { n: 0, top1: 0, top3: 0, pSum: 0 };
  $('promo').hidden = true;
  $('lock').hidden = true;
  $('guess').innerHTML = `<p class="muted">Waiting for the opponent…</p>`;
  updateStats();
  renderMoves();
  if (state.userColor === 'w') startUserTurn();
  else { state.busy = true; render(); botTurn(state.gameId); }
}

function startUserTurn() {
  const [w, b] = elos();
  state.pending = predict(state.game, w, b);
  state.busy = false;
  setStatus(`Your move${state.game.inCheck() ? ' (check!)' : ''}`);
  const lock = $('lock');
  lock.hidden = false;
  lock.innerHTML = `<span class="lock-icon" aria-hidden="true">🔒</span><span><strong>Next guess locked in.</strong> Your move.</span>`;
  if (!state.stats.n) {
    $('guess').innerHTML = `<p class="muted">The model has already predicted your move. Play it and the guess is revealed here.</p>`;
  }
  render();
}

async function botTurn(id) {
  if (id !== state.gameId) return;
  state.busy = true;
  setStatus('Opponent is thinking…');
  render();
  const [w, b] = elos();
  const [move] = await Promise.all([
    BOTS[state.botKey].pick(new Chess(state.game.fen()), w, b),
    new Promise(r => setTimeout(r, 350)),
  ]);
  if (id !== state.gameId) return;
  const played = state.game.move({ from: move.from, to: move.to, promotion: move.promotion });
  state.lastMove = played;
  state.log.push({ color: played.color, san: played.san, mark: null });
  renderMoves();
  if (checkGameOver()) { render(); return; }
  startUserTurn();
}

function checkGameOver() {
  const { game } = state;
  if (!game.isGameOver()) return false;
  state.busy = true;
  $('lock').hidden = true;
  let msg;
  if (game.isCheckmate()) msg = game.turn() === state.userColor ? 'Checkmate. You lost.' : 'Checkmate. You win!';
  else if (game.isStalemate()) msg = 'Stalemate.';
  else if (game.isThreefoldRepetition()) msg = 'Draw by repetition.';
  else if (game.isInsufficientMaterial()) msg = 'Draw: insufficient material.';
  else msg = 'Draw.';
  const { n, top1 } = state.stats;
  setStatus(`${msg} The model called ${top1} of your ${n} moves exactly.`);
  render();
  return true;
}

/* ---------- side panel ---------- */

function setStatus(t) { $('status').textContent = t; }

const pct = x => x === 0 ? '0%' : (x * 100 >= 10 ? Math.round(x * 100) : (x * 100).toFixed(1)) + '%';
const esc = s => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);

function reveal(preds, rank, san) {
  const top = preds[0];
  const badge = rank === 0 ? '<span class="badge hit">Called it</span>'
    : rank < 3 ? `<span class="badge near">#${rank + 1} pick</span>`
    : '<span class="badge miss">Missed</span>';
  const sub = rank === 0
    ? `It gave your move ${pct(top.p)}.`
    : `Its top guess was <strong>${esc(top.san)}</strong> (${pct(top.p)}). Your move was #${rank + 1} of ${preds.length} at ${pct(preds[rank].p)}.`;
  const row = (x, i) => `
    <li class="bar${i === rank ? ' you' : ''}">
      <span class="rank mono">${i + 1}</span><span class="san mono">${esc(x.san)}</span>
      <span class="track"><span class="fill" style="width:${Math.max(1, x.p * 100)}%"></span></span>
      <span class="pct mono">${pct(x.p)}</span>
    </li>`;
  let bars = preds.slice(0, 5).map(row).join('');
  if (rank >= 5) bars += `<li class="bar-gap">⋯</li>` + row(preds[rank], rank);
  $('guess').innerHTML = `
    <p class="guess-label">Move ${Math.ceil(state.log.length / 2)}</p>
    <div class="verdict"><span class="played">You played <strong class="mono">${esc(san)}</strong></span>${badge}</div>
    <p class="verdict-sub">${sub}</p>
    <ol class="bars">${bars}</ol>`;
}

function updateStats() {
  const { n, top1, top3, pSum } = state.stats;
  $('s-top1').textContent = n ? pct(top1 / n) : '–';
  $('s-top3').textContent = n ? pct(top3 / n) : '–';
  $('s-avgp').textContent = n ? pct(pSum / n) : '–';
  $('s-top1-n').textContent = n ? `${top1}/${n}` : '';
  $('s-top3-n').textContent = n ? `${top3}/${n}` : '';
}

function renderMoves() {
  const ol = $('moves');
  let html = '';
  for (let i = 0; i < state.log.length; i += 2) {
    const cell = m => m
      ? `<span class="mv mono">${m.mark ? `<span class="dot ${m.mark}"></span>` : ''}${esc(m.san)}</span>`
      : '<span></span>';
    html += `<li><span class="num mono">${i / 2 + 1}.</span>${cell(state.log[i])}${cell(state.log[i + 1])}</li>`;
  }
  ol.innerHTML = html;
  ol.scrollTop = ol.scrollHeight;
}

/* ---------- boot ---------- */

render();
Promise.all([
  loadModel(f => { $('loading-bar').style.width = `${Math.round(f * 100)}%`; }),
  initEngine(),
])
  .then(() => {
    $('loading').hidden = true;
    $('new-game').disabled = false;
    newGame();
  })
  .catch(err => {
    console.error(err);
    $('loading-text').textContent = `Couldn't load: ${err.message}`;
  });
