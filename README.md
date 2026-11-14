# Chess Prediction

JP Vestjens, Hollin Glaze, Jaxon Mathews. CSSE/MA416 Deep Learning term project.

Given a chess position and the Elo of both players, our model predicts the next move a human will play. Most chess AI hunts for the strongest move. We wanted the move a real person makes, blunders included.

Live demo: https://rhit-vestjejj.github.io/ChessPrediction/

## Results

The final model (`chess_lite_demo_2.ipynb`) was tested on 929,078 positions from games it never saw:

- top-1 44.4%, top-3 71.8%, top-5 82.4%
- castling positions only: top-1 74.2%

How we got there:

- Always guess the most common move (g8f6): 1.04%
- Best n-gram baseline, last 3 moves: 19.04%
- First transformer, 40k games: 26.1%
- Same model with a legal-move mask: 33.7%
- 100k games, Elo buckets, castling and en passant inputs, piece-aware attention bias, early stopping: 41.6%
- 110k games, full geometric attention bias: 44.4%

Our plan set 30% as the minimum, 40% as reasonable, and beating Allie's 55.7% as the stretch goal. We cleared reasonable. The stretch goal is still stretching.

## How it works

Each position becomes 64 square tokens (13 piece types, empty included) plus six extra tokens: a CLS summary token, side to move, white Elo, black Elo, castling rights, and the en passant file. Elo goes into 100-point buckets (36 total), each with a learned embedding. Both players' Elo embeddings also get added to every square, scaled by 0.3.

The encoder has 6 transformer layers (width 256, 8 heads, feed-forward 1024, about 5.6M parameters). Attention gets a learned geometric bias that depends on the pieces on both squares and their offset, following Chessformer. The CLS token goes through a linear head that scores all 1,913 moves seen in the data. Illegal moves get set to -inf before softmax. Before the mask, the model sometimes suggested moving pieces that were not on the board, which FIDE frowns upon.

Training uses cross-entropy, AdamW (lr 3e-4, weight decay 0.01), batch size 256, gradient clipping at 1.0, bf16 autocast, and torch.compile. The learning rate halves when validation top-1 stalls, and training stops after 3 epochs with no gain. Games split 80/10/10 by a hash of the move list, so duplicate games always land in the same split. The final run took 32 epochs at about 7 minutes each on a department GPU (see `train_log.txt`).

## Files

- `data_analysis.ipynb`: scans all 14.2M games. Sources, Elo, game length, duplicates, move class counts, split plan.
- `baseline.ipynb`: n-gram baselines (global, per-ply, last 1 to 3 moves, with and without Elo groups) on 868k games.
- `chess_lite.ipynb`: first transformer. 40k games, 5 epochs. Saved output shows 29.3% top-1.
- `chess_lite_demo.ipynb`: 100k games with early stopping and checkpoint resume. Writes `chess_model_best.pt` and `chess_model_last.pt`. Saved output shows 41.6%.
- `chess_lite_demo_2.ipynb`: the final model. Writes `chess_model_best_demo2.pt`, `chess_model_last_demo2.pt`, `vocab_demo2.json`, and `train_log.txt`.
- `data_analysis.html`, `baseline.html`, `chess_lite.html`: exported notebooks with outputs, readable without running anything.
- `vocab.json`, `vocab_demo2.json`: move string to class index.
- `chess_model_best*.pt` hold the best weights by validation top-1. `chess_model_last*.pt` hold the full checkpoint (model, optimizer, scheduler, epoch) for resuming.
- `export_onnx.py`: converts `chess_model_best_demo2.pt` to `docs/model.onnx` and copies the vocab to `docs/vocab.json`.
- `docs/`: the web demo, served by GitHub Pages. `model.js` runs the ONNX model with onnxruntime-web and masks illegal moves. `app.js` runs the board and game loop. `engine.js` and `bots.js` wrap Stockfish 17.1 (WASM build in `docs/engine/`) as the opponent.
- `Project Information/`: lit review, plan, sprint reports, final slides, and rubric.

## Setup

We used Python 3.11. Training wants a CUDA GPU. Everything else runs fine on a CPU.

```
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas pyarrow matplotlib chess torch tqdm jupyter
pip install onnx onnxruntime   # only needed for export_onnx.py
```

## Data

The data is the Hugging Face dataset [angeluriot/chess_games](https://huggingface.co/datasets/angeluriot/chess_games): 14.2M games from 13 sources in a single file, `dataset.parquet`. Download it into the repo root:

```
curl -L -o dataset.parquet https://huggingface.co/datasets/angeluriot/chess_games/resolve/main/dataset.parquet
```

`data_analysis.ipynb` and `baseline.ipynb` read `dataset.parquet` from the folder they run in. The three `chess_lite` notebooks read and write everything in `data_dir`, set in the second cell. It points at our team folder on the department server (`/work/cssema416/202710students/15`). Change it to `'.'` to use the repo root. These notebooks also save the train/val/test splits there as `.npy` files, which git ignores.

## Running

Open the notebooks in Jupyter and run them top to bottom. `data_analysis.ipynb` uses every CPU core for the full scan, so your laptop may get warm.

To retrain the final model, run `chess_lite_demo_2.ipynb`. It resumes from `chess_model_last_demo2.pt` if that file sits in `data_dir`, so move it somewhere else for a fresh run. On a CPU, set `USE_COMPILE = False` and pack a lunch. Possibly several.

To evaluate the saved weights without training, skip the three cells before the last one (sanity check, model setup, training loop). The last cell loads `chess_model_best_demo2.pt` and prints test and castling accuracy. The data cells still replay all 110k games to rebuild the test split.

For the web demo, use the live link or serve it locally:

```
python3 export_onnx.py   # only after retraining; docs/model.onnx is already built
cd docs
python3 -m http.server 8000
```

Then open http://localhost:8000. The page loads onnxruntime-web and chess.js from a CDN, so it needs internet. You play Stockfish at a chosen strength. Before each of your moves the model locks in a guess, then reveals it after you play. A scoreboard keeps count, which gets humbling fast. Run `export_onnx.py` from the repo root, since it opens the checkpoint and vocab by relative path.

## Team

- Hollin Glaze: board encoding and game replay, transformer encoder, first geometric bias, castling and en passant inputs, long training runs with early stopping, LR scheduling, checkpoint resume, gradient clipping.
- JP Vestjens: data analysis, baseline, legal-move mask, first Elo input, piece-aware geometric bias, training speedups, web demo.
- Jaxon Mathews: Chessformer research, Elo buckets and Elo scaling.

## Related work

- Monroe et al., Chessformer: A Unified Architecture for Chess Modeling (ICLR 2026). Square tokens and geometric attention bias. We borrowed both.
- Zhang et al., Human-Aligned Chess with a Bit of Search (ICLR 2025). Allie predicts human moves conditioned on rating at 55.7% top-1. It convinced us to feed in Elo.
- Zhong, Huang, and Greenberg, Predicting Human Chess Moves (arXiv:2512.01880, 2025). Skill-group n-gram models. Our baseline follows this paper.

## AI usage

We used it to impliment code from our ideas.
- Using the transformer (we got from the paper) and geometric bias
- Adding the legal move mask
- Adding en passant and castling tokens
- We had used to when we were stuck on training times to see what we could use to make it faster
- Front end was claude
- Some debugging issues
In general the code was aided by Claude but the ideas were still ours.