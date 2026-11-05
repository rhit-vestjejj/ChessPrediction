"""Export chess_model_best.pt to ONNX for the in-browser demo (docs/).

Model definition is copied from chess_lite_demo.ipynb (the version that produced
chess_model_best.pt). `side` packs: turn + 2*castling_bits + 32*ep_file (8 = none). The exported graph returns raw
logits (no legal-move mask); the browser masks illegal moves itself.

Usage: python export_onnx.py [--quantize]
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PieceGeometricBias(nn.Module):
    def __init__(self, num_heads, num_extra):
        super().__init__()
        sq = torch.arange(64)
        file, rank = sq % 8, sq // 8
        rel = (file[None, :] - file[:, None] + 7) * 15 + (rank[None, :] - rank[:, None] + 7)
        self.register_buffer('rel', rel)
        self.table = nn.Embedding(13 * 15 * 15, num_heads)
        self.num_extra = num_extra

    def forward(self, boards):
        index = boards[:, :, None] * (15 * 15) + self.rel[None]
        bias = self.table(index).permute(0, 3, 1, 2)
        return F.pad(bias, (0, self.num_extra, 0, self.num_extra))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, bias):
        B, L, D = x.shape
        qkv = self.qkv(self.norm1(x)).view(B, L, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        x = x + self.out(attn.transpose(1, 2).reshape(B, L, D))
        x = x + self.ff(self.norm2(x))
        return x


ELO_BUCKET = 100
NUM_ELO_BUCKETS = 36


class ChessModel(nn.Module):
    def __init__(self, num_moves, d_model=256, num_heads=8, num_layers=6, d_ff=1024, elo_scale=0.3):
        super().__init__()
        self.elo_scale = elo_scale
        self.piece_emb = nn.Embedding(13, d_model)
        self.square_pos = nn.Parameter(torch.zeros(64, d_model))
        self.side_emb = nn.Embedding(2, d_model)
        self.white_elo = nn.Embedding(NUM_ELO_BUCKETS, d_model)
        self.black_elo = nn.Embedding(NUM_ELO_BUCKETS, d_model)
        self.cls = nn.Parameter(torch.zeros(1, d_model))
        self.castle_emb = nn.Embedding(16, d_model)
        self.ep_emb = nn.Embedding(9, d_model)
        self.bias = PieceGeometricBias(num_heads, 6)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, 0.0) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_moves)

    def forward(self, boards, side, elo):
        B = boards.size(0)
        side, castle, ep = side % 2, (side // 2) % 16, side // 32
        bucket = (elo // ELO_BUCKET).clamp(0, NUM_ELO_BUCKETS - 1)
        white, black = self.white_elo(bucket[:, 0]), self.black_elo(bucket[:, 1])
        elo_vec = self.elo_scale * (white + black)
        x = self.piece_emb(boards) + self.square_pos + elo_vec[:, None]
        extra = torch.stack([self.cls.expand(B, -1), self.side_emb(side),
                             self.elo_scale * white, self.elo_scale * black,
                             self.castle_emb(castle), self.ep_emb(ep)], dim=1)
        x = torch.cat([x, extra], dim=1)
        bias = self.bias(boards)
        for block in self.blocks:
            x = block(x, bias)
        return self.head(self.norm(x[:, 64]))


def main():
    vocab = json.load(open('vocab.json'))
    state = torch.load('chess_model_best.pt', map_location='cpu')
    state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
    model = ChessModel(num_moves=len(vocab))
    model.load_state_dict(state)
    model.eval()

    boards = torch.zeros(1, 64, dtype=torch.int64)
    side = torch.tensor([0 + 2 * 15 + 32 * 8], dtype=torch.int64)
    elo = torch.full((1, 2), 1500, dtype=torch.int64)
    torch.onnx.export(
        model, (boards, side, elo), 'docs/model.onnx',
        input_names=['boards', 'side', 'elo'], output_names=['logits'],
        dynamic_axes={'boards': {0: 'B'}, 'side': {0: 'B'}, 'elo': {0: 'B'}, 'logits': {0: 'B'}},
        opset_version=17, dynamo=False,
    )

    if '--quantize' in sys.argv:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic('docs/model.onnx', 'docs/model.int8.onnx', weight_type=QuantType.QInt8)

    json.dump(vocab, open('docs/vocab.json', 'w'))


if __name__ == '__main__':
    main()
