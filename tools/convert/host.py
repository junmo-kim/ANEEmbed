"""Reference implementation of the ANEEmbed host contract (pure numpy + Core ML; no torch).

This is the executable spec the Swift runtime mirrors. It only reads the shipped artifacts
(embeddings.bf16, position_embeddings.f16, tokenizer.json, the .mlmodelc packages).

Text, <= 2048 tokens:   pack sequences into the smallest bucket embed_S (see pack_bucket) -> embedding rows.
Text, 2049..8192:       long path, 20 sliding-layer calls per chunk + 4 global calls (see LongPath.run).
Image:                  HF Gemma4ImageProcessor pixels -> vision_{70,140,280} -> image tokens, wrapped as
                        <bos><|image> tokens <image|><eos> and embedded like a text sequence (packable).
"""
import json
import math
import os
import time

import numpy as np

NEG = -1e4
WINDOW = 512
CHUNK, HALO = 1024, 512
BUCKETS = (64, 128, 256, 512, 1024, 2048)
LONG_BUCKETS = (4096, 8192)
SCALE = math.sqrt(512)
BOS, EOS, PAD, BOI, EOI, IMAGE = 2, 1, 0, 255999, 258882, 258880
SLIDING = [i for i in range(24) if i % 6 != 5]
GLOBAL = [5, 11, 17, 23]
DOC_PREFIX = "title: none | text: "


def k_for(S):
    return max(8, S // 32)


class TokenTable:
    """embeddings.bf16: 262144 rows x 512 bf16 (little endian). Input row = bf16 -> fp32 * sqrt(512)."""

    def __init__(self, path):
        raw = np.memmap(path, dtype=np.uint16, mode="r").reshape(-1, 512)
        self.raw = raw

    def rows(self, ids):
        u = self.raw[np.asarray(ids, np.int64)].astype(np.uint32) << 16
        return u.view(np.float32) * np.float32(SCALE)          # fp32 [n,512]


class Tokenizer:
    def __init__(self, path):
        from tokenizers import Tokenizer as T
        self.t = T.from_file(path)

    def ids(self, text, prefix=DOC_PREFIX):
        return [BOS] + self.t.encode(prefix + text, add_special_tokens=False).ids + [EOS]


# ---------------------------------------------------------------- buckets (<= 2048 rows per call)

def bucket_for(total):
    return next(S for S in BUCKETS if S >= total)


def pack_bucket(rows, S):
    """rows: list of fp32 [n_k,512] sequences (sum n_k <= S, count <= K). Returns the embed_S feed (fp16).
    Sequences occupy consecutive rows from 0; RoPE is relative so restarting positions is unnecessary
    (the functions use absolute positions 0..S-1 internally)."""
    K = k_for(S)
    assert len(rows) <= K and sum(len(r) for r in rows) <= S
    x = np.zeros((1, S, 512), np.float32)
    seg = np.full(S, -1)
    pool = np.zeros((K, S), np.float32)
    off = 0
    for k, r in enumerate(rows):
        n = len(r)
        x[0, off:off + n] = r
        seg[off:off + n] = k
        pool[k, off:off + n] = 1.0 / n
        off += n
    i = np.arange(S)
    same = ((seg[:, None] == seg[None, :]) & (seg[None, :] >= 0)) | (i[:, None] == i[None, :])
    g = np.where(same, 0.0, NEG).astype(np.float16).reshape(1, 1, S, S)
    loc = np.where(same & (np.abs(i[:, None] - i[None, :]) <= WINDOW), 0.0, NEG).astype(np.float16).reshape(1, 1, S, S)
    return {"inputs_embeds": x.astype(np.float16), "global_bias": g, "local_bias": loc, "pool": pool.astype(np.float16)}


# ---------------------------------------------------------------- long path (2049..8192 rows)

def long_bucket(n):
    return next(S for S in LONG_BUCKETS if S >= n)


def chunk_bias(b, n):
    """[1,1,C,W] for chunk b: query rows b*C..b*C+C-1, key rows b*C-H .. b*C-H+W-1 (window-local)."""
    W = CHUNK + 2 * HALO
    qa = b * CHUNK + np.arange(CHUNK)[:, None]
    ka = b * CHUNK - HALO + np.arange(W)[None, :]
    ok = ((np.abs(qa - ka) <= WINDOW) & (ka >= 0) & (ka < n)) | (ka == qa)
    return np.where(ok, 0.0, NEG).astype(np.float16).reshape(1, 1, CHUNK, W)


LONG_POOL_WEIGHT = 1.0 / 1024


def long_pool(n, S):
    """[1,S] fp16: 2^-10 on the n valid rows. NOT 1/n: with 1/n (1.2e-4 at n=8192) the fp16 products in the
    in-graph pooling underflow and cos vs fp32 drops to ~0.9996; the output is L2-normalized, so the scale is free.
    2^-10 keeps the pooled sum <= 8x the mean (1/16 overflowed fp16 in tests)."""
    return np.where(np.arange(S) < n, LONG_POOL_WEIGHT, 0.0).astype(np.float16).reshape(1, S)


class LongPath:
    """fns: dict name -> object with .predict(dict) -> dict (Core ML function handles)."""

    def __init__(self, fns):
        self.fns = fns

    @staticmethod
    def names(S):
        return [f"long_swa_l{i}" for i in SLIDING] + [f"long{S}_glb_l{i}" for i in GLOBAL[:-1]] + [f"long{S}_head"]

    def run(self, rows, skip_pad_chunks=True, timing=None):
        n = len(rows)
        S = long_bucket(n)
        nb = S // CHUNK
        x0 = np.zeros((1, S, 512), np.float16)
        x0[0, :n] = rows
        biases = [chunk_bias(b, n) for b in range(nb)]
        key_bias = np.where(np.arange(S) < n, 0.0, NEG).astype(np.float16).reshape(1, 1, 1, S)
        pool = long_pool(n, S)
        live = [b for b in range(nb) if not (skip_pad_chunks and b * CHUNK >= n)]
        x = x0
        tp = 0.0
        for i in range(24):
            t = time.perf_counter()
            if i in SLIDING:
                xp = np.zeros((1, S + 2 * HALO, 512), np.float16)
                xp[:, HALO:HALO + S] = x
                out = x.copy()                     # skipped (all-padding) chunks keep their previous rows
                f = self.fns[f"long_swa_l{i}"]
                for b in live:
                    out[:, b * CHUNK:(b + 1) * CHUNK] = f.predict({
                        "hidden_window": xp[:, b * CHUNK:b * CHUNK + CHUNK + 2 * HALO],
                        "inputs_embeds": x0[:, b * CHUNK:(b + 1) * CHUNK],
                        "bias": biases[b]})["hidden_out"]
                x = out
            elif i != 23:
                x = self.fns[f"long{S}_glb_l{i}"].predict({"hidden": x, "inputs_embeds": x0, "key_bias": key_bias})[
                    "hidden_out"].astype(np.float16)
            else:
                e = self.fns[f"long{S}_head"].predict({"hidden": x, "inputs_embeds": x0, "key_bias": key_bias,
                                                       "pool": pool})["embedding"]
            tp += time.perf_counter() - t
        if timing is not None:
            timing.append(tp)
        return np.asarray(e[0], np.float32)


# ---------------------------------------------------------------- images

class Vision:
    """vision_{B} feed from HF Gemma4ImageProcessor outputs (pixel_values [P,768] in [0,1], position ids [P,2]
    as (x, y), -1 for padding, P = 9*B)."""

    def __init__(self, table_path):
        self.table = np.fromfile(table_path, np.float16).reshape(2, 1024, 768).astype(np.float32)

    def feed(self, pixel_values, pos, budget):
        P = budget * 9
        pv = np.asarray(pixel_values, np.float32).reshape(P, 768)
        pos = np.asarray(pos, np.int64).reshape(P, 2)
        valid = (pos[:, 0] >= 0)
        nv = int(valid.sum())
        cols = int(pos[valid, 0].max()) + 1
        pe = np.zeros((P, 768), np.float32)
        pe[valid] = self.table[0, pos[valid, 0]] + self.table[1, pos[valid, 1]]
        pool = np.zeros((budget, P), np.float32)
        for p in np.nonzero(valid)[0]:
            xx, yy = pos[p]
            pool[(xx // 3) + (cols // 3) * (yy // 3), p] = 1.0 / 9.0
        return {"patches": pv[None].astype(np.float16), "positions": pos[None].astype(np.float16),
                "position_embeddings": pe[None].astype(np.float16), "valid": valid[None].astype(np.float16),
                "pool": pool.astype(np.float16)}, nv // 9


def image_sequence(table, tokens):
    """<bos><|image> tokens <image|><eos> rows (special rows scaled, image tokens as produced)"""
    return np.concatenate([table.rows([BOS, BOI]), np.asarray(tokens, np.float32), table.rows([EOI, EOS])])


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
