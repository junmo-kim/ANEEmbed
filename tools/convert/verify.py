"""Numerical verification of the shipped package on CPU+ANE vs the fp32 torch reference (eg2.Reference, which
equals HF sentence-transformers at cos 1.000000, see verify_hf.py). Inputs are built only from shipped artifacts
through host.py (the documented host contract).

Usage: verify.py text <dist> <corpus.json> --hf-dir <snapshot>    buckets: single full/half, packed short, mixed
       verify.py long <dist> <corpus.json> --hf-dir <snapshot>    long path 2049..8192 via host.LongPath
corpus.json comes from corpus.py, the snapshot from fetch.py models.
Prints one line per case and a final SUMMARY json {category: {n, min, median}}."""
import argparse
import json
import os
import sys
import time

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import host  # noqa: E402
from eg2 import Weights, ref_embed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("text", "long"))
ap.add_argument("dist")
ap.add_argument("corpus")
ap.add_argument("--hf-dir", required=True, help="google/embeddinggemma-2 snapshot (fp32 reference weights)")
args = ap.parse_args()
torch.set_num_threads(8)
mode, dist, corpus = args.mode, args.dist, args.corpus
W = Weights(args.hf_dir)
tab = host.TokenTable(f"{dist}/embeddings.bf16")
tok = host.Tokenizer(f"{dist}/tokenizer.json")
docs = json.load(open(corpus))
ids = [tok.ids(d["text"]) for d in docs]
PKG = f"{dist}/EmbeddingGemma2Text.mlmodelc"
CU = ct.ComputeUnit.CPU_AND_NE
res = {}


def fn(name):
    return ct.models.CompiledMLModel(PKG, compute_units=CU, function_name=name)


def text_rows(d, n, start=0):
    body = ids[d % len(ids)][1:-1]
    start = start % max(1, len(body) - n)
    seq = [host.BOS] + body[start:start + n - 2] + [host.EOS]
    assert len(seq) == n, (len(seq), n)
    return tab.rows(seq)


def add(cat, c):
    res.setdefault(cat, []).append(c)


def summary():
    s = {k: {"n": len(v), "min": round(min(v), 6), "median": round(float(np.median(v)), 6)} for k, v in res.items()}
    allv = [c for k, v in res.items() if not k.startswith("embed_") for c in v]
    s["ALL"] = {"n": len(allv), "min": round(min(allv), 6), "median": round(float(np.median(allv)), 6)}
    print("SUMMARY", json.dumps(s), flush=True)


if mode == "text":
    for S in host.BUCKETS:
        m = fn(f"embed_{S}")
        K = host.k_for(S)
        rng = np.random.default_rng(S)
        cases = []
        for d in (0, 1, 2, 3, 5, 6):                 # en, ko, ja, zh, ko-constitution, ja-constitution
            cases.append(("single_full", [(d, S)]))
            cases.append(("single_half", [(d, S // 2 + 5)]))
        for trial in range(3):
            lens, tot = [], 0
            while len(lens) < K:
                n = int(rng.integers(6, max(8, min(S // 4, 200))))
                if tot + n > S:
                    break
                lens.append(n)
                tot += n
            cases.append(("packed_short", [((trial * 5 + j) % len(ids), n) for j, n in enumerate(lens)]))
        if S >= 1024:
            big = 600 if S == 1024 else 1500
            cases.append(("mixed_long_short", [(1, big), (4, 120), (11, 64), (6, 200), (8, 31), (2, 7)]))
            cases.append(("two_long", [(5, S // 2 - 40), (9, S // 2 - 30)]))
        for cat, spec in cases:
            rows = [text_rows(d, n, start=53 * d + n) for d, n in spec]
            out = m.predict(host.pack_bucket(rows, S))["embedding"]
            cs = [host.cos(out[k], ref_embed(W, r)) for k, r in enumerate(rows)]
            zero = float(np.abs(out[len(rows):]).max()) if len(rows) < K else 0.0
            for c in cs:
                add(cat, c)
                add(f"embed_{S}", c)
            print(f"S={S} {cat} lens={[n for _, n in spec]} cos={[round(c, 6) for c in cs]} unused-max|x|={zero:.3g}", flush=True)
        del m
    summary()

if mode == "long":
    fns = {}
    t0 = time.time()
    for S in host.LONG_BUCKETS:
        for nm in host.LongPath.names(S):
            if nm not in fns:
                fns[nm] = fn(nm)
    print(f"loaded {len(fns)} long-path functions in {time.time() - t0:.1f}s", flush=True)
    lp = host.LongPath(fns)
    long_docs = [i for i, x in enumerate(ids) if len(x) >= 8192]
    lengths = [2049, 2600, 3300, 4096, 4097, 5000, 6100, 7000, 8192]
    j = 0
    for L in lengths:
        cand = [i for i, x in enumerate(ids) if len(x) >= L]
        for r in range(4 if L in (2049, 4096, 4097, 8192) else 3):
            d = cand[(j + r * 3) % len(cand)]
            seq = ids[d][:L - 1] + [host.EOS]
            rows = tab.rows(seq)
            tm = []
            e = lp.run(rows, timing=tm)
            c = host.cos(e, ref_embed(W, rows))
            band = "2049-4096" if L <= 4096 else "4097-8192"
            add(f"long_{band}", c)
            print(f"long n={L} doc={docs[d]['name']} cos={c:.6f} predict-sum={tm[0] * 1000:.0f} ms", flush=True)
        j += 1
    summary()
