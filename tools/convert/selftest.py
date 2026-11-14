"""Golden self-test set: generation (fp32 torch / HF, CPU) and the reference runner (Core ML, CPU+ANE).

    python selftest.py run <dist_dir> [--functions a,b,...]     # what the runtime does at load (exit 1 on failure)

Every test builds its inputs deterministically from data in selftest.json (token ids, a procedural image) plus
the shipped embeddings.bf16 / position_embeddings.f16, runs ONE function, and compares with fp32 expectations
in selftest.bin (little-endian float32, offsets in selftest.json). Inputs:

  bucket embed_S     sequences (token id lists incl. <bos>/<eos>) packed with host.pack_bucket
                     -> compare each embedding row (cosine) and require unused rows ~0
  long_swa_l{i}      document = first 1800 rows of `doc_ids`, chunk b=1 (rows 1024..2047; 776 valid)
                     hidden_window = padded document rows 512..2559, inputs_embeds = rows 1024..2047,
                     bias = host.chunk_bias(1, 1800)
  long{S}_glb_l{i}   document = first S-300 rows of `doc_ids`; hidden = inputs_embeds = padded document,
                     key_bias for n = S-300
  long{S}_head       as glb, plus pool = host.long_pool(n, S) (2^-10 on valid rows) -> compare the embedding (cosine)
  vision_{B}         procedural image (see pattern()) of gx x gy patches -> compare image tokens

Layer tests compare a digest of the layer's residual update: D = (hidden_out - hidden_in)[valid query rows] @ R,
R = digest matrix [512,8] from selftest.bin. Vision compares D = image_tokens[valid tokens] @ R.
Metrics: cosine of flattened D and relative error |Da - Dr|_F / |Dr|_F.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import host  # noqa: E402

DOC_N_SWA = 1800
VISION_GRIDS = {70: (27, 18), 140: (36, 30), 280: (48, 45)}   # patch grid (cols, rows): 54 / 120 / 240 tokens
DIGEST_DIMS = 8


def pattern(gx, gy, budget):
    """Procedural image as Gemma4ImageProcessor-style outputs: pixel_values [9B,768] (row-major patches, each
    [16 rows][16 cols][RGB]), position ids [9B,2] (x, y; -1 padding). Integer formula, exact in any language:
    value(X, Y, c) = ((5X + 3Y + 70c + 90 * ((X // 24 + Y // 24) % 2)) % 256) / 255, X/Y = pixel column/row."""
    P = 9 * budget
    pv = np.zeros((P, 768), np.float32)
    pos = np.full((P, 2), -1, np.int64)
    r, c, ch = np.meshgrid(np.arange(16), np.arange(16), np.arange(3), indexing="ij")
    for py in range(gy):
        for px in range(gx):
            p = py * gx + px
            X, Y = px * 16 + c, py * 16 + r
            v = (5 * X + 3 * Y + 70 * ch + 90 * ((X // 24 + Y // 24) % 2)) % 256
            pv[p] = (v.astype(np.float32) / 255.0).reshape(-1)
            pos[p] = (px, py)
    return pv, pos


def bucket_sequences(doc, S):
    lens = {64: (40, 11, 9), 128: (80, 30, 12)}.get(S, (S * 5 // 8, 37, S // 8 + 3))
    seqs, off = [], 0
    for n in lens:
        seqs.append([host.BOS] + doc[off:off + n - 2] + [host.EOS])
        off += n
    return seqs


# ------------------------------------------------------------------------------------------- generation

def generate(hf_dir, out_dir):
    import torch
    import build
    from eg2 import GlobalHead, GlobalLayer, SwaLayer, Bucket, Weights, ref_embed
    torch.set_num_threads(8)
    W = Weights(hf_dir)
    tab = host.TokenTable(f"{out_dir}/embeddings.bf16")
    tok = host.Tokenizer(f"{out_dir}/tokenizer.json")
    text = open(f"{out_dir}/LICENSE").read()
    doc = []
    while len(doc) < 8192:
        doc += tok.ids(text, prefix="")[1:-1]
    doc = doc[:8192]
    rng = np.random.default_rng(20261114)
    R = rng.standard_normal((512, DIGEST_DIMS)).astype(np.float32) / np.sqrt(512)
    blob = bytearray()

    def put(a):
        a = np.ascontiguousarray(a, "<f4")
        off = len(blob)
        blob.extend(a.tobytes())
        return {"offset": off, "shape": list(a.shape)}

    tests = []
    spec = {s[0]: s for s in build.function_specs()}

    def run_torch(mod, feed):
        with torch.no_grad():
            return mod(*[torch.from_numpy(np.asarray(v, np.float32)) for v in feed.values()]).numpy()

    for S in host.BUCKETS:
        seqs = bucket_sequences(doc, S)
        exp = np.stack([ref_embed(W, tab.rows(s)) for s in seqs])
        # also record what the fp32 bucket graph itself gives (sanity: equals the reference)
        got = run_torch(Bucket(W, S).eval(), host.pack_bucket([tab.rows(s) for s in seqs], S))
        assert min(host.cos(got[k], exp[k]) for k in range(len(seqs))) > 0.9999
        tests.append({"function": f"embed_{S}", "kind": "bucket", "sequences": seqs, "expected": put(exp)})
        print(f"[selftest] embed_{S} lens={[len(s) for s in seqs]}", flush=True)
    for name, (fn, kind, args, ins, outs) in spec.items():
        if kind == "swa":
            feed, qin, nvalid = swa_feed(tab, doc)
            y = run_torch(SwaLayer(W, args[0]).eval(), feed)
            d = ((y - qin.astype(np.float32))[0, :nvalid]) @ R
            tests.append({"function": fn, "kind": kind, "doc_rows": DOC_N_SWA, "chunk": 1, "valid_rows": nvalid,
                          "expected": put(d)})
        elif kind in ("glb", "head"):
            S = args[-1]
            feed, n = glb_feed(tab, doc, S, kind == "head")
            if kind == "glb":
                y = run_torch(GlobalLayer(W, args[0], S).eval(), feed)
                d = ((y - feed["hidden"].astype(np.float32))[0, :n]) @ R
                tests.append({"function": fn, "kind": kind, "doc_rows": n, "valid_rows": n, "expected": put(d)})
            else:
                e = run_torch(GlobalHead(W, S).eval(), feed)
                tests.append({"function": fn, "kind": kind, "doc_rows": n, "expected": put(e)})
        else:
            continue
        print(f"[selftest] {fn}", flush=True)
    # vision: fp32 HF vision tower + embedder on the procedural image
    from transformers import AutoModel
    hfm = AutoModel.from_pretrained(hf_dir, dtype=torch.float32).eval()
    for B, (gx, gy) in VISION_GRIDS.items():
        pv, pos = pattern(gx, gy, B)
        with torch.no_grad():
            f = hfm.get_image_features(pixel_values=torch.from_numpy(pv)[None],
                                       image_position_ids=torch.from_numpy(pos)[None]).pooler_output[0].numpy()
        ntok = gx * gy // 9
        assert f.shape == (ntok, 512), f.shape
        tests.append({"function": f"vision_{B}", "kind": "vision", "package": "EmbeddingGemma2Vision",
                      "grid": [gx, gy], "budget": B, "valid_tokens": ntok, "expected": put(f @ R),
                      "expected_full_l2": float(np.linalg.norm(f))})
        print(f"[selftest] vision_{B} tokens={ntok}", flush=True)
    meta = {
        "format": "aneembed-selftest/1",
        "bin": "selftest.bin", "dtype": "float32", "byte_order": "little",
        "digest": put(R),
        "doc_ids_source": "LICENSE (Apache License 2.0 text shipped with the package), tokenized without prefix, "
                          "<bos>/<eos> stripped, repeated, first 8192 ids",
        "doc_ids": doc,
        "tolerance": TOLERANCE,
        "procedure": __doc__.strip().split("\n\n", 1)[1],
        "tests": tests,
    }
    open(f"{out_dir}/selftest.bin", "wb").write(bytes(blob))
    json.dump(meta, open(f"{out_dir}/selftest.json", "w"), indent=1)
    print(f"[selftest] {len(tests)} tests, selftest.bin {len(blob)} bytes", flush=True)


TOLERANCE = {   # filled from measurements (see README); min cosine / max relative error
    "bucket": {"min_cos": 0.9990, "zero_rows_max_abs": 1e-3},
    "head": {"min_cos": 0.9990},
    "swa": {"min_cos": 0.9990, "max_rel_err": 0.05},
    "glb": {"min_cos": 0.9990, "max_rel_err": 0.05},
    "vision": {"min_cos": 0.9990, "max_rel_err": 0.05},
}


def swa_feed(tab, doc):
    n, b, C, H = DOC_N_SWA, 1, host.CHUNK, host.HALO
    x = np.zeros((1, 2 * C + 2 * H, 512), np.float16)       # padded document (H zero rows in front)
    x[0, H:H + n] = tab.rows(doc[:n])
    feed = {"hidden_window": x[:, b * C:b * C + C + 2 * H], "inputs_embeds": x[:, H + b * C:H + (b + 1) * C],
            "bias": host.chunk_bias(b, n)}
    return feed, feed["hidden_window"][:, H:H + C], min(C, n - b * C)


def glb_feed(tab, doc, S, with_pool):
    n = S - 300
    x = np.zeros((1, S, 512), np.float16)
    x[0, :n] = tab.rows(doc[:n])
    feed = {"hidden": x, "inputs_embeds": x, "key_bias": np.where(np.arange(S) < n, 0.0, host.NEG).astype(np.float16).reshape(1, 1, 1, S)}
    if with_pool:
        feed["pool"] = host.long_pool(n, S)
    return feed, n


# ------------------------------------------------------------------------------------------- runner

def metrics(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return host.cos(a, b), float(np.linalg.norm(a - b) / np.linalg.norm(b))


def run(dist, only=None, compute_units="ane", verbose=True):
    import coremltools as ct
    meta = json.load(open(f"{dist}/selftest.json"))
    blob = open(f"{dist}/selftest.bin", "rb").read()

    def get(e):
        return np.frombuffer(blob, "<f4", int(np.prod(e["shape"])), e["offset"]).reshape(e["shape"])

    R = get(meta["digest"])
    tab = host.TokenTable(f"{dist}/embeddings.bf16")
    vis = host.Vision(f"{dist}/position_embeddings.f16")
    doc = meta["doc_ids"]
    tol = meta["tolerance"]
    cu = ct.ComputeUnit.CPU_ONLY if compute_units == "cpu" else ct.ComputeUnit.CPU_AND_NE
    results, ok_all = [], True
    for t in meta["tests"]:
        fn, kind = t["function"], t["kind"]
        if only and fn not in only:
            continue
        pkg = t.get("package", "EmbeddingGemma2Text")
        m = ct.models.CompiledMLModel(f"{dist}/{pkg}.mlmodelc", compute_units=cu, function_name=fn)
        exp = get(t["expected"])
        r = {"function": fn}
        if kind == "bucket":
            S = int(fn.split("_")[1])
            out = m.predict(host.pack_bucket([tab.rows(s) for s in t["sequences"]], S))["embedding"]
            k = len(t["sequences"])
            r["min_cos"] = min(host.cos(out[i], exp[i]) for i in range(k))
            r["zero_rows_max_abs"] = float(np.abs(out[k:]).max()) if k < len(out) else 0.0
            ok = r["min_cos"] >= tol[kind]["min_cos"] and r["zero_rows_max_abs"] <= tol[kind]["zero_rows_max_abs"]
        elif kind == "swa":
            feed, qin, nv = swa_feed(tab, doc)
            y = m.predict(feed)["hidden_out"]
            r["min_cos"], r["rel_err"] = metrics(((y.astype(np.float32) - qin.astype(np.float32))[0, :nv]) @ R, exp)
            ok = r["min_cos"] >= tol[kind]["min_cos"] and r["rel_err"] <= tol[kind]["max_rel_err"]
        elif kind == "glb":
            S = int(fn.split("_")[0][4:])
            feed, n = glb_feed(tab, doc, S, False)
            y = m.predict(feed)["hidden_out"]
            r["min_cos"], r["rel_err"] = metrics(((y.astype(np.float32) - feed["hidden"].astype(np.float32))[0, :n]) @ R, exp)
            ok = r["min_cos"] >= tol[kind]["min_cos"] and r["rel_err"] <= tol[kind]["max_rel_err"]
        elif kind == "head":
            S = int(fn.split("_")[0][4:])
            feed, n = glb_feed(tab, doc, S, True)
            e = m.predict(feed)["embedding"]
            r["min_cos"] = host.cos(e, exp)
            ok = r["min_cos"] >= tol[kind]["min_cos"]
        else:
            B = t["budget"]
            pv, pos = pattern(*t["grid"], B)
            feed, ntok = vis.feed(pv, pos, B)
            y = m.predict(feed)["image_tokens"][0, :ntok].astype(np.float32)
            r["min_cos"], r["rel_err"] = metrics(y @ R, exp)
            ok = r["min_cos"] >= tol[kind]["min_cos"] and r["rel_err"] <= tol[kind]["max_rel_err"]
        r["pass"] = bool(ok)
        ok_all &= ok
        results.append(r)
        if verbose:
            print(json.dumps(r), flush=True)
        del m
    return ok_all, results


if __name__ == "__main__" and sys.argv[1] == "run":
    only = None
    cu = "ane"
    for a in sys.argv[3:]:
        if a.startswith("--functions="):
            only = set(a.split("=", 1)[1].split(","))
        if a == "--cpu":
            cu = "cpu"
    ok, res = run(sys.argv[2], only, cu)
    print(f"SELFTEST {'PASS' if ok else 'FAIL'} {sum(r['pass'] for r in res)}/{len(res)} "
          f"min_cos={min(r['min_cos'] for r in res):.6f} max_rel_err={max(r.get('rel_err', 0) for r in res):.4f}")
    sys.exit(0 if ok else 1)
