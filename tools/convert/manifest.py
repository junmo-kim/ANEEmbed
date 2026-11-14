"""manifest.json for the release package (called by build.py --stages manifest)."""
import datetime
import hashlib
import importlib.metadata as md
import json
import os

import host

VISION_SOURCE = {"id": "FluidInference/embeddinggemma-2-coreml", "revision": "3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4",
                 "license": "Apache-2.0", "note": "EmbeddingGemma2Vision.mlpackage and position_embeddings.f16 copied unchanged "
                 "(sha256 equal to the HF LFS objects at this revision)"}
DTYPES = {65552: "float16", 65568: "float32", 131104: "int32"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def file_list(out):
    files = []
    for root, dirs, fs in os.walk(out):
        dirs.sort()
        for f in sorted(fs):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, out)
            if rel in ("manifest.json", "README.md") or f.startswith(".") or rel.startswith("_tmp_"):
                continue
            files.append({"path": rel, "size": os.path.getsize(p), "sha256": sha256(p)})
    return files


def signatures(pkg):
    import coremltools as ct
    spec = ct.utils.load_spec(pkg)
    out = {}
    for f in spec.description.functions:
        def io(v):
            t = v.type.multiArrayType
            return {"name": v.name, "shape": [int(s) for s in t.shape], "dtype": DTYPES.get(t.dataType, str(t.dataType))}
        out[f.name] = {"inputs": [io(v) for v in f.input], "outputs": [io(v) for v in f.output]}
    return out


def write(a, revision):
    out = a.out
    text_sigs = signatures(f"{out}/EmbeddingGemma2Text.mlpackage")
    vis_sigs = signatures(f"{out}/EmbeddingGemma2Vision.mlpackage")
    st = json.load(open(f"{out}/selftest.json")) if os.path.exists(f"{out}/selftest.json") else {}
    pkgs = {p: md.version(p) for p in ("coremltools", "torch", "numpy", "safetensors", "tokenizers", "transformers",
                                         "sentence-transformers", "pillow", "torchvision")}
    prompts = json.load(open(f"{a.hf_dir}/config_sentence_transformers.json"))["prompts"]
    m = {
        "name": "embeddinggemma-2-ane",
        "manifest_version": 1,
        "created": datetime.date.today().isoformat(),
        "source_model": {"id": "google/embeddinggemma-2", "revision": revision, "license": "Apache-2.0",
                         "model.safetensors_sha256": "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79",
                         "use_policy": "https://ai.google.dev/gemma/prohibited_use_policy",
                         "attribution": "EmbeddingGemma 2 by Google DeepMind"},
        "vision_source": VISION_SOURCE,
        "converter": {"script": "tools/convert/build.py", "python_packages": pkgs,
                      "precision": "fp16 ML Program (compute_precision FLOAT16), minimum macOS 15",
                      "compute_units": "cpuAndNeuralEngine (the package is tuned and verified for the ANE only)"},
        "license": "Apache-2.0 (LICENSE, NOTICE)",
        "embedding": {"dim": 768, "normalized": "L2, in graph", "matryoshka_dims": [768, 512, 256, 128],
                      "matryoshka": "keep the first d values, then L2-normalize again (host)"},
        "special_tokens": {"pad": host.PAD, "eos": host.EOS, "bos": host.BOS, "image_start (<|image>)": host.BOI,
                           "image_end (<image|>)": host.EOI, "image_placeholder (<|image|>, HF only)": host.IMAGE,
                           "audio_start": 256000, "audio_end": 258883},
        "text_sequence": "[bos] + tokenizer.encode(prompt + text, add_special_tokens=False) + [eos]",
        "prompts": prompts,
        "token_table": {"file": "embeddings.bf16", "dtype": "bfloat16", "byte_order": "little", "shape": [262144, 512],
                        "row_to_input": "fp32(bf16 row) * sqrt(512) (= 22.627417), then cast to fp16",
                        "sha256_equals": "FluidInference/embeddinggemma-2-coreml embeddings.bf16 (same bytes)"},
        "packages": {
            "EmbeddingGemma2Text": {"mlpackage": "EmbeddingGemma2Text.mlpackage", "mlmodelc": "EmbeddingGemma2Text.mlmodelc",
                                    "compute_units": "cpuAndNeuralEngine", "functions": text_sigs},
            "EmbeddingGemma2Vision": {"mlpackage": "EmbeddingGemma2Vision.mlpackage", "mlmodelc": "EmbeddingGemma2Vision.mlmodelc",
                                      "compute_units": "cpuAndNeuralEngine", "functions": vis_sigs},
        },
        "buckets": [{"function": f"embed_{S}", "tokens": S, "slots": host.k_for(S)} for S in host.BUCKETS],
        "bucket_contract": {
            "use": "texts/images of total length <= 2048 rows; pick the smallest S >= total rows of the packed sequences",
            "inputs_embeds": "[1,S,512] fp16: sequences back to back from row 0, zero rows after",
            "global_bias": "[1,1,S,S] fp16: 0 if row i and column j belong to the same sequence or i == j, else -10000",
            "local_bias": "[1,1,S,S] fp16: global_bias AND |i-j| <= 512 (else -10000); equals global_bias when S <= 512",
            "pool": "[K,S] fp16: row k = 1/n_k over the n_k rows of sequence k, 0 elsewhere; unused rows all 0 "
                    "(any positive per-row constant gives the same normalized output; 1/n_k is verified)",
            "embedding": "[K,768] fp32: row k = L2-normalized embedding of sequence k; unused rows are 0",
            "positions": "none: RoPE uses absolute positions 0..S-1 inside the graph (exact, RoPE is relative and the "
                         "bias is block diagonal)"},
        "long_path": {
            "tokens": [2049, 8192], "buckets": list(host.LONG_BUCKETS), "S": "4096 if n <= 4096 else 8192",
            "chunk": host.CHUNK, "halo": host.HALO, "sliding_window": host.WINDOW,
            "sliding_layers": host.SLIDING, "global_layers": host.GLOBAL,
            "inputs": {
                "x0": "[1,S,512] fp16 token rows (as for buckets), rows >= n zero",
                "chunk_bias(b)": "[1,1,1024,2048] fp16: query row q = b*1024 + r, key row k = b*1024 - 512 + c; 0 if "
                                 "(|q-k| <= 512 and 0 <= k < n) or k == q, else -10000",
                "key_bias": "[1,1,1,S] fp16: 0 for k < n, else -10000",
                "pool": "[1,S] fp16: 2^-10 (0.0009765625) for rows < n, else 0. Not 1/n: 1/n underflows fp16 in the "
                        "in-graph pooling for large n (cos 0.9996 instead of 0.9999 at n=8192); the scale cancels in the L2 "
                        "normalization"},
            "call_order": [
                "x = x0",
                "for layer i in 0..23:",
                "  sliding i: xp = [512 zero rows; x; 512 zero rows]; for each chunk b with b*1024 < n: "
                "out[b*1024:(b+1)*1024] = long_swa_l{i}(hidden_window=xp[b*1024 : b*1024+2048], "
                "inputs_embeds=x0[b*1024:(b+1)*1024], bias=chunk_bias(b)).hidden_out; chunks with b*1024 >= n keep "
                "their rows; x = out (read all chunks from the previous x: double buffer)",
                "  global i in {5,11,17}: x = long{S}_glb_l{i}(hidden=x, inputs_embeds=x0, key_bias).hidden_out",
                "  i = 23: embedding = long{S}_head(hidden=x, inputs_embeds=x0, key_bias, pool).embedding [1,768] fp32"],
            "calls": "20 * ceil(n/1024) + 4",
            "host_side_work": "token lookup and scale (same as buckets), halo padding / slicing, bias construction (per "
                              "document; reuse across the 20 sliding layers). No host math on activations: the final "
                              "norm, pooling, projection and L2 normalize run in long{S}_head."},
        "vision": {
            "package": "EmbeddingGemma2Vision", "functions": {f"vision_{b}": {"max_tokens": b, "patches": 9 * b} for b in (70, 140, 280)},
            "preprocessing": "HF Gemma4ImageProcessor: RGB, aspect-preserving resize to the largest H x W (multiples of 48 px) "
                             "with H*W <= 9*B*256 px, bicubic + antialias, rescale 1/255 (no normalize; the model applies 2x-1); "
                             "patches row-major [16 rows][16 cols][RGB] = 768 values; pad to 9*B patches",
            "inputs": {"patches": "[1,9B,768] fp16", "positions": "[1,9B,2] fp16 (x, y) patch coords, -1 for padding",
                       "position_embeddings": "[1,9B,768] fp16 = table[0][x] + table[1][y] from position_embeddings.f16 "
                                              "([2,1024,768] fp16), 0 for padding",
                       "valid": "[1,9B] fp16 1/0", "pool": "[B,9B] fp16: 1/9 at [(x//3) + (cols//3)*(y//3), p] for valid patch p"},
            "output": "image_tokens [1,B,512] fp16; the first (valid patches / 9) rows are the image tokens",
            "sequence": "rows = [bos*sqrt(512), boi*sqrt(512), image tokens (unscaled), eoi*sqrt(512), eos*sqrt(512)] -> "
                        "bucket embed_S like a text sequence (packable with texts and other images)",
            "default_budget": 280},
        "selftest": {"files": ["selftest.json", "selftest.bin"], "tests": len(st.get("tests", [])),
                     "tolerance": st.get("tolerance"), "runner": "tools/convert/selftest.py run <dir>"},
        "files": file_list(out),
    }
    json.dump(m, open(f"{out}/manifest.json", "w"), indent=1, ensure_ascii=False)
    print(f"[manifest] {len(m['files'])} files", flush=True)
