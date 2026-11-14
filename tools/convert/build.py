#!/usr/bin/env python3
"""Build the ANEEmbed EmbeddingGemma 2 Core ML release package from the original HF weights.

    python -I build.py --hf-dir <google/embeddinggemma-2 snapshot> --vision-dir <FluidInference package dir> --out <dist>

fetch.py models downloads both inputs at the pinned revisions (see sources.json and README.md).

Stages (default: all, in order; each is idempotent and skips finished work unless --force):
  convert   one single-function .mlpackage per function into <work>/fn/  (fp16 ML Program, macOS 15):
            embed_{64..1280} (full attention), winp{1536,2048}_b512 (windowed, packing),
            winl{3072,4096,6144,8192}_b512 (windowed, one long text per call)
  merge     multifunction merge -> <out>/EmbeddingGemma2Text.mlpackage (coremltools dedups identical weights)
  compile   xcrun coremlcompiler -> <out>/EmbeddingGemma2Text.mlmodelc
  assets    embeddings.bf16 (from model.safetensors), tokenizer.json, vision package + position table (copied
            unchanged from the published FluidInference conversion), LICENSE/NOTICE
  selftest  selftest.json + selftest.bin (fp32 torch expectations; see selftest.py)
  manifest  manifest.json (file sha256/sizes, signatures, host contract, token ids, provenance)

No step uses the GPU; conversion and fp32 references run on the CPU. Requirements: requirements.txt.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
EXPECTED_SAFETENSORS_SHA256 = "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"
TEXT_PKG = "EmbeddingGemma2Text"
VISION_PKG = "EmbeddingGemma2Vision"
# FluidInference/embeddinggemma-2-coreml @ 3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4 (HF LFS sha256)
VISION_SHA256 = {
    f"{VISION_PKG}.mlpackage/Data/com.apple.CoreML/weights/weight.bin": "5b5b6dd8d80aa438bf91ba143fbe90d56a2e25bfcf685a56b990aad71cf75232",
    f"{VISION_PKG}.mlpackage/Data/com.apple.CoreML/model.mlmodel": "0085b5e04a13d000fe5969adeaa13d897f5a3cb7be60c76c8685c023747f89ef",
    "position_embeddings.f16": "17de9480acc23638c0634a97ed89e51ed664433256d7d6f109587dcd91c79124",
}


def sha256(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hf_revision(hf_dir):
    """commit sha recorded by `hf download --local-dir` (.cache/huggingface/download/*.metadata, first line)"""
    p = os.path.join(hf_dir, ".cache/huggingface/download/model.safetensors.metadata")
    if os.path.exists(p):
        lines = open(p).read().split()
        return lines[0], lines[1]
    return None, None


def function_specs():
    """(name, kind, args, inputs [(name, shape)], outputs [(name, dtype)])"""
    from eg2 import BAND, BUCKETS, WINDOW_LONG, WINDOW_PACK, k_for
    from host import function_name
    specs = []
    for S in BUCKETS:
        specs.append((function_name(S), "bucket", (S,),
                      [("inputs_embeds", (1, S, 512)), ("global_bias", (1, 1, S, S)), ("local_bias", (1, 1, S, S)),
                       ("pool", (k_for(S), S))], [("embedding", "fp32")]))
    for S in WINDOW_PACK:
        specs.append((function_name(S), "winp", (S,),
                      [("inputs_embeds", (1, S, 512)), ("band_bias", (1, 1, S, BAND)), ("global_bias", (1, 1, S, S)),
                       ("pool", (k_for(S), S))], [("embedding", "fp32")]))
    for S in WINDOW_LONG:
        specs.append((function_name(S, long=True), "winl", (S,),
                      [("inputs_embeds", (1, S, 512)), ("band_bias", (1, 1, S, BAND)), ("key_bias", (1, 1, 1, S)),
                       ("pool", (1, S))], [("embedding", "fp32")]))
    return specs


def make_module(W, kind, args):
    from eg2 import Bucket, Window
    return {"bucket": Bucket, "winp": Window, "winl": Window}[kind](W, *args).eval()


def convert_one(W, spec, path):
    import coremltools as ct
    import numpy as np
    import torch
    name, kind, args, ins, outs = spec
    mod = make_module(W, kind, args)
    ex = tuple(torch.zeros(*s) for _, s in ins)
    with torch.no_grad():
        tr = torch.jit.trace(mod, ex, check_trace=False)
    dt = {"fp32": np.float32, "fp16": np.float16}
    ml = ct.convert(tr, inputs=[ct.TensorType(name=n, shape=s, dtype=np.float16) for n, s in ins],
                    outputs=[ct.TensorType(name=n, dtype=dt[d]) for n, d in outs],
                    convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=ct.target.macOS15, compute_units=ct.ComputeUnit.CPU_AND_NE,
                    skip_model_load=True)
    ml.save(path)


def stage_convert(a):
    import torch
    from eg2 import Weights
    torch.set_num_threads(8)
    os.makedirs(f"{a.work}/fn", exist_ok=True)
    W = Weights(a.hf_dir)
    for spec in function_specs():
        p = f"{a.work}/fn/{spec[0]}.mlpackage"
        if os.path.exists(p) and not a.force:
            continue
        t = time.time()
        tmp = f"{a.work}/fn/_tmp_{spec[0]}.mlpackage"
        convert_one(W, spec, tmp)
        if os.path.exists(p):
            shutil.rmtree(p)
        os.replace(tmp, p)
        print(f"[convert] {spec[0]} {time.time() - t:.1f}s", flush=True)


def stage_merge(a):
    import coremltools as ct
    out = f"{a.out}/{TEXT_PKG}.mlpackage"
    if os.path.exists(out) and not a.force:
        return
    t = time.time()
    desc = ct.utils.MultiFunctionDescriptor()
    for spec in function_specs():
        desc.add_function(f"{a.work}/fn/{spec[0]}.mlpackage", src_function_name="main", target_function_name=spec[0])
    desc.default_function_name = "embed_512"
    tmp = f"{a.out}/_tmp_{TEXT_PKG}.mlpackage"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    ct.utils.save_multifunction(desc, tmp)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.replace(tmp, out)
    print(f"[merge] {out} {time.time() - t:.1f}s", flush=True)


def stage_compile(a):
    src = f"{a.out}/{TEXT_PKG}.mlpackage"
    dst = f"{a.out}/{TEXT_PKG}.mlmodelc"
    if os.path.exists(dst) and not a.force:
        return
    if os.path.exists(dst):
        shutil.rmtree(dst)
    t = time.time()
    subprocess.run(["xcrun", "coremlcompiler", "compile", src, a.out], check=True, capture_output=True)
    print(f"[compile] {dst} {time.time() - t:.1f}s", flush=True)


def write_apache_license(path):
    """Apache License 2.0 full text (from the transformers wheel, which is Apache-2.0), header line stripped.
    selftest.py tokenizes this file, so its bytes are part of the self-test expectations."""
    import importlib.metadata as md
    src = next(f for f in md.files("transformers") if f.name == "LICENSE")
    text = src.read_text()
    text = text[text.index("Apache License"):]
    with open(path, "w") as f:
        f.write("                                 " + text)


def stage_assets(a):
    import numpy as np
    from safetensors import safe_open
    emb = f"{a.out}/embeddings.bf16"
    if not os.path.exists(emb) or a.force:
        t = safe_open(f"{a.hf_dir}/model.safetensors", "pt").get_tensor("language_model.embed_tokens.weight")
        import torch
        assert t.dtype == torch.bfloat16 and tuple(t.shape) == (262144, 512)
        t.contiguous().view(torch.int16).numpy().astype("<i2").tofile(emb + ".tmp")
        os.replace(emb + ".tmp", emb)
    shutil.copy2(f"{a.hf_dir}/tokenizer.json", f"{a.out}/tokenizer.json")
    # vision: the published FluidInference package, unchanged (sha256 checked), compiled here
    vp = f"{a.out}/{VISION_PKG}.mlpackage"
    if not os.path.exists(vp) or a.force:
        if os.path.exists(vp):
            shutil.rmtree(vp)
        shutil.copytree(os.path.join(a.vision_dir, f"{VISION_PKG}.mlpackage"), vp)
        shutil.copy2(os.path.join(a.vision_dir, "position_embeddings.f16"), f"{a.out}/position_embeddings.f16")
        for rel, want in VISION_SHA256.items():
            got = sha256(os.path.join(a.out, rel))
            assert got == want, f"{rel}: sha256 {got} != published {want}"
        if os.path.exists(f"{a.out}/{VISION_PKG}.mlmodelc"):
            shutil.rmtree(f"{a.out}/{VISION_PKG}.mlmodelc")
        subprocess.run(["xcrun", "coremlcompiler", "compile", vp, a.out], check=True, capture_output=True)
    write_apache_license(f"{a.out}/LICENSE")
    shutil.copy2(os.path.join(HERE, "NOTICE"), f"{a.out}/NOTICE")
    print("[assets] done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True, help=f"snapshot of {MODEL_ID}@{MODEL_REVISION[:7]} (fetch.py models)")
    ap.add_argument("--vision-dir", required=True, help="snapshot of FluidInference/embeddinggemma-2-coreml@3f978dd "
                    "(EmbeddingGemma2Vision.mlpackage, position_embeddings.f16; fetch.py models)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--stages", default="convert,merge,compile,assets,selftest,manifest")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    a.work = a.work or os.path.join(os.path.dirname(os.path.abspath(a.out)), "work")
    os.makedirs(a.out, exist_ok=True)
    rev, etag = hf_revision(a.hf_dir)
    if rev is None:
        # no download metadata (e.g. a copied snapshot): the sha256 check below pins the weights instead
        rev = MODEL_REVISION
    elif rev != MODEL_REVISION:
        print(f"[source] warning: snapshot revision {rev} is not the pinned {MODEL_REVISION}", flush=True)
    print(f"[source] {MODEL_ID} revision {rev}", flush=True)
    stages = a.stages.split(",")
    if "convert" in stages:
        got = sha256(f"{a.hf_dir}/model.safetensors")
        assert got == EXPECTED_SAFETENSORS_SHA256, f"model.safetensors sha256 {got} != {EXPECTED_SAFETENSORS_SHA256}"
        stage_convert(a)
    if "merge" in stages:
        stage_merge(a)
    if "compile" in stages:
        stage_compile(a)
    if "assets" in stages:
        stage_assets(a)
    if "selftest" in stages:
        import selftest
        selftest.generate(a.hf_dir, a.out)
    if "manifest" in stages:
        import manifest
        manifest.write(a, rev)


if __name__ == "__main__":
    main()
