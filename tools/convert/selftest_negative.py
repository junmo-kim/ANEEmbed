"""Negative control for the self-test: convert global-layer functions with the repeat_interleave K/V layout (known
to miscompile on the ANE) and run their self-tests from selftest.json. Measured (M5 Ultra, macOS 27.0.1): all three
FAIL on CPU+ANE (cos 0.54-0.65); on the CPU the two layer tests PASS (cos 0.99994+) and long8192_head scores 0.998
(the fp16 repeat layout loses precision there too).
Usage: selftest_negative.py <hf_dir> <dist> <work_dir>"""
import json
import os
import subprocess
import sys

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build  # noqa: E402
import eg2  # noqa: E402
import host  # noqa: E402
import selftest  # noqa: E402


def attend_repeat(q, k, v, bias, kvh):
    g = 4 // kvh
    k = k.repeat_interleave(g, dim=1)
    v = v.repeat_interleave(g, dim=1)
    return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) + bias, dim=-1), v)


hf, dist, work = sys.argv[1:4]
os.makedirs(work, exist_ok=True)
eg2.attend_fold = attend_repeat
W = eg2.Weights(hf)
spec = {s[0]: s for s in build.function_specs()}
meta = json.load(open(f"{dist}/selftest.json"))
blob = open(f"{dist}/selftest.bin", "rb").read()
get = lambda e: np.frombuffer(blob, "<f4", int(np.prod(e["shape"])), e["offset"]).reshape(e["shape"])  # noqa: E731
R = get(meta["digest"])
tab = host.TokenTable(f"{dist}/embeddings.bf16")
tests = {t["function"]: t for t in meta["tests"]}
for name in ("long4096_glb_l5", "long8192_glb_l5", "long8192_head"):
    p = f"{work}/neg_{name}.mlpackage"
    build.convert_one(W, spec[name], p)
    subprocess.run(["xcrun", "coremlcompiler", "compile", p, work], check=True, capture_output=True)
    t = tests[name]
    S = int(name.split("_")[0][4:])
    feed, n = selftest.glb_feed(tab, meta["doc_ids"], S, t["kind"] == "head")
    for cu_name, cu in (("cpu", ct.ComputeUnit.CPU_ONLY), ("ane", ct.ComputeUnit.CPU_AND_NE)):
        m = ct.models.CompiledMLModel(f"{work}/neg_{name}.mlmodelc", compute_units=cu)
        out = m.predict(feed)
        if t["kind"] == "head":
            c = host.cos(out["embedding"], get(t["expected"]))
            print(f"NEG repeat-layout {name} {cu_name}: cos={c:.6f} pass={c >= meta['tolerance']['head']['min_cos']}", flush=True)
        else:
            d = ((out["hidden_out"].astype(np.float32) - feed["hidden"].astype(np.float32))[0, :n]) @ R
            c, e = selftest.metrics(d, get(t["expected"]))
            tol = meta["tolerance"]["glb"]
            print(f"NEG repeat-layout {name} {cu_name}: cos={c:.6f} rel_err={e:.4f} "
                  f"pass={c >= tol['min_cos'] and e <= tol['max_rel_err']}", flush=True)
        del m
