# Model conversion and verification

These scripts build the model package published as
[junmo-k/embeddinggemma-2-ane](https://huggingface.co/junmo-k/embeddinggemma-2-ane) from public inputs
only, and check it against the original model in fp32. `aneembed` does not need them; they exist so that the
package can be rebuilt and audited.

| Script | What it does |
|---|---|
| `sources.json` | Every input, pinned: the two Hugging Face repositories (revision + sha256) and the evaluation texts and images (GitHub commit, Wikisource page revision or Python package version, path, license, sha256). |
| `fetch.py` | `models`: downloads `google/embeddinggemma-2` and the FluidInference vision package at the pinned revisions. `data`: downloads the evaluation files (and copies the ones that ship inside pinned packages). Checks every sha256. |
| `eg2.py` | The EmbeddingGemma 2 text encoder re-written in PyTorch for the ANE: bucket, sliding-layer, global-layer and head modules, plus `Reference`, an fp32 ground truth equal to Hugging Face. |
| `host.py` | Executable spec of the host contract (token rows, bucket packing, long path, vision feed) in numpy. The Swift runtime mirrors it. |
| `build.py` | Builds the package, stage by stage (below). |
| `selftest.py` | Generates `selftest.json` / `selftest.bin` (fp32 expectations) and runs them: `selftest.py run <dist>`. |
| `manifest.py` | Writes `manifest.json` (called by `build.py`). |
| `verify.py` | The package on CPU + ANE through `host.py` against `eg2.Reference` (fp32): buckets, packing, long path. |
| `verify_hf.py` | Anchors to Hugging Face: `eg2.Reference` against `transformers` / `sentence-transformers` fp32 (`anchor`), and the image path on CPU + ANE against `transformers` fp32 for all three image budgets (`images`). |
| `corpus.py`, `images.py` | Build the evaluation corpus (`corpus.json`, 12 documents) and image folder (13 images) from the fetched files. |
| `selftest_negative.py` | Optional negative control: rebuilds three global-layer functions with the usual `repeat_interleave` attention; their self-tests fail on the ANE (cosine about 0.55) while the layer tests pass on the CPU. |

`scripts/accuracy/` at the repository root is the other half: it measures a running `aneembed serve` (the Swift
runtime, over HTTP) against the same fp32 reference on the same public data. `verify*.py` here test the package
itself through the Python host contract.

## Setup

macOS 15 or later on Apple silicon, Xcode (for `xcrun coremlcompiler`), Python 3.12 and
[uv](https://docs.astral.sh/uv/).

```bash
cd tools/convert
uv venv --python 3.12.14 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

`requirements.txt` pins every package of the environment that built the published package. Conversion and the
fp32 references run on the CPU only: no step selects MPS or the GPU (torch has no separate CPU wheel for macOS;
the PyPI wheel is used). On an M5 Ultra a full build takes about 2.5 minutes and 11 GB of RAM (peak resident
size), and needs about 6 GB of disk: 1.7 GB of downloads, 2.0 GB of intermediate per-function packages, 1.5 GB of
output and 0.7 GB for the environment. The first `selftest.py run` on a new directory also waits for the ANE
compile (about 4 minutes for all functions). Run the scripts with `python -I` (they find their own modules;
nothing is read from the current directory).

## Reproducing the package

```bash
W=../../.cache                                    # any directory (.cache/ is git-ignored)
.venv/bin/python -I fetch.py models --dest $W/models     # 1.7 GB, sha256 checked
.venv/bin/python -I build.py --hf-dir $W/models/embeddinggemma-2 \
    --vision-dir $W/models/embeddinggemma-2-coreml --out $W/dist
.venv/bin/python -I selftest.py run $W/dist               # SELFTEST PASS 37/37 ...
```

Any directory works for `--dest` and `--out`; intermediate per-function packages go to `<out>/../work`
(`--work` to change). Stages, in order (`--stages` to run a subset; each skips finished work unless `--force`):

1. **convert**: checks the sha256 of `model.safetensors`, then traces each of the 34 functions (`embed_64` …
   `embed_2048`, `long_swa_l{i}`, `long{4096,8192}_glb_l{5,11,17}`, `long{4096,8192}_head`) and converts it with
   coremltools to an fp16 ML Program (minimum macOS 15), one `.mlpackage` per function.
2. **merge**: merges them into one multifunction `EmbeddingGemma2Text.mlpackage`; identical weight constants are
   stored once (305 MB instead of 2.05 GB).
3. **compile**: `xcrun coremlcompiler compile` to `EmbeddingGemma2Text.mlmodelc`.
4. **assets**: extracts `embeddings.bf16` (unchanged bf16 rows) from `model.safetensors`, copies `tokenizer.json`,
   copies the FluidInference vision package and `position_embeddings.f16` unchanged (sha256 checked) and compiles
   it, writes `LICENSE` (Apache-2.0 text from the pinned `transformers` wheel) and `NOTICE`.
5. **selftest**: one golden test per function (37), expectations from the fp32 reference (Hugging Face fp32 for the
   vision functions). The token ids are the tokenized `LICENSE`.
6. **manifest**: `manifest.json` with every file's sha256 and size, function signatures, host contract and
   provenance.

### What is byte-identical

With the pinned environment, a clean rebuild reproduces these files byte for byte (sha256 equal to the published
package):

- `EmbeddingGemma2Text.mlpackage/Data/com.apple.CoreML/weights/weight.bin` (= `.mlmodelc/weights/weight.bin`)
- `EmbeddingGemma2Text.mlmodelc/model.mil`, `coremldata.bin`, `metadata.json`
- `embeddings.bf16`, `tokenizer.json`, `selftest.json`, `selftest.bin`, `LICENSE`, `NOTICE`
- the vision package, which is copied unchanged

Three files differ between runs, none of them in the weights or the program:

- `EmbeddingGemma2Text.mlpackage/Manifest.json`: coremltools gives every package item a fresh random UUID.
- `EmbeddingGemma2Text.mlpackage/Data/com.apple.CoreML/model.mlmodel`: the merged protobuf is the same message
  (equal when parsed, and equal bytes with `SerializeToString(deterministic=True)`), but protobuf writes map
  fields in no fixed order.
- `EmbeddingGemma2Text.mlmodelc/analytics/coremldata.bin`: compiler analytics, not part of the program or the
  weights. The vision package, whose input bytes do not change, compiles to an identical one.

`manifest.json` differs only in those three hashes (and in its `created` date on another day). The compiled program depends
on the Core ML compiler: another `coremlcompiler` version changes the version stamp in `model.mil`.

## Verifying against the original model

```bash
.venv/bin/python -I fetch.py data --dest $W/sources
.venv/bin/python -I corpus.py $W/sources $W/data               # corpus.json, 12 documents
.venv/bin/python -I images.py $W/sources $W/data/images        # 13 images
HF=$W/models/embeddinggemma-2
.venv/bin/python -I verify_hf.py anchor $W/dist $W/data/corpus.json --hf-dir $HF   # reference == official fp32
.venv/bin/python -I verify_hf.py images $W/dist $W/data/images --hf-dir $HF        # image path vs official fp32
.venv/bin/python -I verify.py text $W/dist $W/data/corpus.json --hf-dir $HF        # buckets, packing
.venv/bin/python -I verify.py long $W/dist $W/data/corpus.json --hf-dir $HF        # 2049..8192 tokens
```

Each prints one line per case and a final `SUMMARY` with the minimum and median cosine per category. The corpus
mixes English, Korean, Japanese, Chinese and French READMEs, the Korean and Japanese constitutions, technical
documents and source code; the images are photos, diagrams, a plot and synthetic patterns of odd sizes. For the
Swift runtime, run `scripts/accuracy/reference.py` with the same `corpus.json` and image folder, then
`scripts/accuracy/compare.py` against `aneembed serve`.

## ANE caveats

The Neural Engine compiler runs on the user's Mac, per macOS build, and a miscompiled graph raises no error: it
returns wrong numbers. What we hit, and what the code does about it:

- **Attention uses the fold layout** (`eg2.attend_fold`): the query heads that share a K/V head are folded into
  the row dimension. The usual `repeat_interleave` layout returns zeros for query heads 1–3 of the global layers
  (one K/V head) at long sequence lengths; an explicit K/V copy is wrong as well. `selftest_negative.py`
  reproduces it. Do not change the attention layout without running the self-test on the ANE.
- **Masks are runtime inputs and RoPE tables are graph constants.** A sliding-window mask built into the graph
  and a runtime positions input both produced wrong results.
- **Do not trust `MLComputePlan` for correctness.** It reports which device each operation is planned on, not
  whether the ANE computes it correctly. Only comparing outputs catches a miscompile.
- **Run the self-test** (`selftest.py run <dist>`, or `aneembed`, which runs it at load) on every Mac and macOS
  version you ship to, with `cpuAndNeuralEngine`. `selftest.py run <dist> --cpu` runs the same tests on the CPU.
- Fixed shapes only: variable (enumerated) shapes move the whole graph to the CPU.

## License

The scripts are part of ANEEmbed and MIT licensed (see `LICENSE` at the repository root). The package they build
contains EmbeddingGemma 2 by Google (Apache-2.0, Gemma Prohibited Use Policy) and the vision encoder conversion by
FluidInference (Apache-2.0); `NOTICE` here is the notice file copied into the package.
