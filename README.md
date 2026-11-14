# ANEEmbed (`aneembed`)

[![CI](https://github.com/junmo-kim/ANEEmbed/actions/workflows/ci.yml/badge.svg)](https://github.com/junmo-kim/ANEEmbed/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Platform: macOS 15+ · Apple Silicon](https://img.shields.io/badge/platform-macOS%2015%2B%20%C2%B7%20Apple%20Silicon-lightgrey)

EmbeddingGemma 2 embeddings that run **only on the Apple Neural Engine**, as a single **100% Swift
binary with no third-party dependencies** (Foundation, Network, CoreML, ImageIO, Accelerate).

- **OpenAI drop-in**: `POST /v1/embeddings` with text and images, `encoding_format` float/base64,
  Matryoshka `dimensions` 768/512/256/128.
- **Leaves the GPU alone**: nothing runs on the GPU, so a local LLM keeps it to itself.
- **Accurate**: cosine ≥ 0.99989 (median 0.99997) to the **official** model (google/embeddinggemma-2 in fp32 through its official `sentence-transformers` stack) on every public
  test input, text up to 8192 tokens and images (see [Accuracy](#accuracy)).
- **Fast and small**: a short query in 5 ms and a 2048-token document in 0.25 s on an M4, 3–11x faster for
  text (2x for images) than the fp32 reference on the same Mac's CPU or GPU, whichever is faster; about
  160 MB resident at the default settings (see [Performance](#performance)).
- **Checks itself**: every model function is tested against golden outputs when it loads, and one that the
  Neural Engine compiler got wrong is not used (see [Self-test](#self-test)).
- **CLI, HTTP and MCP** from one binary: `aneembed embed`, `aneembed serve`, `aneembed mcp`.

Limits: text up to 8192 tokens (2048 by default, `--max-tokens` raises it), images at the model's official
280-token budget, macOS 15 or later on Apple silicon.

## Install

Apple Silicon, macOS 15 or later.

**Homebrew**

```bash
brew install junmo-kim/tap/aneembed
```

**mise**

```bash
mise use -g github:junmo-kim/ANEEmbed
```

**From source** (Xcode 16 or later)

```bash
swift build -c release && cp .build/release/aneembed /usr/local/bin/
```

Then install the model ([junmo-k/embeddinggemma-2-ane](https://huggingface.co/junmo-k/embeddinggemma-2-ane),
about 900 MB). This downloads a pinned revision, checks every file's SHA-256 against a hash compiled into
the binary, compiles the package on this Mac, then compiles each function for the Neural Engine once and
runs the self-test. The OS caches the Neural Engine compile, so later starts take seconds instead of minutes:

```bash
aneembed models install                  # about 2.5 minutes the first time (M4)
aneembed doctor        # Neural Engine, model files, compile cache, self-test: what is ready and what to fix
```

For texts over 2048 tokens set the limit for every command, e.g. `export ANEEMBED_MAX_TOKENS=8192`, and run
`aneembed models warm` once (the long-text functions: about 2 more minutes and 370 MB more memory, see
[trade-offs](#--max-tokens-trade-offs)).

`aneembed models install --from DIR` installs a local copy of the package instead (same checks, no
network). `aneembed models warm` compiles again in place, which an upgraded `aneembed` needs once: the OS
keeps the compile per model path, macOS build and executable (`serve` does it by itself before it reports
healthy, and `doctor` says when it is due).

The release binary is ad-hoc signed. A tarball downloaded with a browser carries a quarantine attribute;
clear it with `xattr -d com.apple.quarantine aneembed` (Homebrew, mise and curl do not set it).

## Usage

```bash
aneembed serve                                    # http://127.0.0.1:18180
curl -s localhost:18180/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": "task: search result | query: how do I reset my router?"}'

aneembed embed "first text" "second text" --format json
aneembed embed --prompt query "how do I reset my router?"    # adds the model's query prefix
aneembed embed --image photo.jpg --dimensions 256
cat docs.jsonl | aneembed embed --stdin --prompt document   # JSONL in, JSONL out, batched
aneembed similarity "a cat on a mat" --image-b cat.jpg
aneembed tokenize "hello world" --show-tokens
```

One-off commands load and self-test the model each time (4.5 s on an M4 with a warm cache, 27 s with
`--max-tokens 8192`). For regular use keep `serve` or `mcp` running.

### MCP

```json
{ "mcpServers": { "aneembed": { "command": "aneembed", "args": ["mcp"] } } }
```

Tools: `embed_text`, `embed_image`, `similarity`. The same tools are reachable over HTTP as
`POST /mcp` and as one-shot `QUERY /{tool}?<options>` with the raw text or image bytes as the body.

## API

| Endpoint | |
|---|---|
| `POST /v1/embeddings` | OpenAI-compatible. `input`: string, list of strings, or list of llama-server image items `{"content": [{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}]}` (texts and images are not mixed in one request) |
| `GET /v1/models` | the model id, `max_context_length`, `max_image_tokens`, `dimensions`, `image_preprocessing` |
| `GET /health` | `200` ready, `503` while loading (the first start compiles for the Neural Engine) or after a load failure |
| `POST /mcp` | MCP JSON-RPC, one message per request |
| `QUERY /{tool}` | `embed_text` (UTF-8 body), `embed_image` (image body), `similarity` (JSON body); YAML by default, `?format=json` |

Request fields of `/v1/embeddings`:

- `encoding_format`: `float` (default) or `base64` (little-endian float32).
- `dimensions`: 768 (default), 512, 256 or 128: the first N values, re-normalized.
- `input_type` (extension): `query` or `document` adds the model's task prefix
  (`task: search result | query: ` / `title: none | text: `). Without it texts are embedded as given,
  like llama-server, and callers add the prefix themselves.
- `model`, `user`: accepted and ignored; the response names the configured model.

Texts over the configured limit are rejected with `400 context_length_exceeded`, never truncated.
`usage.prompt_tokens` counts real tokens including `<bos>`/`<eos>`; an image counts its soft tokens + 4.
Errors use OpenAI's shape: `{"error": {"message", "type", "param", "code"}}`.

## Configuration

Every option is a flag and an environment variable of the same name (flag > environment > default).
`aneembed serve --print-config` shows each value and where it came from.

| Flag | Environment | Default | |
|---|---|---|---|
| `--model-dir` | `ANEEMBED_MODEL_DIR` | `<data-dir>/models/current` | model package directory |
| `--data-dir` | `ANEEMBED_DATA_DIR` | `~/Library/Application Support/ANEEmbed` | installed models and state |
| `--host` | `ANEEMBED_HOST` | `127.0.0.1` | anything else exposes an unauthenticated API |
| `--port` | `ANEEMBED_PORT` | `18180` | |
| `--max-tokens` | `ANEEMBED_MAX_TOKENS` | `2048` | up to 8192; only functions up to it are loaded and compiled |
| `--model-name` | `ANEEMBED_MODEL_NAME` | `embeddinggemma-2` | |
| `--max-body` | `ANEEMBED_MAX_BODY` | `72` (MiB) | 8 images of 8 MiB plus JSON |
| `--max-inputs` / `--max-images` | `ANEEMBED_MAX_INPUTS` / `ANEEMBED_MAX_IMAGES` | `2048` / `8` | per request |
| `--query-concurrency` / `--image-concurrency` / `--long-concurrency` | `ANEEMBED_*_CONCURRENCY` | `8` / `2` / `2` | separate lanes so long documents and images cannot hold short queries back |
| `--runtime` | `ANEEMBED_RUNTIME` | `coreml` | `fake` runs without a model, for testing integrations |

## Performance

`aneembed bench` measures this binary in process on macOS 27.0.1 (requests go through the same router as
`serve`; only the loopback round trip is missing): warm median of 10 runs (3 above 2048 tokens), texts of exactly N tokens
including `<bos>`/`<eos>` and the document prompt. Memory is `phys_footprint` after the run and 150 s of
idling (freed pages are reclaimed late); the Neural Engine's own memory is not part of it.

| | M4 (16 GB) | M5 Ultra | Official fp32, M4 CPU | Official fp32, M4 GPU (MPS) |
|---|---:|---:|---:|---:|
| 32 tokens | 5.1 ms | 3.2 ms | 55 ms | 64 ms |
| 128 tokens | 6.7 ms | 5.3 ms | 197 ms | 63 ms |
| 512 tokens | 32 ms | 27 ms | 493 ms | 135 ms |
| 1024 tokens | 84 ms | 69 ms | 753 ms | 310 ms |
| 2048 tokens | 250 ms | 194 ms | 1545 ms | 746 ms |
| 4096 tokens (`--max-tokens 8192`) | 674 ms | 531 ms | 4877 ms | 2989 ms |
| 8192 tokens (`--max-tokens 8192`) | 1749 ms | 1388 ms | 9564 ms | 493 s ¹ |
| 10 texts of 32 tokens, one request | 18.5 ms | 14.6 ms | 142 ms | 82 ms |
| 10 texts of 2048 tokens, one request | 2.47 s | 0.99 s ² | 10.0 s | 33.1 s |
| one image (640 × 488, 266 soft tokens) | 489 ms | 383 ms | 1425 ms | 1031 ms |
| 8 clients sending single queries | 494 req/s, 16 ms median | 1002 req/s | | |
| memory at the default `--max-tokens 2048` | 161 MB | 174 MB | | |
| memory with `--max-tokens 8192` | 529 MB | 563 MB | | |

The Official columns are google/embeddinggemma-2 in fp32 through its official `sentence-transformers` stack on the same M4 (CPU with 10
threads, or the GPU through PyTorch MPS), warm median of 10 runs (3 above 2048 tokens), texts of about N
tokens with the document prompt; they are the accuracy reference, not a tuned GPU runtime. ¹ MPS at 8192
tokens took 493 s in that run (not investigated further). ² M5 Ultra has two Neural Engines; the long lane runs
two documents at once. The M5 Ultra numbers were taken while another Neural Engine embedding service was
running on the same machine.

Where the time goes (M4): over 98% of every text request is the Neural Engine. For a 2048-token text the
host spends 0.4 ms tokenizing, 0.1 ms on token rows, biases and pool, and 0.15 ms on Core ML inputs, the
output and JSON, around a 247 ms call. An image spends 16 ms decoding and resizing, 438 ms in the vision
encoder and 30 ms in the text function. Concurrent short queries that arrive while the Neural Engine is busy
are merged into shared calls (8 clients: 209 → 494 requests/s, median 38 → 16 ms, one client unchanged).

### `--max-tokens` trade-offs

The default favours start time and memory; the model's limit is 8192 tokens. Functions above the limit are
not loaded or compiled.

| `--max-tokens` | 2048 (default) | 8192 |
|---|---|---|
| functions loaded | 9 (6 text buckets, 3 vision) | 37 (+ 20 sliding-layer, 8 global/head) |
| first install, compile included (M4 / M5 Ultra) | 150 s / 138 s | + 134 s / + 113 s |
| start with the compile cached, self-test included | 4.5 s / 3.7 s | 27 s / 21 s |
| self-test at load | 1.2 s / 0.9 s | 10.5 s / 2.9 s |
| memory (M4) | 161 MB | 529 MB |
| text latency | up to 250 ms (2048 tokens) | 2049–4096: 0.5–0.7 s, 4097–8192: 1.35–1.75 s |

Texts from 2049 tokens on run layer by layer in 1024-token chunks (20 · ⌈n/1024⌉ + 4 calls): a text of
2049 tokens already takes about 0.5 s, and one of 4097 jumps to the 8192-row global layers (1.35 s).

## Accuracy

The reference ("official") is google/embeddinggemma-2 in fp32 through its official `transformers` stack on the CPU (mean pooling
over all tokens, L2; identical to `sentence-transformers` at cosine 1.0000000), not another quantized port.
Every input went through the release binary over HTTP exactly as a client sends it (`aneembed serve
--max-tokens 8192`, texts as strings, images as data URLs) on an M4, and the table reports the cosine
similarity to the reference per input. Inputs are public: multilingual READMEs (en/ko/ja/zh/fr) of an
Apache-2.0 project, the Korean and Japanese constitutions (public domain), MIT-licensed technical documents
and Apache-2.0 source code, cut to exact token counts with the document prompt; 13 images bundled with
scikit-learn, networkx, llama.cpp and stable-diffusion.cpp plus synthetic ones, at the official 280-token
budget. `usage.prompt_tokens` matched the reference token count in all 113 requests.

| Input | n | min cos | median cos |
|---|---:|---:|---:|
| text, one per request, 37–512 tokens | 48 | 0.999947 | 0.999975 |
| text, one per request, 517–2048 tokens | 24 | 0.999945 | 0.999976 |
| text, 14–40 short texts (9–200 tokens) per request, packed | 200 | 0.999946 | 0.999975 |
| text, 2049–4096 tokens (long path) | 8 | 0.999966 | 0.999970 |
| text, 4097–8192 tokens (long path) | 10 | 0.999899 | 0.999941 |
| image, one per request | 13 | 0.999954 | 0.999979 |
| images, up to 8 per request | 13 | 0.999957 | 0.999979 |

The default `--max-tokens 2048` gives the same numbers for everything up to 2048 tokens (same functions).
To reproduce: `scripts/accuracy/reference.py` writes the reference vectors (needs PyTorch and the original
model), `scripts/accuracy/compare.py` queries a running `aneembed serve` (standard library only).

The tokenizer is checked token for token against Hugging Face `tokenizers` on 12,420 generated texts
(every id must match), and image preprocessing bit for bit against `Gemma4ImageProcessor`.

## Self-test

The Neural Engine compiler runs on your Mac, per macOS build, and a miscompiled graph does not fail: it
returns wrong numbers. While converting the model, the usual attention layout came back with three of four
query heads zeroed in the global layers (cosine 0.55 for the layer), and a sliding-window mask built into the
graph as a constant was also computed wrongly; the package avoids both. To catch whatever a future macOS
might do, the package ships one golden test per function (`selftest.json` / `selftest.bin`, expectations
from the original model in fp32), and `aneembed` runs the test of every function it loads before serving.
A function below the tolerance (cosine 0.999, relative error 0.05) is not used: a larger bucket, a shorter
text limit or fewer image sizes take its place, the log says so, and `aneembed doctor` lists it.

## Known pitfalls

- **Graph-constant masks miscompile on the ANE.** The sliding-window band of Gemma's local layers is an
  input (`local_bias`), not a constant in the graph.
- **Variable shapes fall back to the CPU.** Every function has fixed shapes (buckets of 64…2048 tokens,
  1024-token chunks above that); several short texts are packed into one call with block-diagonal attention.
- **The first compile takes minutes**, and the OS caches it per model path, macOS build and executable (the
  same binary copied elsewhere compiles again). `models install` does it once; after a macOS update or an
  upgrade of `aneembed`, `doctor` reports `stale` and `serve` compiles again before it reports healthy. Do not
  move the model directory.
- **Memory reporting**: the ANE side does not show in the process footprint, and freed malloc pages only
  become reclaimable about two minutes later, so short-lived measurements overstate memory.
- **llama.cpp GGUF**: text vectors are compatible; image preprocessing in llama.cpp differs from the
  official recipe, so image vectors are not interchangeable.

## Running as a service

```xml
<!-- ~/Library/LaunchAgents/com.github.junmo-kim.aneembed.plist -->
<plist version="1.0"><dict>
  <key>Label</key><string>com.github.junmo-kim.aneembed</string>
  <key>ProgramArguments</key><array><string>/opt/homebrew/bin/aneembed</string><string>serve</string></array>
  <key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
  <key>StandardErrorPath</key><string>/tmp/aneembed.log</string>
</dict></plist>
```

There is no authentication or TLS. To serve other machines, keep `--host 127.0.0.1` and put a reverse
proxy in front.

## Build and test

```bash
swift build -c release
CI=true swift test                           # unit, golden and fake-runtime contract tests
Tests/Integration/release-binary-launch-smoke.sh .build/release/aneembed
ANEEMBED_TEST_MODEL_DIR=/path/to/model swift test --filter ModelTests   # real model, needs the ANE
aneembed bench --image photo.jpg --buckets --breakdown   # latency, memory, per-bucket costs
```

The test binary is another executable, so the first `ModelTests` run compiles the model for the Neural
Engine again (minutes).

## License

MIT for this code. The model package is downloaded separately and keeps its own license: EmbeddingGemma 2
by Google DeepMind, Apache-2.0, also subject to the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy); the vision encoder
conversion is from FluidInference/embeddinggemma-2-coreml (Apache-2.0). See [NOTICE](NOTICE).
