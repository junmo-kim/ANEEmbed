# ANEEmbed (`aneembed`)

[![CI](https://github.com/junmo-kim/ANEEmbed/actions/workflows/ci.yml/badge.svg)](https://github.com/junmo-kim/ANEEmbed/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Platform: macOS 15+ · Apple Silicon](https://img.shields.io/badge/platform-macOS%2015%2B%20%C2%B7%20Apple%20Silicon-lightgrey)

EmbeddingGemma 2 embeddings that run **only on the Apple Neural Engine**, as a single **100% Swift
binary with no third-party dependencies** (Foundation, Network, CoreML, ImageIO, Accelerate).

- **OpenAI drop-in**: `POST /v1/embeddings` with text and images, `encoding_format` float/base64,
  Matryoshka `dimensions` 768/512/256/128.
- **Leaves the GPU alone**: nothing runs on the GPU, so a local LLM keeps it to itself.
- **Accurate**: cosine ≥ 0.99990 (median 0.999975) to the **official** model (google/embeddinggemma-2 in fp32 through its official `sentence-transformers` stack) on every public
  test input, text up to 8192 tokens and images (see [Accuracy](#accuracy)).
- **Fast**: on an M5 Ultra a short query takes 3.4 ms, a 2048-token document 0.17 s and an 8192-token one
  1.2 s, each text in one Neural Engine call (see [Performance](#performance)).
- **Memory the OS can take back when idle**: the process itself uses about 160–380 MB at the default settings,
  all of it pageable. The Neural Engine's working buffers (5.3 GB at the default settings, 13.3 GB with
  `--max-tokens 8192`) are wired only while a call runs and about 10 s after it; idle, they are purgeable
  and the OS discards them under memory pressure (see [Memory](#memory)).
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
aneembed models install                  # about 4 minutes the first time (M5 Ultra)
aneembed doctor        # Neural Engine, model files, compile cache, self-test: what is ready and what to fix
```

For texts over 2048 tokens set the limit for every command, e.g. `export ANEEMBED_MAX_TOKENS=8192`, and run
`aneembed models warm` once (the long-text functions: about 6.5 more minutes on an M5 Ultra, see
[trade-offs](#--max-tokens-trade-offs)).

`aneembed models install --from DIR` installs a local copy of the package instead (same checks, no
network; every file must be a regular file). Installs take a lock in the models directory, so a second
`models install` waits for the first; the new revision replaces an old one in a single rename, and a revision
already installed is reused only when its manifest matches the pinned hash (or, with `--revision`, the
manifest the repository serves now). `--revision` takes one plain name (a commit hash, or letters, digits,
`.`, `_`, `-`); `models install` does not run with `--runtime fake`. `aneembed models warm` compiles again in place, which an upgraded `aneembed` needs once: the OS
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

One-off commands load and self-test the model each time (4.9 s on an M5 Ultra with a warm cache, 9.1 s with
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
| `GET /health` | `200` ready: `{"status": "ok"}`, or `{"status": "degraded", "not_served": [...], "max_context_length", "max_image_tokens"}` when loaded functions failed the self-test (or `"selftest": "skipped"` under `--selftest skip`); `503` while loading (the first start compiles for the Neural Engine) or after a load failure |
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
| `--selftest` | `ANEEMBED_SELFTEST` | `require` | `skip` serves every function without the [self-test](#self-test): **unsafe**, a miscompiled function returns wrong vectors without an error |

## Performance

`aneembed bench` measures this binary in process on an M5 Ultra, macOS 27.0.1 (requests go through the same router
as `serve`; only the loopback round trip is missing): warm median of 10 runs (3 above 2048 tokens), texts of exactly
N tokens including `<bos>`/`<eos>` and the document prompt. Memory is `phys_footprint` after the run and 150 s of
idling (freed pages are reclaimed late); see [Memory](#memory) for what it does not include. The first-release
column is the previous model layout and binary on the same Mac (v0.1.0 README).

| | M5 Ultra | M5 Ultra, first release |
|---|---:|---:|
| 32 tokens | 3.4 ms | 3.2 ms |
| 128 tokens | 5.5 ms | 5.3 ms |
| 512 tokens | 27.5 ms | 27 ms |
| 768 / 769 tokens | 46.4 / 70.4 ms | |
| 1024 tokens | 71 ms | 69 ms |
| 1280 / 1281 tokens | 94.0 / 114.5 ms | |
| 2048 tokens | 168 ms | 194 ms |
| 2049 / 3072 tokens (`--max-tokens 8192`) | 281 / 287 ms | |
| 3073 / 4096 tokens | 422 / 424 ms | 531 ms (4096) |
| 4097 / 6144 tokens | 769 / 759 ms | |
| 6145 / 8192 tokens | 1168 / 1187 ms | 1388 ms (8192) |
| 10 texts of 32 tokens, one request | 8.8 ms | 14.6 ms |
| 10 texts of 2048 tokens, one request | 0.86 s | 0.99 s |
| one image (640 × 488, 266 soft tokens) | 378 ms | 383 ms |
| 8 clients sending single queries | 1325 req/s, 5.9 ms median | 1002 req/s |
| process memory at the default `--max-tokens 2048` | 221 MB | 174 MB |
| process memory with `--max-tokens 8192` | 444 MB | 563 MB |

A text runs in the smallest function that holds it, so latency steps at 769 (`embed_1024`), 1281 (`winp1536`),
2049 (`winl3072`), 3073, 4097 and 6145 tokens and stays flat in between. Measured per call (`bench --buckets`):
`embed_64` … `embed_1280` 3.2 / 5.2 / 10.7 / 27.3 / 46.1 / 70.7 / 93.5 ms, `winp1536` / `winp2048` 115 / 171 ms,
`winl3072` … `winl8192` 286 / 424 / 771 / 1169 ms. The M5 Ultra numbers were taken while another Neural Engine
embedding service was running on the same machine.

M4: this layout has not been measured on an M4 yet. With the first layout a single request was 1.2–1.6× faster
on the M5 Ultra than on the M4 (one request runs on one Neural Engine; the M5 Ultra has two, which helps concurrent and batched
work), and process memory did not depend on the chip.

### `--max-tokens` trade-offs

The default favours start time and memory; the model's limit is 8192 tokens. Functions above the limit are
not loaded or compiled.

| `--max-tokens` | 2048 (default) | 8192 |
|---|---|---|
| functions loaded | 12 (7 `embed_S`, 2 `winp`, 3 vision) | 16 (+ 4 `winl`) |
| first install, compile included (M5 Ultra) | 242 s | + 383 s |
| start with the compile cached, self-test included | 4.9 s | 9.1 s |
| self-test at load | 1.1 s | 3.9 s |
| process memory | 221 MB | 444 MB |
| text latency | up to 171 ms (2048 tokens) | 2049–3072: 0.28 s, –4096: 0.42 s, –6144: 0.77 s, –8192: 1.19 s |

Functions load four at a time: one by one, a cached start took 10.5 s instead of 4.9 s (default) and 17.5 s instead
of 9.1 s (8192), but loading four at a time raises the brief peak of the process's memory during the start from
0.4 GB to 1.4 GB (`/usr/bin/time -l`, peak footprint) before it settles to the numbers above.

## Memory

Measured on an M5 Ultra with `footprint -w` and `vmmap`, macOS 27.0.1, `aneembed serve` with the release binary.
An `aneembed` process has two kinds of memory.

**Its own memory** (`phys_footprint`, what Activity Monitor shows): at the default `--max-tokens 2048`, 155–166 MB
idle and 267–383 MB five seconds after a mix of 200 requests (221 MB in `bench` after 150 s); with
`--max-tokens 8192`, 210–243 MB idle and 283–599 MB after the mix (444 MB in `bench`). It is malloc memory
(190–240 MB after the mix at the default settings), the Core ML input/output surfaces (`IOSurface`, 62 MB / 147 MB)
and memory-mapped files, mostly the 256 MB token table, of which 9–31 MB was resident, all clean. All of it is
pageable: with `memory_pressure -p 12` pushing the system to 12 % available memory, 292 of the process's 294 MB of
dirty memory, the IOSurfaces included, was compressed or swapped out; once the pressure was gone, the first
requests took 47 / 62 / 143 ms (query / 500 / 1500 tokens; 5 / 47 / 117 ms before), then warm again. Starting
needs more for a moment: loading peaks at about 1.4 GB, which falls back within one to two minutes.

**The Neural Engine's buffers.** Core ML gives each loaded function working memory for the Neural Engine. It is
owned by the process but not part of its footprint (`footprint` lists it as `Owned physical footprint (unmapped)
(neural)`): 5.25 GB at the default settings (12 functions), 13.3 GB with `--max-tokens 8192` (16 functions). It is
**wired only while it is used**:

| | default (2048) | `--max-tokens 8192` |
|---|---:|---:|
| right after start (the self-test runs every function), for about 60 s | 5.25 GB wired | 13.3 GB wired |
| idle | 0 wired (5.25 GB purgeable) | 0 wired (13.3 GB purgeable) |
| during and about 10 s after a short query | 266 MB wired | 266 MB wired |
| … a 1617-token text | 579 MB wired | |
| … a 6631-token text | | 3.9 GB wired |
| 5 s after a mix of 200 requests (queries, 500/1500-token texts, images; 8192 also 8100-token texts) | 3.1 GB wired | 6.6 GB wired |

Idle buffers are purgeable: under the same memory pressure the OS discarded 1.6 GB of them (5.25 → 3.65 GB)
without any error, and the next calls recreated them. That has a cost: the first request after the buffers were
unwired or discarded took 19–139 ms for a short query (3–5 ms warm) and 0.12–0.14 s more for a 6631-token text.
Under steady load the buffers of the functions in use stay wired; the "mix" row, taken 5 s after the last request,
shows the scale.

For comparison, the first release (per-layer long path) on the same Mac: 4.0 GB of Neural Engine buffers at the
default settings, 21.5 GB with `--max-tokens 8192`, and a 6631-token text wired 14.3 GB instead of 3.9 GB. By
contrast, a Metal (GPU) LLM server running on the same Mac had 168 GB of `IOAccelerator (graphics)` memory, all
of it wired, at the time of the measurement.

## Design

**Fixed shapes, packed.** The Neural Engine runs fixed-shape graphs; enumerated or flexible shapes moved the
whole graph to the CPU in our tests. So the model package has one function per size, all sharing one copy of
the weights: `embed_64` … `embed_1280` with full attention, and above that windowed functions (below). A request's
texts are packed back to back into as few calls as possible, with a block-diagonal attention mask so packed
texts never see each other (positions need no reset: RoPE is relative). The planner picks the cheapest set of
calls from the measured per-call latency of each size, and short queries that arrive while the Neural Engine
is busy are merged into the next call.

**Windowed attention, folded.** EmbeddingGemma 2 has 24 layers; 20 of them only look 512 tokens to each side.
Full attention over S rows costs S² in those layers for nothing, so the windowed functions split the rows into
blocks of 512 and let each block attend to the 1536 rows around it (512 on each side), with the mask passed in as
a `[S, 1536]` band. The 4 global layers attend to every row. That keeps one fixed-shape call per text up to 8192
tokens (`winl3072` … `winl8192`) and makes 1281–2048-token calls cheaper (`winp1536`, `winp2048`, which still
pack). The first release ran texts over 2048 tokens layer by layer in 1024-row chunks, 20·⌈n/1024⌉ + 4 calls per
text; one call is simpler and faster. Attention uses a "fold" layout: the query heads that share a K/V head are
folded into the row dimension, because the usual `repeat_interleave` layout is miscompiled (next paragraph).

**Miscompiles, and the self-test.** The Neural Engine compiler runs on your Mac and a wrong graph does not fail,
it returns wrong numbers. Converting this model, the usual attention layout came back with three of four query
heads zeroed in the global layers, a sliding-window mask built into the graph as a constant was computed wrongly,
and padding inside the graph was correct but about 3× slower. Rebuilt with the usual layout, `winp2048`,
`winl4096` and `winl8192` score cosine 0.96, 0.95 and 0.89 on the Neural Engine. The package therefore passes
every mask as an input, uses the fold layout, and ships one golden test per function that `aneembed` runs before
serving (see [Self-test](#self-test)).

**Compiled at install, per Mac.** The download is the portable `.mlpackage`. `models install` compiles it, then
loads each function once so the OS compiles it for the Neural Engine and caches the result. The cache is keyed by
the model path, the macOS build and the executable: an upgraded `aneembed`, a macOS update or a moved model
directory compiles again (`doctor` says when).

## Accuracy

The reference ("official") is google/embeddinggemma-2 in fp32 through its official `transformers` stack on the CPU (mean pooling
over all tokens, L2; identical to `sentence-transformers` at cosine 1.0000000), not another quantized port.
Every input went through the release binary over HTTP exactly as a client sends it (`aneembed serve
--max-tokens 8192`, texts as strings, images as data URLs) on an M5 Ultra, and the table reports the cosine
similarity to the reference per input. Inputs are public: multilingual READMEs (en/ko/ja/zh/fr) of an
Apache-2.0 project, the Korean and Japanese constitutions (public domain), MIT-licensed technical documents
and Apache-2.0 source code, cut to exact token counts with the document prompt; 13 images bundled with
scikit-learn, networkx, llama.cpp and stable-diffusion.cpp plus synthetic ones, at the official 280-token
budget. `usage.prompt_tokens` matched the reference token count in all 167 requests.

| Input | n | min cos | median cos |
|---|---:|---:|---:|
| text, one per request, 37–512 tokens | 48 | 0.999947 | 0.999975 |
| text, one per request, 389–2048 tokens (every function size, and one token over each) | 70 | 0.999945 | 0.999976 |
| text, 10–40 short texts (9–200 tokens) per request, packed | 200 | 0.999946 | 0.999975 |
| text, 2049–4096 tokens (incl. 3072 / 3073) | 12 | 0.999961 | 0.999969 |
| text, 4097–8192 tokens (incl. 6144 / 6145) | 14 | 0.999902 | 0.999936 |
| image, one per request | 13 | 0.999954 | 0.999979 |
| images, up to 8 per request | 13 | 0.999957 | 0.999979 |

The default `--max-tokens 2048` gives the same numbers for everything up to 2048 tokens (same functions). The
model package itself, checked through the Python host contract (`tools/convert/verify.py`), gives the same
picture per function: 380 packed and single texts up to 2048 tokens at ≥ 0.999936 and 44 long texts at ≥ 0.999889
(the lowest at 8191–8192 tokens).
To reproduce: `tools/convert/fetch.py data`, `corpus.py` and `images.py` rebuild the inputs from pinned public
sources, `scripts/accuracy/reference.py` writes the reference vectors (needs PyTorch and the original
model), `scripts/accuracy/compare.py` queries a running `aneembed serve` (standard library only).

### Image input formats

The official path reads an image with transformers' `load_image` (PIL): RGB, EXIF orientation applied, alpha
dropped. `aneembed` decodes JPEG with its own decoder (libjpeg-turbo's algorithms, bit-exact with PIL) and every
other format with ImageIO, taking the decoded values without colour management. Measured like the table above
(over HTTP, M5 Ultra; JPEG: the 4 JPEG files among the public images), with the 13 public images re-encoded in
each format (`scripts/accuracy/formats.py`; 3 files above the 8 MiB data-URL limit are left out; the reference
reads the same files with `load_image`, HEIC through `pillow-heif` 1.8.1, libheif 1.23.6):

| Format | Decoder | n | min cos | median cos | Decoded pixels vs PIL |
|---|---|---:|---:|---:|---|
| JPEG | own | 4 | 0.999954 | 0.999966 | identical (bit-exact on the golden test images) |
| PNG | ImageIO | 13 | 0.999954 | 0.999979 | identical |
| WebP, lossy (quality 90) and lossless | ImageIO | 26 | 0.999949 | 0.999978 | identical |
| GIF (256 colours) | ImageIO | 13 | 0.999950 | 0.999979 | identical |
| TIFF, BMP | ImageIO | 24 | 0.999954 | 0.999977 | identical |
| animated GIF / WebP / PNG (3 frames) | ImageIO, first frame | 38 | 0.999950 | 0.999979 | identical (frame 0) |
| HEIC (8-bit, written by `sips`) | ImageIO | 13 | 0.997272 | 0.999463 | differ: mean 0.4–1.0 levels, 11.8 on a noise image |
| CMYK / YCCK JPEG | ImageIO | | | | differ: mean 32, max 130 levels (synthetic test image) |

- **Animated images** use the first frame only, as the official transformers path does; other frames are ignored.
- **HEIC** is the one format that does not match: ImageIO's HEVC decode is about 0.2–0.8 levels darker per
  channel even on flat areas and differs strongly at sharp colour edges (chroma upsampling), compared with libheif.
  It is not colour management (both files and decoders are sRGB): the same HEIC pixels decoded by PIL and sent as
  PNG score min 0.999913. Matching it would need a HEVC decoder in `aneembed`; convert HEIC to PNG or JPEG where
  this matters. Note that PIL itself reads HEIC only with the `pillow-heif` plugin.
- **CMYK JPEG** falls back to ImageIO, which converts colours differently from PIL.
- Images are limited to 40 megapixels and to 8 MiB as a data URL (about 6 MiB of file); formats ImageIO cannot
  read return `400 image_decode_failed`.

The tokenizer is checked token for token against Hugging Face `tokenizers` on 12,420 generated texts
(every id must match), and image preprocessing bit for bit against `Gemma4ImageProcessor`.

## Self-test

The model package ships one golden test per function (`selftest.json` / `selftest.bin`, 16 tests, expectations
from the original model in fp32), and `aneembed` runs the test of every function it loads before serving. Only a
function that passes in this run serves (an allowlist): one below the tolerance (cosine 0.999; packing functions
also unused rows ≤ 0.001, images relative error 0.05; a package may tighten these, not loosen them), one with a
NaN or infinite output, or one the package has no test for is not used. A larger function, a shorter text limit
or fewer image sizes take its place, the log gives the limits that remain, `/health` reports `degraded`, and
`aneembed doctor` lists it (records of functions a smaller `--max-tokens` did not load are kept). On an M5 Ultra
(macOS 27.0.1) every function scores cosine ≥ 0.99990; the same functions rebuilt with the usual attention layout
score 0.89–0.96 on the Neural Engine and fail (see [Design](#design)).

A package without readable self-test data, or without a valid `manifest.json` declaring every function this build
loads, does not load (`doctor` shows why). `--selftest skip` serves every function unchecked; it is meant for
diagnosing a package, not for use. Each load also checks every file against the manifest's sizes and hashes
`tokenizer.json`, `selftest.json` and `selftest.bin`, which the self-test cannot judge (it feeds token ids);
`aneembed models verify` hashes everything and reports whether each compiled model is present and complete
(a compile is not byte-for-byte reproducible, so its hash is informational).

## Known pitfalls

- **Graph-constant masks miscompile on the ANE.** The sliding-window mask of Gemma's local layers is an
  input (`local_bias`, or the `band_bias` of the windowed functions), not a constant in the graph.
- **Variable shapes fall back to the CPU.** Every function has fixed shapes (64…1280 tokens with full attention,
  1536 and 2048 windowed and packable, 3072…8192 windowed for one text); several short texts are packed into one
  call with block-diagonal attention.
- **The first compile takes minutes**, and the OS caches it per model path, macOS build and executable (the
  same binary copied elsewhere compiles again). `models install` does it once; after a macOS update or an
  upgrade of `aneembed`, `doctor` reports `stale` and `serve` compiles again before it reports healthy. Do not
  move the model directory.
- **Memory reporting**: the Neural Engine's buffers do not show in the process footprint (`footprint -w` lists
  them as `(neural)`), and freed malloc pages only become reclaimable about a minute later, so short-lived
  measurements overstate the process's own memory and miss the Neural Engine's.
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

The model package itself is built by [`tools/convert`](tools/convert/README.md) (Python, CPU only) from the
original weights at a pinned revision; a rebuild reproduces its weights, compiled program and self-test byte for
byte.

## License

MIT for this code. The model package is downloaded separately and keeps its own license: EmbeddingGemma 2
by Google DeepMind, Apache-2.0, also subject to the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy); the vision encoder
conversion is from FluidInference/embeddinggemma-2-coreml (Apache-2.0). See [NOTICE](NOTICE).
