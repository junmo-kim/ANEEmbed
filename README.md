# ANEEmbed (`aneembed`)

[![CI](https://github.com/junmo-kim/ANEEmbed/actions/workflows/ci.yml/badge.svg)](https://github.com/junmo-kim/ANEEmbed/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Platform: macOS 15+ · Apple Silicon](https://img.shields.io/badge/platform-macOS%2015%2B%20%C2%B7%20Apple%20Silicon-lightgrey)


EmbeddingGemma 2 embeddings that run **only on the Apple Neural Engine**, as a single **100% Swift binary with no third-party dependencies** (Foundation, Network, CoreML, ImageIO, Accelerate). An OpenAI-compatible `/v1/embeddings` server, a CLI and an MCP server in one.

## At a glance

The same 16 GB Apple M4 (macOS 27), the same inputs, warm medians, measured on a quiet machine. Every runtime runs the official google/embeddinggemma-2 model; accuracy is the lowest cosine to the official fp32 vectors over the same 26 texts. Details, versions and flags: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

| Apple M4, 16 GB | **ANEEmbed** (Neural Engine) | llama.cpp (Metal GPU) | MLX (Metal GPU) | official PyTorch, fp32 (MPS GPU) |
|---|---:|---:|---:|---:|
| short query (32 tokens) | **7.0 ms** | 14 ms | 15–17 ms | 59–72 ms |
| 2048-token document | **221 ms** | 406 ms | 396 ms | 616 ms |
| 8192-token document | **1.45 s** | 2.9 s | 3.5 s | 6.6 s (13.4 GB process) |
| image (official 280-token budget) | 484 ms | 575 ms¹ | **468 ms**² | 958 ms |
| 10 × 2048-token documents, one request | **2.2 s** | 4.0 s | 4.0 s | 6.4 s³ |
| 8 clients sending short queries | **412 req/s** | 189 req/s | n/a⁴ | n/a⁴ |
| process memory, idle | **140–270 MB** | 0.8–1.2 GB | 5.1–5.7 GB held | 6.7 GB |
| memory wired while idle | **none** | 2.1–2.6 GB | none | none |
| uses the GPU | **no** | yes | yes | yes |
| lowest text cosine vs official fp32 | 0.99997 | 0.999997 (BF16) / 0.99989 (Q8_0) | 0.99994 (bf16) / 0.99976 (8bit) | 1 (reference) |

ANEEmbed and llama.cpp are measured as servers over loopback HTTP; MLX and PyTorch have no embedding server here and are measured in process, which spares them the HTTP round trip. For llama.cpp and MLX each cell shows the best of BF16 and 8-bit. On an M5 Ultra ANEEmbed takes 3.4 ms, 168 ms and 1.19 s for the three text sizes.

¹ At `--image-min-tokens 280` (304 tokens, cosine 0.99934). llama.cpp's default budget is 134 tokens: 217 ms, but cosine 0.9963 to the official image vector.

² In process, with PyTorch and torchvision added for the image processor; its cosine to the official image vector was not measured.

³ Only with the user logged out: the process reached 13.1 GB. With apps open it ran out of memory, as did the 8192-token case (6.6 s logged out, 13.4 GB).

⁴ No embedding server; measured in process only.

## Why

- **Leaves the GPU alone.** Nothing runs on the GPU, so a local LLM keeps all of it.
- **Memory the OS can take back.** The process itself uses about 140–380 MB, all of it pageable: under memory pressure it is compressed or swapped like any app. The Neural Engine's working buffers are wired only while a call runs and for about 10 s after it; idle, they are purgeable. A Metal runtime keeps its buffers wired, or held, for as long as the model is loaded. Request data is never written to disk.
- **Fast.** One Neural Engine call per text up to 8192 tokens, short texts packed together, and separate lanes so long documents and images do not hold short queries back.
- **Accurate.** Cosine ≥ 0.99990 (median 0.999975) to the official fp32 model on every public test input, texts up to 8192 tokens and images (see [Accuracy](#accuracy)).
- **Checks itself.** Every model function is tested against golden outputs when it loads, and one the Neural Engine compiler got wrong is not used (see [Self-test](#self-test)).
- **OpenAI drop-in, CLI and MCP** from one binary: `POST /v1/embeddings` with text and images, `encoding_format` float/base64, Matryoshka `dimensions` 768/512/256/128; `aneembed embed`, `aneembed serve`, `aneembed mcp`.

## Trade-offs

- **The first install compiles for the Neural Engine:** about 4 minutes (M5 Ultra) to 5.5 minutes (M5 Max), plus about 6 more for `--max-tokens 8192`. The OS caches the result; an upgraded `aneembed` or a macOS update compiles again. llama.cpp and MLX start in seconds.
- **llama.cpp BF16 is closer to the official vectors** (0.999997 against ANEEmbed's 0.99997). Both are far closer than the 8-bit ports.
- **Texts over 2048 tokens need `--max-tokens 8192`**, which compiles four more functions. At start the self-test wires about 9 GB of Neural Engine buffers for about a minute; on a 16 GB Mac, start it when memory is free.
- **The Neural Engine runs one call at a time.** While a request of large images is processed, short queries wait for it (on an M4: query p90 0.28 s while one client uploads 8 photos of 12–24 MP per request, 0.66 s with three such clients). See [Best practices](#best-practices) for bulk image jobs.
- **Apple silicon and macOS 15 or later only.** Text up to 8192 tokens (2048 by default); images at the model's official 280-token budget; HEIC decodes slightly differently from the official path (see [Image input formats](#image-input-formats)).

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

Then install the model ([junmo-k/embeddinggemma-2-ane](https://huggingface.co/junmo-k/embeddinggemma-2-ane), about 900 MB). This downloads a pinned revision, checks every file's SHA-256 against a hash compiled into the binary, compiles the package on this Mac, then compiles each function for the Neural Engine once and runs the self-test. The OS caches the Neural Engine compile, so later starts take seconds instead of minutes:

```bash
aneembed models install                  # about 4 minutes the first time (M5 Ultra)
aneembed doctor        # Neural Engine, model files, compile cache, self-test: what is ready and what to fix
```

For texts over 2048 tokens set the limit for every command, e.g. `export ANEEMBED_MAX_TOKENS=8192`, and run `aneembed models warm` once (the long-text functions: about 6.5 more minutes on an M5 Ultra, see [trade-offs](#--max-tokens-trade-offs)).

`aneembed models install --from DIR` installs a local copy of the package instead (same checks, no network; every file must be a regular file). Installs take a lock in the models directory, so a second `models install` waits for the first; the new revision replaces an old one in a single rename, and a revision already installed is reused only when its manifest matches the pinned hash (or, with `--revision`, the manifest the repository serves now). `--revision` takes one plain name (a commit hash, or letters, digits, `.`, `_`, `-`); `models install` does not run with `--runtime fake`. `aneembed models warm` compiles again in place, which an upgraded `aneembed` needs once: the OS keeps the compile per model path, macOS build and executable (`serve` does it by itself before it reports healthy, and `doctor` says when it is due).

The release binary is ad-hoc signed. A tarball downloaded with a browser carries a quarantine attribute; clear it with `xattr -d com.apple.quarantine aneembed` (Homebrew, mise and curl do not set it).

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

One-off commands load and self-test the model each time (4.9 s on an M5 Ultra with a warm cache, 9.1 s with `--max-tokens 8192`). For regular use keep `serve` or `mcp` running.

### MCP

```json
{ "mcpServers": { "aneembed": { "command": "aneembed", "args": ["mcp"] } } }
```

Tools: `embed_text`, `embed_image`, `similarity`. The same tools are reachable over HTTP as `POST /mcp` and as one-shot `QUERY /{tool}?<options>` with the raw text or image bytes as the body. Over stdio a message (one line) is limited to `--max-body`: a longer one is skipped and answered with a `-32600` error (`id` null). At most 16 messages are handled at once; further input is not read until one finishes. `embed --stdin` and `tokenize --stdin` stop with `line_too_long` (exit 65) at a line over `--max-body`. `embed_image` checks the number of images before decoding any of them, and each image's size before decoding it.

## API

| Endpoint | |
|---|---|
| `POST /v1/embeddings` | OpenAI-compatible. `input`: string, list of strings, or list of llama-server image items `{"content": [{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}]}` (texts and images are not mixed in one request) |
| `GET /v1/models` | OpenAI's model object (`id`, `object`, `created`, `owned_by`) plus `max_context_length`, `max_image_tokens`, `dimensions`, `image_preprocessing` |
| `GET /health` | `200` ready: `{"status": "ok"}`, or `{"status": "degraded", "not_served": [...], "max_context_length", "max_image_tokens"}` when loaded functions failed the self-test (or `"selftest": "skipped"` under `--selftest skip`); `503` while loading (the first start compiles for the Neural Engine) or after a load failure |
| `POST /mcp` | MCP JSON-RPC, one message per request |
| `QUERY /{tool}` | `embed_text` (UTF-8 body), `embed_image` (image body), `similarity` (JSON body); YAML by default, `?format=json` |

Request fields of `/v1/embeddings`:

- `encoding_format`: `float` (default) or `base64` (little-endian float32).
- `dimensions`: 768 (default), 512, 256 or 128: the first N values, re-normalized.
- `input_type` (extension): `query` or `document` adds the model's task prefix (`task: search result | query: ` / `title: none | text: `). Without it texts are embedded as given, like llama-server, and callers add the prefix themselves.
- `model`, `user`: accepted and ignored; the response names the configured model.

Texts over the configured limit are rejected with `400 context_length_exceeded`, never truncated; a text of more than 93 bytes per allowed token (the longest vocabulary entry) is refused before it is tokenized, as having at least that many tokens. `usage.prompt_tokens` counts real tokens including `<bos>`/`<eos>`; an image counts its soft tokens + 4. Errors use OpenAI's shape: `{"error": {"message", "type", "param", "code"}}`. Server-side failures (`500`, and `503` after a failed load) carry the stable `code` and a generic message; their detail is in the server log. JSON bodies must be UTF-8, like OpenAI's API: invalid bytes are `400 invalid_json` (a raw `QUERY /embed_text` body `400 invalid_utf8`), never replaced with U+FFFD.

HTTP/1.1 with keep-alive. `HEAD /health` answers with the `GET` response's headers, `Content-Length` included, and no body. A body needs `Content-Length`: chunked uploads are `411`, and ambiguous framing (`Transfer-Encoding` with `Content-Length`, other transfer codings, or repeated `Content-Length` headers with different values) is `400` and closes the connection. A request head must arrive within 60 s (of connecting, or of the previous response on a kept-alive connection); a body may take longer as long as no pause between its bytes exceeds 60 s. A response must be sent within 60 s plus 1 s per 64 KiB; a client that stops reading is disconnected, so it cannot hold a connection slot.

## Configuration

Every option is a flag and an environment variable of the same name (flag > environment > default). `aneembed serve --print-config` shows each value and where it came from.

| Flag | Environment | Default | |
|---|---|---|---|
| `--model-dir` | `ANEEMBED_MODEL_DIR` | `<data-dir>/models/current` | model package directory |
| `--data-dir` | `ANEEMBED_DATA_DIR` | `~/Library/Application Support/ANEEmbed` | installed models and state |
| `--host` | `ANEEMBED_HOST` | `127.0.0.1` | anything else exposes an unauthenticated API |
| `--port` | `ANEEMBED_PORT` | `18180` | |
| `--max-tokens` | `ANEEMBED_MAX_TOKENS` | `2048` | up to 8192; only functions up to it are loaded and compiled |
| `--model-name` | `ANEEMBED_MODEL_NAME` | `embeddinggemma-2` | |
| `--max-body` | `ANEEMBED_MAX_BODY` | `72` (MiB) | 8 image data URLs of up to 8 MiB plus JSON; also what all connections together may buffer of bodies over 1 MiB ([memory](docs/PERFORMANCE.md#memory)) |
| `--max-connections` | `ANEEMBED_MAX_CONNECTIONS` | `64` | open HTTP connections; more are answered `503 too_many_connections` (`Retry-After: 1`) and closed |
| `--max-inputs` / `--max-images` | `ANEEMBED_MAX_INPUTS` / `ANEEMBED_MAX_IMAGES` | `2048` / `8` | per request |
| `--query-concurrency` / `--image-concurrency` / `--long-concurrency` | `ANEEMBED_*_CONCURRENCY` | `8` / `2` / `2` | separate lanes so long documents and images cannot hold short queries back |
| `--runtime` | `ANEEMBED_RUNTIME` | `coreml` | `fake` runs without a model, for testing integrations |
| `--selftest` | `ANEEMBED_SELFTEST` | `require` | `skip` serves every function without the [self-test](#self-test): **unsafe**, a miscompiled function returns wrong vectors without an error |


## Best practices

- **Use the model's prompts.** Queries as `task: search result | query: …` and documents as `title: none | text: …` (or send `input_type`: `query` / `document`). Without them texts are embedded as given.
- **Embed an index and its queries with the same runtime.** Vectors from ANEEmbed, llama.cpp, MLX and PyTorch agree to cosine ≈ 0.9999 for text, but not bit for bit, and llama.cpp's image preprocessing differs from the official recipe (cosine ≈ 0.996). Re-embed when you switch.
- **Bulk image jobs (backfills): one client, 4–8 images per request.** More clients barely add throughput: the Neural Engine runs one call at a time, and large JPEGs are decoded on the CPU, at most `--image-concurrency` (default 2) at a time. On an M4 with 12–24 MP photos, one client reached 1.19 images/s at 1 image per request, 1.46 at 4 and 1.51 at 8. Two or three clients reached 1.56–1.59 (+3–5 %), but short queries sent meanwhile waited about 2.5 times longer (p90 0.66–0.70 s instead of 0.28 s). If searches run at the same time, prefer one client. Downscaling very large photos on the client speeds up decoding, at the cost of pixels that differ slightly from the official preprocessing.
- **Long documents: chunk them, or raise `--max-tokens`.** The default 2048 keeps start time and memory low; texts over the limit are rejected (`400 context_length_exceeded`), never truncated.
- **Keep `serve` (or `mcp`) running.** One-off CLI commands load and self-test the model each time (about 5 s).

## Performance and memory

Summary; the measurements, methods and every number are in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

| ANEEmbed, warm median | Apple M4 | M5 Max | M5 Ultra |
|---|---:|---:|---:|
| short query | 7.0 ms (HTTP) | 3.2 ms | 3.4 ms |
| 512 / 2048 tokens | 35 / 221 ms | 26 / 167 ms | 27.5 / 168 ms |
| 4096 / 8192 tokens (`--max-tokens 8192`) | – / 1.45 s | 410 ms / 1.16 s | 424 ms / 1.19 s |
| image, 280-token budget | 484 ms | 383 ms | 378 ms |

M4 over loopback HTTP, M5 Max and M5 Ultra with `aneembed bench` in process. A text runs in the smallest function that holds it, so latency steps at each function size (for example at 769, 1281, 2049, 3073, 4097 and 6145 tokens) and is flat in between.

- **Process memory**: about 140–380 MB, all pageable. A request keeps at most 3 Neural Engine calls of its short texts in flight, so even a batch of 2048 texts holds only a few input buffers (2048 × 512-token texts: 24 MiB).
- **Neural Engine buffers**: owned by the process but outside its footprint, 5.3 GB at the default settings and 13.3 GB with `--max-tokens 8192`. They are wired only while used: for about a minute after start (the self-test), and during and about 10 s after each call. Idle, none is wired and the OS may discard them; the next call recreates them, which costs up to about 0.1 s once.
- **Request bodies**: all connections share a budget of one `--max-body` (72 MiB). Bodies up to 1 MiB, which covers a single text of up to 8192 tokens, never wait; larger ones (image uploads, big batches) queue for it.
- **Images**: a JPEG streams from the decoder into the resize, so a 40-megapixel baseline JPEG adds about 4 MiB. Progressive JPEG and other formats need the whole image: up to about 310 MiB per concurrent decode.

### `--max-tokens` trade-offs

| `--max-tokens` | 2048 (default) | 8192 |
|---|---|---|
| functions loaded | 12 (7 `embed_S`, 2 `winp`, 3 vision) | 16 (+ 4 `winl`) |
| first install, compile included (M5 Ultra) | 242 s | + 383 s |
| start with the compile cached, self-test included (M5 Ultra) | 4.9 s | 9.1 s |
| Neural Engine buffers wired during the start | 5.3 GB | 9–13 GB |
| longest text (M5 Ultra) | 2048 tokens, 171 ms | 8192 tokens, 1.19 s |

## Design

**Fixed shapes, packed.** The Neural Engine runs fixed-shape graphs; enumerated or flexible shapes moved the whole graph to the CPU in our tests. So the model package has one function per size, all sharing one copy of the weights: `embed_64` … `embed_1280` with full attention, and above that windowed functions (below). A request's texts are packed back to back into as few calls as possible, with a block-diagonal attention mask so packed texts never see each other (positions need no reset: RoPE is relative). The planner picks the cheapest set of calls from the measured per-call latency of each size, and short queries that arrive while the Neural Engine is busy are merged into the next call.

**Windowed attention, folded.** EmbeddingGemma 2 has 24 layers; 20 of them only look 512 tokens to each side. Full attention over S rows costs S² in those layers for nothing, so the windowed functions split the rows into blocks of 512 and let each block attend to the 1536 rows around it (512 on each side), with the mask passed in as a `[S, 1536]` band. The 4 global layers attend to every row. That keeps one fixed-shape call per text up to 8192 tokens (`winl3072` … `winl8192`) and makes 1281–2048-token calls cheaper (`winp1536`, `winp2048`, which still pack). Attention uses a "fold" layout: the query heads that share a K/V head are folded into the row dimension, because the usual `repeat_interleave` layout is miscompiled (next paragraph).

**Miscompiles, and the self-test.** The Neural Engine compiler runs on your Mac and a wrong graph does not fail, it returns wrong numbers. Converting this model, the usual attention layout came back with three of four query heads zeroed in the global layers, a sliding-window mask built into the graph as a constant was computed wrongly, and padding inside the graph was correct but about 3× slower. Rebuilt with the usual layout, `winp2048`, `winl4096` and `winl8192` score cosine 0.96, 0.95 and 0.89 on the Neural Engine. The package therefore passes every mask as an input, uses the fold layout, and ships one golden test per function that `aneembed` runs before serving (see [Self-test](#self-test)).

**Compiled at install, per Mac.** The download is the portable `.mlpackage`. `models install` compiles it, then loads each function once so the OS compiles it for the Neural Engine and caches the result. The cache is keyed by the model path, the macOS build and the executable: an upgraded `aneembed`, a macOS update or a moved model directory compiles again (`doctor` says when).

## Accuracy

The reference ("official") is google/embeddinggemma-2 in fp32 through its official `transformers` stack on the CPU (mean pooling over all tokens, L2; identical to `sentence-transformers` at cosine 1.0000000), not another quantized port. Every input went through the release binary over HTTP exactly as a client sends it (`aneembed serve --max-tokens 8192`, texts as strings, images as data URLs) on an M5 Ultra, and the table reports the cosine similarity to the reference per input. Inputs are public: multilingual READMEs (en/ko/ja/zh/fr) of an Apache-2.0 project, the Korean and Japanese constitutions (public domain), MIT-licensed technical documents and Apache-2.0 source code, cut to exact token counts with the document prompt; 13 images bundled with scikit-learn, networkx, llama.cpp and stable-diffusion.cpp plus synthetic ones, at the official 280-token budget. `usage.prompt_tokens` matched the reference token count in all 167 requests.

| Input | n | min cos | median cos |
|---|---:|---:|---:|
| text, one per request, 37–512 tokens | 48 | 0.999947 | 0.999975 |
| text, one per request, 389–2048 tokens (every function size, and one token over each) | 70 | 0.999945 | 0.999976 |
| text, 10–40 short texts (9–200 tokens) per request, packed | 200 | 0.999946 | 0.999975 |
| text, 2049–4096 tokens (incl. 3072 / 3073) | 12 | 0.999961 | 0.999969 |
| text, 4097–8192 tokens (incl. 6144 / 6145) | 14 | 0.999902 | 0.999936 |
| image, one per request | 13 | 0.999954 | 0.999979 |
| images, up to 8 per request | 13 | 0.999957 | 0.999979 |

The default `--max-tokens 2048` gives the same numbers for everything up to 2048 tokens (same functions). The model package itself, checked through the Python host contract (`tools/convert/verify.py`), gives the same picture per function: 380 packed and single texts up to 2048 tokens at ≥ 0.999936 and 44 long texts at ≥ 0.999889 (the lowest at 8191–8192 tokens). To reproduce: `tools/convert/fetch.py data`, `corpus.py` and `images.py` rebuild the inputs from pinned public sources, `scripts/accuracy/reference.py` writes the reference vectors (needs PyTorch and the original model), `scripts/accuracy/compare.py` queries a running `aneembed serve` (standard library only).

### Image input formats

The official path reads an image with transformers' `load_image` (PIL): RGB, EXIF orientation applied, alpha dropped. `aneembed` decodes JPEG with its own decoder (libjpeg-turbo's algorithms, bit-exact with PIL) and every other format with ImageIO, taking the decoded values without colour management. Measured like the table above (over HTTP, M5 Ultra; JPEG: the 4 JPEG files among the public images), with the 13 public images re-encoded in each format (`scripts/accuracy/formats.py`; 3 files whose data URLs were over the 8 MiB limit are left out; the reference reads the same files with `load_image`, HEIC through `pillow-heif` 1.8.1, libheif 1.23.6):

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
- **HEIC** is the one format that does not match: ImageIO's HEVC decode is about 0.2–0.8 levels darker per channel even on flat areas and differs strongly at sharp colour edges (chroma upsampling), compared with libheif. It is not colour management (both files and decoders are sRGB): the same HEIC pixels decoded by PIL and sent as PNG score min 0.999913. Matching it would need a HEVC decoder in `aneembed`; convert HEIC to PNG or JPEG where this matters. Note that PIL itself reads HEIC only with the `pillow-heif` plugin.
- **CMYK JPEG** falls back to ImageIO, which converts colours differently from PIL.
- **Progressive JPEGs** with more than 100 scans, or with scans out of the order the format defines, are refused (`400 image_decode_failed`, "JPEG refused: …") instead of decoded; libjpeg and PIL would decode the latter with a warning. Standard progressive files (10 scans) are unaffected.
- Images are limited to 40 megapixels and to 8 MiB as a data URL (about 6 MiB of image file); formats ImageIO cannot read return `400 image_decode_failed`.

The tokenizer is checked token for token against Hugging Face `tokenizers` on 12,420 generated texts (every id must match), and image preprocessing bit for bit against `Gemma4ImageProcessor`.

## Self-test

The model package ships one golden test per function (`selftest.json` / `selftest.bin`, 16 tests, expectations from the original model in fp32), and `aneembed` runs the test of every function it loads before serving. Only a function that passes in this run serves (an allowlist): one below the tolerance (cosine 0.999; packing functions also unused rows ≤ 0.001, images relative error 0.05; a package may tighten these, not loosen them), one with a NaN or infinite output, or one the package has no test for is not used. A larger function, a shorter text limit or fewer image sizes take its place, the log gives the limits that remain, `/health` reports `degraded`, and `aneembed doctor` lists it (records of functions a smaller `--max-tokens` did not load are kept). On an M5 Ultra (macOS 27.0.1) every function scores cosine ≥ 0.99990; the same functions rebuilt with the usual attention layout score 0.89–0.96 on the Neural Engine and fail (see [Design](#design)).

A package without readable self-test data, or without a valid `manifest.json` declaring every function this build loads, does not load (`doctor` shows why). `--selftest skip` serves every function unchecked; it is meant for diagnosing a package, not for use. Each load also checks every file against the manifest's sizes and hashes `tokenizer.json`, `selftest.json` and `selftest.bin`, which the self-test cannot judge (it feeds token ids); `aneembed models verify` hashes everything and reports whether each compiled model is present and complete (a compile is not byte-for-byte reproducible, so its hash is informational).

## Known pitfalls

- **Graph-constant masks miscompile on the ANE.** The sliding-window mask of Gemma's local layers is an input (`local_bias`, or the `band_bias` of the windowed functions), not a constant in the graph.
- **Variable shapes fall back to the CPU.** Every function has fixed shapes (64…1280 tokens with full attention, 1536 and 2048 windowed and packable, 3072…8192 windowed for one text); several short texts are packed into one call with block-diagonal attention.
- **The first compile takes minutes**, and the OS caches it per model path, macOS build and executable (the same binary copied elsewhere compiles again). `models install` does it once; after a macOS update or an upgrade of `aneembed`, `doctor` reports `stale` and `serve` compiles again before it reports healthy. Do not move the model directory.
- **Memory reporting**: the Neural Engine's buffers do not show in the process footprint (`footprint -w` lists them as `(neural)`), and freed malloc pages only become reclaimable about a minute later, so short-lived measurements overstate the process's own memory and miss the Neural Engine's.
- **llama.cpp GGUF**: text vectors are compatible; image preprocessing in llama.cpp differs from the official recipe, so image vectors are not interchangeable.

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

There is no authentication or TLS. To serve other machines, keep `--host 127.0.0.1` and put a reverse proxy in front.

## Build and test

```bash
swift build -c release
CI=true swift test                           # unit, golden and fake-runtime contract tests
Tests/Integration/release-binary-launch-smoke.sh .build/release/aneembed
swift run -c release jpeg-fuzz --iterations 200000 Tests/Fixtures/images   # JPEG decoder mutation fuzzing
ANEEMBED_TEST_MODEL_DIR=/path/to/model swift test --filter ModelTests   # real model, needs the ANE
aneembed bench --image photo.jpg --buckets --breakdown   # latency, memory, per-bucket costs
```

The test binary is another executable, so the first `ModelTests` run compiles the model for the Neural Engine again (minutes).

The model package itself is built by [`tools/convert`](tools/convert/README.md) (Python, CPU only) from the original weights at a pinned revision; a rebuild reproduces its weights, compiled program and self-test byte for byte.

## License

MIT for this code. The model package is downloaded separately and keeps its own license: EmbeddingGemma 2 by Google DeepMind, Apache-2.0, also subject to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy); the vision encoder conversion is from FluidInference/embeddinggemma-2-coreml (Apache-2.0). See [NOTICE](NOTICE).

