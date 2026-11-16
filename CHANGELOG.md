# Changelog

## v0.2.0 (unreleased)

The long-text functions (`winl3072` … `winl8192`, used with `--max-tokens` above 2048) attend in blocks in their global layers and divide by a matmul instead of the Neural Engine's softmax, which loses accuracy as texts get longer. On an M4 the minimum cosine against the fp32 reference (9 texts per function, 3 for the new `winl8192`) rises from 0.99989–0.999967 to 0.999976–0.999979, and `winl8192` takes 1.24 s instead of 1.46 s at 8192 tokens (median of 5 calls); the first compile of these functions takes 1.3–1.6x longer. Texts up to 2048 tokens and images are unchanged. This needs a new model package, so it ships with the next release; see `tools/convert/README.md`.

## v0.1.1

`POST /tokenize` and `POST /detokenize` in llama-server's format (`add_special`, `parse_special`, `with_pieces`), so clients that measure or split their texts with llama-server's tokenizer API work unchanged; for EmbeddingGemma 2 the responses match llama-server's. Content is limited to 1 MiB, and invalid requests are refused with a `400` where llama-server is lenient. `docs/PERFORMANCE.md` now compares the M4, M5 Max and M5 Ultra in one table, measured the same way. See the README for details.

## v0.1.0

First release. EmbeddingGemma 2 text and image embeddings on the Apple Neural Engine from one dependency-free Swift binary: an OpenAI-compatible `/v1/embeddings` server, MCP over stdio and HTTP, and CLI commands. Nothing runs on the GPU and idle memory stays around 200 MB; texts up to 8192 tokens run entirely on the Neural Engine, and every function is checked against reference vectors before it serves. See the README for setup, limits and measurements.
