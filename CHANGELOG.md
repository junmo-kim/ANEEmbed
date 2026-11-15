# Changelog

## Unreleased

Second model layout ("layoutR"), which needs a new model package (`manifest_version` 2; the first package is
rejected with a message saying so). Texts of 1281–2048 tokens run in windowed packing functions and texts of
2049–8192 tokens in one windowed call each (`winl3072` … `winl8192`) instead of layer by layer: 2048 tokens
194 → 168 ms, 4096 tokens 531 → 424 ms, 8192 tokens 1388 → 1187 ms on an M5 Ultra, and the Neural Engine
buffers with `--max-tokens 8192` 21.5 → 13.3 GB. New `embed_768` and `embed_1280` sizes. Cached starts load
functions four at a time. `doctor` lists the functions a `--max-tokens` value loads; `bench --buckets` also times
the long functions. README: measured memory character, image input formats, design notes.

Stricter self-test and store checks. Only functions that pass the self-test in this run serve (an allowlist):
a function the package has no test for, or whose output has a NaN or an infinity, is not served; a package
without readable self-test data or a complete `manifest.json` does not load, unless `--selftest skip`
(`ANEEMBED_SELFTEST=skip`, unsafe) serves every function unchecked. Tolerances are per kind and a package can
only tighten them; every `selftest.json` field is bounds-checked. `/health` stays 200 but reports `degraded`,
with the functions not served and the remaining limits, and doctor keeps the records of functions a smaller
`--max-tokens` did not load. Each load checks file sizes against the manifest and hashes `tokenizer.json` and
the self-test files (0.01–0.03 s on an M4). `models install` validates `--revision` and `--repo`, takes a lock
so a second install waits, swaps the revision directory in one rename, reuses an installed revision only when
its manifest matches the expected hash, stops a download that exceeds its declared size, hashes only regular
files, and refuses `--runtime fake`; `models verify` reports the compiled models. The warm state records the
package's manifest hash. Workflow actions are pinned to commits. Empty token sequences, oversized vocabularies
and negative token ids are errors instead of traps, and vectors of very small or large values normalize.

## v0.1.0

First release. EmbeddingGemma 2 text and image embeddings on the Apple Neural Engine from one Swift binary
with no third-party dependencies: the OpenAI-compatible `/v1/embeddings` server, MCP over stdio and HTTP,
and the `embed`, `similarity`, `tokenize`, `doctor`, `models` and `bench` commands. Texts up to 2048 tokens
are packed into fixed-shape bucket calls; with `--max-tokens` up to 8192, longer texts run layer by layer in
1024-token chunks. Images use the official 280-token preprocessing, bit-exact against Hugging Face.
`models install` downloads the pinned model package (or `--from` a local copy), checks every file's SHA-256,
compiles it on the Mac and runs the package's self-test; any function the Neural Engine compiler gets
wrong is not served. Measured against the original fp32 model over HTTP on public data: cosine ≥ 0.99989
for text up to 8192 tokens and ≥ 0.99995 for images; a short query takes 5 ms and a 2048-token text 0.25 s
on an M4.
