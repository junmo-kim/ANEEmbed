# Changelog

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
