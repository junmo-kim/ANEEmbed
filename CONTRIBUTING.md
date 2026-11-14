# Contributing to ANEEmbed

ANEEmbed is a **single-binary, zero-dependency** Swift tool that runs EmbeddingGemma 2 on the Apple
Neural Engine only. Staying small and dependency-free is a core goal, so changes that add third-party
packages, GPU or CPU inference paths, or other models are unlikely to land without a strong reason.

## Build and test

```bash
swift build -c release
swift test                                                  # must stay green
Tests/Integration/release-binary-launch-smoke.sh .build/release/aneembed
ruby Tests/Integration/release-workflow-contract.rb
```

Use Xcode 16 or later; the deployment target is macOS 15 (the first release with multifunction Core ML
models). APIs newer than macOS 15 must be behind `@available`.

What runs where:

| Tests | Needs | CI |
|---|---|---|
| `EmbedCoreTests`: JSON, tokenizer rules and golden ids, token table, packing, bias/pool builders, image preprocessing, config, store | nothing | yes |
| `ServiceTests`: service, HTTP/OpenAI contract, QUERY, MCP through the fake runtime | nothing | yes |
| `ModelTests`: real model on the Neural Engine | a model package (`ANEEMBED_TEST_MODEL_DIR` or an installed model) | skipped (`CI` is set) |
| `Tests/Integration/*.sh`, `serve-contract.py`: the built binary over real sockets, fake runtime | `python3` | yes |

`serve-contract.py` also runs against a real server: `python3 -I Tests/Integration/serve-contract.py
http://127.0.0.1:18180 [--reference ref.json]`.

## Golden data

Test fixtures are generated, never hand-edited, and contain no third-party text or images.

```bash
uv venv /tmp/aneembed-golden
VIRTUAL_ENV=/tmp/aneembed-golden uv pip install tokenizers transformers torch torchvision pillow numpy
/tmp/aneembed-golden/bin/python -I scripts/golden/make_tokenizer_golden.py <tokenizer.json> Tests/Fixtures/tokenizer-golden.jsonl.xz
/tmp/aneembed-golden/bin/python -I scripts/golden/make_image_golden.py Tests/Fixtures/images
```

For a larger local tokenizer check (not stored), write any corpus as one JSON string per line, then:

```bash
/tmp/aneembed-golden/bin/python -I scripts/golden/make_tokenizer_golden.py <tokenizer.json> /tmp/extra.jsonl.xz --lines corpus.jsonl
ANEEMBED_EXTRA_GOLDEN=/tmp/extra.jsonl.xz swift test --filter matchesExtraGoldenCorpus
```

## Pull requests

- One logical change per PR; keep the diff focused, with tests for behaviour changes.
- Commit subjects are short imperative sentences ("Add ...", "Fix ..."); the body says why.
- Performance changes come with a measurement (before/after, machine, command).

## Releasing

A release is cut by pushing a `vX.Y.Z` tag. `.github/workflows/release.yml` tests and builds the stripped
binary on macOS 26, verifies the same packaged tarball (checksum, contents, signature, version, smoke
test) on macOS 15, and only then publishes one `aneembed-vX.Y.Z-macos-arm64.tar.gz` with its `.sha256`.

1. Bump `aneembedVersion` in `Sources/EmbedCore/Support/Version.swift`.
2. Rename `## Unreleased` to `## vX.Y.Z` in `CHANGELOG.md`; that section becomes the release-notes keynote.
3. Commit, tag and push the tag.
4. After the release succeeds, update `junmo-kim/homebrew-tap`'s `Formula/aneembed.rb` to the published
   tarball URL and SHA-256, and check `brew upgrade junmo-kim/tap/aneembed` and `aneembed --version`.
