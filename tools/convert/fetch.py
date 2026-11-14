"""Download the pinned public inputs listed in sources.json and check their sha256.

    python -I fetch.py models --dest DIR   DIR/embeddinggemma-2        google/embeddinggemma-2 snapshot (build, references)
                                           DIR/embeddinggemma-2-coreml  FluidInference vision package (build)
    python -I fetch.py data --dest DIR     DIR/<file>                   evaluation texts and images (corpus.py, images.py)

`models` uses huggingface_hub at the pinned revisions (no token needed). `data` downloads raw files from GitHub at
pinned commits and from Wikisource at pinned page revisions, and copies the files that ship inside pinned Python
packages (requirements.txt) out of the active environment. Existing files with the right sha256 are kept.
Exit status 1 if any file does not match.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
USER_AGENT = "ANEEmbed-fetch/1 (https://github.com/junmo-kim/ANEEmbed)"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def check(path, want):
    got = sha256(path) if os.path.exists(path) else "missing"
    ok = got == want
    print(f"{'ok ' if ok else 'BAD'} {path}" + ("" if ok else f" sha256 {got} != {want}"), flush=True)
    return ok


def models(sources, dest):
    from huggingface_hub import snapshot_download
    ok = True
    for m in sources["models"]:
        target = os.path.join(dest, m["dir"])
        snapshot_download(repo_id=m["id"], revision=m["revision"], local_dir=target, allow_patterns=m["allow_patterns"])
        for rel, want in m["sha256"].items():
            ok &= check(os.path.join(target, rel), want)
    return ok


def url_of(entry):
    if "url" in entry:
        return entry["url"]
    owner_repo = entry["repo"].removeprefix("https://github.com/")
    return f"https://raw.githubusercontent.com/{owner_repo}/{entry['commit']}/{entry['path']}"


def data(sources, dest):
    os.makedirs(dest, exist_ok=True)
    ok = True
    for e in sources["files"]:
        path = os.path.join(dest, e["file"])
        if not (os.path.exists(path) and sha256(path) == e["sha256"]):
            req = urllib.request.Request(url_of(e), headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r, open(path + ".part", "wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(path + ".part", path)
        ok &= check(path, e["sha256"])
    for e in sources["packaged"]:
        have = md.version(e["package"])
        if have != e["version"]:
            print(f"BAD {e['file']}: {e['package']} {have} installed, {e['version']} pinned (requirements.txt)")
            ok = False
            continue
        path = os.path.join(dest, e["file"])
        shutil.copyfile(md.distribution(e["package"]).locate_file(e["path"]), path)
        ok &= check(path, e["sha256"])
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("models", "data"))
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    sources = json.load(open(os.path.join(HERE, "sources.json"), encoding="utf-8"))
    ok = (models if a.what == "models" else data)(sources, a.dest)
    print("FETCH OK" if ok else "FETCH FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
