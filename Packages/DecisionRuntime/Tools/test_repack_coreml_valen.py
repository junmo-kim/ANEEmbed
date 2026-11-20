import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform == "darwin", "requires the macOS Swift toolchain")
class SharedWeightsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.build.cleanup)
        cls.binary = Path(cls.build.name) / "repack"
        source = Path(__file__).with_name("repack_coreml_valen.swift")
        subprocess.run(["swiftc", str(source), "-o", str(cls.binary)], check=True)

    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.root = Path(work.name)
        self.original = self.root / "original"
        self.kit = self.root / "kit"
        self.dest = self.root / "rebuilt"
        self.original.mkdir()
        self.kit.mkdir()
        (self.original / "model.json").write_bytes(b"{}")
        (self.original / "weight.bin").write_bytes(b"weights")
        self.recipe = dict(original=hashlib.sha256(b"{}").hexdigest(),
                           base={"weight.bin": hashlib.sha256(b"weights").hexdigest()},
                           overlay={}, emptyDirectories=[], stages={})

    def repack(self):
        (self.kit / "recipe.json").write_text(json.dumps(self.recipe))
        return subprocess.run([self.binary, self.original, self.kit, self.dest],
                              capture_output=True)

    def test_shared_weights(self):
        target = "short/weights/weight.bin"
        self.recipe["sharedFiles"] = {target: "weight.bin"}
        result = self.repack()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.dest / target).read_bytes(), b"weights")
        self.assertTrue((self.dest / target).samefile(self.dest / "weight.bin"))

    def test_recipe_without_aliases(self):
        self.assertEqual(self.repack().returncode, 0)
        self.assertEqual((self.dest / "weight.bin").read_bytes(), b"weights")

    def test_invalid_alias_preserves_destination(self):
        for target, source in [("short.bin", "missing.bin"), ("../escape", "weight.bin"),
                               ("weight.bin", "weight.bin"), ("valenMLP0.fp16", "weight.bin")]:
            with self.subTest(target=target):
                self.recipe["stages"] = {"0": {}}
                self.recipe["sharedFiles"] = {target: source}
                self.assertNotEqual(self.repack().returncode, 0)
                self.assertFalse(self.dest.exists())
                self.assertFalse((self.root / "escape").exists())
                self.assertFalse(list(self.root.glob(".valen-repack-*")))


if __name__ == "__main__":
    unittest.main()
