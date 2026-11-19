"""Exercise publication of an AOT stage after the compiler returns a placement report."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from prepare import PartialPlacement, compile_stage


class CompilationTests(unittest.TestCase):
    def compile(self, reports):
        with TemporaryDirectory() as directory:
            dest = Path(directory) / "out.aimodelc"
            attempts = []
            def compiler(args, check):
                path = Path(args[-1])
                path.mkdir()
                attempts.append(path)
            with patch("prepare.subprocess.run", side_effect=compiler), patch("prepare.verified", side_effect=reports):
                try:
                    compile_stage(Path(directory) / "in.aimodel", dest, "h17d", "digest", max_attempts=3)
                finally:
                    self.assertEqual(len(attempts), len(set(attempts)))
                    self.assertTrue(all(not p.exists() for p in attempts))
                    self.assertEqual(dest.exists(), reports[-1] is None)
            return len(attempts)

    def test_retries_only_partial_placement_and_publishes_verified_stage(self):
        self.assertEqual(self.compile([PartialPlacement(), None]), 2)
        with self.assertRaisesRegex(PartialPlacement, "after 3 attempts"):
            self.compile([PartialPlacement()] * 3)
        with self.assertRaisesRegex(ValueError, "source hash"):
            self.compile([ValueError("source hash")])
