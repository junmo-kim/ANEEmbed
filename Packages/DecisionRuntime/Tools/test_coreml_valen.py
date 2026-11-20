"""CPU-only tests of the local Core ML exporter contract."""
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from coreml_valen import export_bundle, export_function


class CoreMLValenExportTests(unittest.TestCase):
    def test_bundle_maps_logical_functions_to_flat_packages_without_host_weights(self):
        tiers = [256]
        cfg = SimpleNamespace(hidden=2560, rotary_dim=64, heads=16, kv_heads=4, head_dim=256,
                              layer_types=("linear_attention",) * 31 + ("full_attention",))
        embedding = "backbone.language_model.embed_tokens.weight"
        reader = SimpleNamespace(files={embedding: "weights.safetensors"}, get_tensor=None)
        backbone = SimpleNamespace(cfg=cfg, reader=reader)

        def functions(_, start, end, i, capacity, *, block_mask):
            self.assertTrue(block_mask)
            yield f"scan{i}", None, {}, ["hidden_out", "packed_state_out"]
            if i == 31:
                for tier in capacity:
                    yield f"attend{i}c{tier}", None, {}, ["hidden_out"]

        def convert(path, module, inputs, outputs):
            path.mkdir()
            (path / "model.mlmodel").write_text(path.stem)

        def operators(keys, get_tensor, table, output, graph):
            return {"role_mixer": graph("valenMixer", None, {}, "logit")}

        with (
            tempfile.TemporaryDirectory() as temp,
            patch("checkpoint_integrity.verify_text_checkpoint"),
            patch("decision_graph.load_weights", return_value=backbone) as load,
            patch("base_coreai.stage_functions", side_effect=functions),
            patch("safetensors.safe_open"),
            patch("valen_operators.export_operator_graphs", side_effect=operators),
            patch("decision_vision.VisionGraph", return_value=None),
            patch("decision_vision.vision_inputs", return_value={}),
            patch("prepare.tokenize_tables"),
            patch("export_base.copy_notices"),
            patch("coreml_valen.export_function", side_effect=convert),
        ):
            source = Path(temp) / "checkpoint"
            source.mkdir()
            (source / "config.json").write_text('{"vision_config": {}}')
            output = Path(temp) / "bundle"
            export_bundle(source, output, small_vision=False, tiers=tiers)
            load.assert_called_once_with(source, scaled_recurrence=True, valen_recurrence=True)
            manifest = json.loads((output / "model.json").read_text())
            self.assertEqual(manifest["backend"], "coreml")
            self.assertEqual(manifest["architecture"], "h16g")
            self.assertEqual(manifest["tiers"], tiers)
            self.assertEqual(manifest["context"], 256)
            self.assertEqual(len(manifest["assets"]), 32)
            attention = [f"attend31c{tier}" for tier in tiers]
            self.assertEqual(manifest["assets"][31]["functions"], ["scan31", *attention])
            self.assertEqual(manifest["vision_asset"], "vision.aimodel")
            self.assertEqual(manifest["vision_functions"], ["p1024"])
            names = {f"scan{i}" for i in range(32)} | {*attention, "valenMixer", "p1024"}
            self.assertEqual(manifest["coreml_functions"], {name: f"{name}.mlpackage" for name in names})
            for path in manifest["coreml_functions"].values():
                self.assertTrue((output / path).is_dir())
            self.assertFalse((output / "model.safetensors").exists())

    @unittest.skipUnless(find_spec("coremltools"), "Apple coremltools required for conversion")
    def test_triangular_prefix_lowers_to_linear(self):
        import torch
        from coremltools.proto import Model_pb2

        class Prefix(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("lower", torch.ones(16, 16).tril())

            def forward(self, rate):
                return self.lower @ rate

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefix.mlpackage"
            export_function(path, Prefix(), {"rate": (2, 3, 16, 1)}, ["prefix"])
            model = Model_pb2.Model()
            model.ParseFromString(next(path.rglob("*.mlmodel")).read_bytes())
            main = model.mlProgram.functions["main"]
            ops = [op.type for block in main.block_specializations.values() for op in block.operations]
            self.assertIn("linear", ops)
            self.assertNotIn("matmul", ops)
