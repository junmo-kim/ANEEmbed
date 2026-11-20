"""CPU-only tests of the local Core ML exporter contract."""
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from coreml_valen import export_bundle, export_function, feature_scale_graph, role_graphs


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

        def operators(keys, get_tensor, table, output, graph, *, roles):
            return {"role_mixer": graph("valenScore", None, {}, ["probabilities", "statistics"]),
                    "role_pool": graph("valenPool288", None, {}, "pooled")}

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
            patch("coreml_scan_pipeline.partition_scan") as partition,
        ):
            source = Path(temp) / "checkpoint"
            source.mkdir()
            (source / "config.json").write_text('{"vision_config": {}}')
            output = Path(temp) / "bundle"
            export_bundle(source, output, small_vision=False, tiers=tiers)
            load.assert_called_once_with(source, scaled_recurrence=True, valen_recurrence=True)
            self.assertEqual([c.args[0].stem for c in partition.call_args_list], [f"scan{i}" for i in range(31)])
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
            names = {f"scan{i}" for i in range(32)} | {*attention, "valenScore", "valenPool288", "p1024"}
            self.assertNotIn("decision_head", manifest)
            self.assertNotIn("result_stats", manifest)
            self.assertEqual(manifest["coreml_functions"], {name: f"{name}.mlpackage" for name in names})
            for path in manifest["coreml_functions"].values():
                self.assertTrue((output / path).is_dir())
            self.assertFalse((output / "model.safetensors").exists())

    def test_dense_feature_scale_preserves_the_vision_abi(self):
        import torch

        rng = torch.Generator().manual_seed(749)
        x = torch.randn((1, 2560, 1, 256), generator=rng).half()
        torch.testing.assert_close(feature_scale_graph().half()(x), x * 64, rtol=0, atol=0)

    def test_role_accumulation_and_first_winner_from_masked_probabilities(self):
        import torch
        from elephant.operators import Stats

        class MixerStub(torch.nn.Module):
            def score_rows(self, rows):
                return rows[:, 2, :1]

        with patch("valen_operators.Mixer", return_value=MixerStub()):
            pool, score = role_graphs({})
        length = torch.zeros(1, 1, 1, 288)
        length[..., 0] = 5
        previous = torch.zeros(1, 2560, 1, 288)
        for count, value in [(2, 2.0), (3, 3.0)]:
            valid = torch.zeros(1, 1, 128, 288)
            valid[:, :, :count, 0] = 1
            hidden = torch.full((1, 2560, 1, 128), value)
            previous = pool(hidden, valid, length, previous)
        torch.testing.assert_close(previous[..., 0], torch.full((1, 2560, 1), 2.6))
        self.assertTrue(previous[..., 1:].eq(0).all())

        roles = torch.zeros(1, 2560, 4, 256, dtype=torch.float16)
        roles[0, 0, 2, :3] = torch.tensor([2, 2, 1])
        mask = torch.full((1, 256, 1, 1), -30000, dtype=torch.float16)
        mask[:, :3] = 0
        scalar = lambda value: torch.full((1, 1, 1, 1), value, dtype=torch.float16)
        args = (scalar(1.5), scalar(-0.5), scalar(1))
        probabilities, statistics = score(roles, mask, *args)
        self.assertEqual(probabilities.argmax(dim=1).item(), 0)
        self.assertEqual(probabilities[0, 0].item(), probabilities[0, 1].item())
        self.assertTrue(probabilities[:, 3:].eq(0).all())
        expected = Stats()(probabilities, scalar(0), *args)
        wrong = Stats()(probabilities, scalar(1), *args)
        self.assertNotEqual(expected[0, 2].item(), wrong[0, 2].item())
        torch.testing.assert_close(statistics, expected, rtol=0, atol=0)

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

class CoreMLScanPipelineTests(unittest.TestCase):
    @unittest.skipUnless(find_spec("coremltools"), "Apple coremltools required for conversion")
    def test_partition_preserves_two_outputs_and_exact_external_weights(self):
        import coremltools as ct
        import numpy as np
        import torch
        from coreml_scan_pipeline import partition_scan
        from coremltools.proto import Model_pb2

        class Scan(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.project = torch.nn.Conv2d(16, 16, 1, bias=False)
                self.update = torch.nn.Conv2d(16, 16, 1, bias=False)
                with torch.no_grad():
                    self.project.weight.copy_(torch.eye(16).reshape(16, 16, 1, 1))
                    self.update.weight.copy_(self.project.weight * 0.5)

            def forward(self, hidden_in, packed_state):
                carry = self.project(hidden_in) + packed_state
                return self.update(carry.relu()) + hidden_in, carry

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.mlpackage"
            shape = (1, 16, 1, 4)
            export_function(path, Scan(), {"hidden_in": shape, "packed_state": shape},
                            ["hidden_out", "packed_state_out"])
            partition_scan(path)
            spec = Model_pb2.Model.FromString(next(path.rglob("*.mlmodel")).read_bytes())
            self.assertEqual(len(spec.pipeline.models), 2)
            hidden = (np.arange(64, dtype=np.float16) - 32).reshape(shape)
            state = np.ones(shape, dtype=np.float16)
            model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
            result = model.predict({"hidden_in": hidden, "packed_state": state})
            np.testing.assert_array_equal(result["hidden_out"], np.maximum(hidden + state, 0) * 0.5 + hidden)
            np.testing.assert_array_equal(result["packed_state_out"], hidden + state)
