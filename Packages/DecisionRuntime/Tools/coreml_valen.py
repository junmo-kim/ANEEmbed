"""Export the Valen device ABI as named, fixed-shape Core ML programs."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile


TIERS = (256, 512, 1024, 2048, 4096)


def register_frontend_lowerings():
    """Lower fixed axis moves and PyTorch scalar clamp promotion in the converter."""
    import numpy as np

    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch import register_torch_op
    from coremltools.converters.mil.frontend.torch.ops import matmul as default_matmul
    from coremltools.converters.mil.frontend.torch.ops import maximum, minimum
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY

    @register_torch_op(override=True)
    def clamp_max(context, node):
        minimum(context, node)

    @register_torch_op(override=True)
    def clamp_min(context, node):
        maximum(context, node)

    @register_torch_op(override=True)
    def matmul(context, node):
        x, y = (context[name] for name in node.inputs)
        if (x.shape != (16, 16) or x.val is None or len(y.shape) != 4
                or y.shape[-2:] != (16, 1) or not np.array_equal(x.val, np.tri(16))):
            default_matmul(context, node)
            return
        # The triangular recurrence prefix is a batch of 16-value dot
        # products. MIL's broadcast matrix-vector form can place it on CPU.
        rows = mb.reshape(x=y, shape=(-1, 16))
        prefix = mb.linear(x=rows, weight=x)
        context.add(mb.reshape(x=prefix, shape=y.shape, name=node.name))

    if _TORCH_OPS_REGISTRY.get_func("movedim") is not None:
        return

    @register_torch_op
    def movedim(context, node):
        tensor, source, destination = (context[name] for name in node.inputs)
        rank = len(tensor.shape)
        axes = list(range(rank))
        axes.insert(int(destination.val) % rank, axes.pop(int(source.val) % rank))
        context.add(mb.transpose(x=tensor, perm=axes, name=node.name))


def export_function(path, module, inputs, outputs):
    """Convert a first-party graph without loading a prediction runtime."""
    import coremltools as ct
    import numpy as np
    import torch

    register_frontend_lowerings()
    with torch.no_grad():
        # Core ML lowers the FP32 trace to FP16 as a single pass. Mixing FP16
        # PyTorch bias and frontend-promoted weights can fail MIL type checking.
        samples = tuple(torch.zeros(shape) for shape in inputs.values())
        traced = torch.jit.trace(module.eval().float(), samples, check_trace=False)
        model = ct.convert(
            traced,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS26,
            inputs=[ct.TensorType(name=name, shape=shape, dtype=np.float16) for name, shape in inputs.items()],
            outputs=[ct.TensorType(name=name, dtype=np.float16) for name in outputs],
            compute_precision=ct.precision.FLOAT16,
            skip_model_load=True,
        )
    model.save(str(path))


def export_fixture(destination):
    """A two-channel learned affine fixture requiring only the Core ML SDK at test time."""
    import torch

    layer = torch.nn.Conv2d(2, 2, 1).half().eval()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[[[2]], [[-1]]], [[[1]], [[3]]]], dtype=torch.float16))
        layer.bias.copy_(torch.tensor([1, -2], dtype=torch.float16))
    export_function(destination, layer, {"hidden_in": (1, 2, 1, 2)}, ["hidden_out"])


def package_digest(package):
    """Bind the logical manifest entry to all files in its Core ML package."""
    sha = hashlib.sha256()
    for file in sorted(path for path in package.rglob("*") if path.is_file()):
        relative = file.relative_to(package).as_posix().encode()
        sha.update(len(relative).to_bytes(4, "big"))
        sha.update(relative)
        with file.open("rb") as stream:
            sha.update(hashlib.file_digest(stream, "sha256").digest())
    return sha.hexdigest()


def export_bundle(checkpoint, destination, small_vision, tiers=TIERS):
    """Produce a local, uncompiled bundle from the pinned publisher snapshot."""
    from safetensors import safe_open

    from base_coreai import stage_functions
    from checkpoint_integrity import verify_text_checkpoint
    from decision_graph import CHUNK, execution_schedule, load_weights, stage_ranges
    from decision_vision import VisionGraph, vision_inputs
    from export_base import PROFILES, copy_notices
    from prepare import tokenize_tables
    from valen_operators import ScaledVision, export_operator_graphs

    tiers = tuple(tiers)
    if not tiers or tiers[0] != 256 or tiers != tuple(sorted(set(tiers))) or not set(tiers) <= set(TIERS):
        raise ValueError("Core ML Valen tiers must be an increasing supported subset beginning at 256")
    repo, revision, _ = PROFILES["valen-4b"]
    verify_text_checkpoint(checkpoint, repo, revision)
    backbone = load_weights(checkpoint, scaled_recurrence=True, valen_recurrence=True)
    cfg = backbone.cfg
    ranges = stage_ranges(cfg)
    if len(ranges) != 32 or cfg.hidden != 2560:
        raise ValueError("Unexpected Valen stage geometry")
    vision_cfg = json.loads((checkpoint / "config.json").read_text())["vision_config"]
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        output = Path(temp) / "bundle"
        output.mkdir()
        functions = {}
        hashes = {}

        def convert(name, module, inputs, outputs):
            path = output / f"{name}.mlpackage"
            if name in functions:
                raise ValueError(f"Duplicate logical function: {name}")
            export_function(path, module, inputs, outputs)
            functions[name] = path.name
            hashes[name] = package_digest(path)
            print(name, hashes[name], flush=True)

        def asset(group, names):
            # Asset identifiers describe schedule groups. Only coreml_functions
            # points to executable files when backend is coreml.
            fingerprint = hashlib.sha256(b"".join(bytes.fromhex(hashes[name]) for name in names))
            return dict(asset=f"{group}.aimodel", functions=names, source_hash=fingerprint.hexdigest())

        def operator(name, module, inputs, result):
            convert(name, module, inputs, [result])
            return asset(name, [name])

        stages = []
        for index, (start, end) in enumerate(ranges):
            names = []
            for name, module, inputs, outputs in stage_functions(
                    backbone, start, end, index, tiers, block_mask=True):
                convert(name, module, inputs, outputs)
                names.append(name)
            stages.append(asset(f"stage{index}", names))
        embedding = "backbone.language_model.embed_tokens.weight"
        reader = backbone.reader
        with safe_open(checkpoint / reader.files[embedding], framework="pt", device="cpu") as shard:
            extras = export_operator_graphs(reader.files, reader.get_tensor, shard.get_slice(embedding), output, operator)

        vision = VisionGraph(reader, "backbone.visual.", vision_cfg)
        convert("p1024", vision, vision_inputs(vision_cfg), ["features"])
        vision_asset = asset("vision", ["p1024"])
        if small_vision:
            convert("p256", ScaledVision(vision), vision_inputs(vision_cfg, 256), ["features"])
            extras["vision_256"] = asset("vision256", ["p256"])

        tokenize_tables(checkpoint / "tokenizer.json", output)
        copy_notices("valen-4b", checkpoint, output)
        manifest = dict(
            backend="coreml", architecture="h16g", coreml_functions=functions,
            source=dict(repo=repo, revision=revision), assets=stages, context=tiers[-1], tiers=tiers,
            width=CHUNK, hidden=cfg.hidden, rotary_dim=cfg.rotary_dim, heads=cfg.heads,
            kv_heads=cfg.kv_heads, head_dim=cfg.head_dim, block_mask=True, prompts=[],
            execution=execution_schedule(ranges, None),
            stage_kinds=["attention" if cfg.layer_types[end] == "full_attention" else "linear"
                         for _, end in ranges],
            vision_asset=vision_asset["asset"], vision_functions=vision_asset["functions"],
            vision_source_hash=vision_asset["source_hash"], **extras,
        )
        (output / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
        output.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    fixture = subcommands.add_parser("fixture", help="export the small Core ML CI fixture")
    fixture.add_argument("destination", type=Path)
    bundle = subcommands.add_parser("bundle", help="export all functions from a pinned Valen checkpoint")
    bundle.add_argument("checkpoint", type=Path)
    bundle.add_argument("destination", type=Path)
    bundle.add_argument("--tiers", nargs="+", type=int, choices=TIERS, default=TIERS,
                        help="increasing set of attention tiers, beginning with 256")
    bundle.add_argument("--vision-256", action="store_true", help="also export the scaled 256-patch tier")
    args = parser.parse_args()
    if args.destination.exists():
        parser.error(f"Refusing to overwrite {args.destination}")
    if args.command == "fixture":
        export_fixture(args.destination)
    else:
        export_bundle(args.checkpoint, args.destination, args.vision_256, args.tiers)


if __name__ == "__main__":
    main()
