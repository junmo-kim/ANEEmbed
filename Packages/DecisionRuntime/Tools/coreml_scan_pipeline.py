"""Make a fixed-shape Valen recurrent scan into two FP16 Core ML programs."""
import math


def partition_scan(package):
    from coremltools.proto import Model_pb2, MIL_pb2, FeatureTypes_pb2
    from coremltools.libmilstoragepython import _BlobStorageReader, _BlobStorageWriter

    descriptor, = package.rglob("*.mlmodel")
    model = Model_pb2.Model.FromString(descriptor.read_bytes())
    function = model.mlProgram.functions["main"]
    block = function.block_specializations[function.opset]
    if (set(v.name for v in function.inputs) != {"hidden_in", "packed_state"}
            or set(block.outputs) != {"hidden_out", "packed_state_out"}):
        raise ValueError("Expected a Valen linear scan with FP16 carry")
    values = {v.name: v for v in function.inputs}
    definitions = {name: -1 for name in values}
    constants = {}
    ops = list(block.operations)
    uses = []
    executable = []
    for i, op in enumerate(ops):
        if op.blocks:
            raise ValueError("Expected a flat scan program")
        uses.append({binding.name for arg in op.inputs.values() for binding in arg.arguments
                     if binding.WhichOneof("binding") == "name"})
        for value in op.outputs:
            values[value.name] = value
            definitions[value.name] = i
            if op.type == "const":
                constants[value.name] = i
        if op.type != "const":
            executable.append(i)

    def transfer_cost(cut):
        needed = set(block.outputs).union(*(uses[i] for i in executable if i >= cut))
        carried = [values[name].type.tensorType for name in needed
                   if name not in constants and definitions[name] < cut]
        if any(t.dataType != MIL_pb2.FLOAT16 for t in carried):
            return math.inf
        return sum(math.prod(d.constant.size for d in t.dimensions) for t in carried)

    # Within the middle of the scan, pick the cut with the fewest FP16 values
    # crossing it. This keeps each part independently compilable on the ANE.
    middle = len(executable) / 2
    cuts = [(transfer_cost(pos), abs(j - middle), pos) for j, pos in enumerate(executable)
            if abs(j - middle) <= max(10, 0.35 * middle) and j > 0]
    cost, _, cut = min(cuts)
    if not math.isfinite(cost):
        raise ValueError("No fixed FP16 scan boundary")

    weights = descriptor.parent / "weights"
    original = weights / "weight.bin"
    reader = _BlobStorageReader(str(original))
    result = Model_pb2.Model()
    result.specificationVersion = model.specificationVersion
    result.description.CopyFrom(model.description)
    for part_index, (start, end) in enumerate(((0, cut), (cut, len(ops)))):
        active = [i for i in executable if start <= i < end]
        needed = set().union(*(uses[i] for i in active))
        included = set(active) | {constants[n] for n in needed if n in constants}
        inputs = sorted(n for n in needed if n not in constants and definitions[n] < start)
        downstream = set(block.outputs).union(*(uses[i] for i in executable if i >= end))
        outputs = [v.name for i in active for v in ops[i].outputs if v.name in downstream]
        part = result.pipeline.models.add()
        part.CopyFrom(model)
        fn = part.mlProgram.functions["main"]
        body = fn.block_specializations[fn.opset]
        del fn.inputs[:]
        fn.inputs.extend(values[n] for n in inputs)
        del body.operations[:]
        body.operations.extend(ops[i] for i in sorted(included))
        del body.outputs[:]
        body.outputs.extend(outputs)
        for field, names in (("input", inputs), ("output", outputs)):
            part.description.ClearField(field)
            for name in names:
                tensor = values[name].type.tensorType
                if tensor.dataType != MIL_pb2.FLOAT16:
                    raise ValueError(f"Unexpected boundary type for {name}")
                feature = getattr(part.description, field).add()
                feature.name = name
                array = feature.type.multiArrayType
                array.dataType = FeatureTypes_pb2.ArrayFeatureType.FLOAT16
                array.shape.extend(d.constant.size for d in tensor.dimensions)

        # A compiled pipeline bundles each child's referenced blob separately.
        # Give each child only the weights it uses, preserving their bits.
        filename = f"part{part_index}.bin"
        writer = _BlobStorageWriter(str(weights / filename))
        offsets = {}
        for op in body.operations:
            for value in op.attributes.values():
                if not value.HasField("blobFileValue"):
                    continue
                ref = value.blobFileValue
                if ref.fileName != "@model_path/weights/weight.bin":
                    raise ValueError("Unexpected source weight file")
                old = ref.offset
                if old not in offsets:
                    dtype = MIL_pb2.DataType.Name(value.type.tensorType.dataType)
                    method = {"FLOAT16": "fp16", "FLOAT32": "float", "INT32": "int32", "UINT16": "uint16"}[dtype]
                    data = getattr(reader, "read_" + method + "_data")(old)
                    offsets[old] = getattr(writer, "write_" + method + "_data")(data)
                ref.offset = offsets[old]
                ref.fileName = "@model_path/weights/" + filename
        del writer
    descriptor.write_bytes(result.SerializeToString(deterministic=True))
    original.unlink()
