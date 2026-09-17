#!/usr/bin/env python3
"""Compare Rust header-only inspection with V3 GGUF and stored JSON directories.
Run in the pinned V3 Python environment. No tensor data is decoded or uploaded.
"""
import argparse
import json
import struct
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspector", type=Path, required=True)
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    from engine.platform.storage import FileSource
    from engine.weights.formats.gguf import read_directory
    from engine.models.qwen35.formats.gguf import inspect_dense, inspect_moe
    from engine.weights.identity import ArtifactIdentity
    for path in args.paths:
        result = subprocess.run([str(args.inspector), str(path)], check=True, capture_output=True, text=True)
        actual = json.loads(result.stdout)
        if path.is_dir():
            from engine.weights.formats.mlx_safetensors import MLXFormat
            from engine.models.qwen35.formats.mlx import describe
            artifact = MLXFormat(str(path))
            try:
                model = describe(artifact).model_dump(mode="json")
                model["geometry"]["activation_dtype"] = "bf16"
                assert actual["qwen_description"] == model, (path.name,"MLX Qwen roles/geometry/identity")
                assert actual["tensors"] == len(artifact.tensors)
                print(json.dumps({"path":str(path),"qwen_roles":"equal","content_identity":artifact.identity,"tensors":len(artifact.tensors),"layers":len(model["blocks"])}),flush=True)
            finally:
                artifact.close()
            continue
        with FileSource(path) as source:
            if path.suffix == ".gguf":
                directory = read_directory(source)
                inspect = inspect_dense if directory.value("general.architecture") == "qwen35" else inspect_moe
                model = inspect(directory, ArtifactIdentity("0"*64)).model_dump(mode="json")
                model.pop("artifact_identity")
                model["geometry"]["activation_dtype"] = {"bfloat16":"bf16","float16":"f16"}[model["geometry"]["activation_dtype"]]
                assert actual["qwen_description"] == model, (path.name, "Qwen roles/geometry")
                expected = directory.model_dump(mode="json")
                expected["metadata"] = {m["name"]: m["value"] for m in expected["metadata"]}
                for key, value in expected.items():
                    got = actual[key].lower() if key == "byte_order" else actual[key]
                    assert got == value, (path.name, key)
                counts = {"tensors": len(expected["tensors"]), "metadata": len(expected["metadata"]), "qwen_roles":"equal", "layers":len(model["blocks"])}
            else:
                length, = struct.unpack("<Q", source.read(0, 8))
                raw = json.loads(source.read(8, length))
                expected = []
                for name, entry in raw.items():
                    if name == "__metadata__":
                        continue
                    start, end = entry["data_offsets"]
                    expected.append(dict(name=name, dtype=entry["dtype"].lower(), shape=entry["shape"], offset=8+length+start, nbytes=end-start))
                assert actual["data_offset"] == 8+length
                assert actual["tensors"] == expected, path.name
                counts = {"tensors":len(expected)}
            assert actual["file_size"] == source.size
            print(json.dumps({"path":str(path),"file_size":source.size,"directory":"equal","reference":"V3 GGUF / Safetensors header fields",**counts}),flush=True)


if __name__ == "__main__":
    main()
