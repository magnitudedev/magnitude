"""Make a test-only FP32 oracle artifact from independently interpreted GGUF.

This isolates reference-runtime quantized arithmetic from model equations. The
engine never uses this file: production weights stay encoded. Conversion stages
one bounded chunk at a time through a temporary mapped tensor.
"""

import argparse
import hashlib
import importlib.metadata
import json
import tempfile
from pathlib import Path

import gguf
import numpy as np


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def convert(source: Path, destination: Path):
    reader = gguf.GGUFReader(source)
    writer = gguf.GGUFWriter(destination, reader.fields["general.architecture"].contents())
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name in ("general.architecture", "general.file_type"):
            continue
        writer.add_key_value(name, field.contents(), field.types[0], field.types[-1])
    writer.add_uint32("general.file_type", 0)
    for tensor in reader.tensors:
        writer.add_tensor_info(
            tensor.name, tuple(reversed(tensor.shape)), np.dtype("float32"), tensor.n_elements * 4
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        for index, tensor in enumerate(reader.tensors):
            mapped_path = Path(temporary) / "tensor.f32"
            decoded = np.memmap(
                mapped_path, dtype=np.float32, mode="w+", shape=(tensor.n_elements,)
            )
            block_elements, block_bytes = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
            source_blocks = tensor.data.view(np.uint8).reshape(-1, block_bytes)
            chunk_blocks = max(1, 1024**2 // (block_elements * 4))
            for start in range(0, len(source_blocks), chunk_blocks):
                chunk = gguf.dequantize(
                    source_blocks[start : start + chunk_blocks], tensor.tensor_type
                ).ravel()
                decoded[start * block_elements : start * block_elements + chunk.size] = chunk
            writer.write_tensor_data(decoded)
            decoded._mmap.close()
            mapped_path.unlink()
            if index % 32 == 0:
                print(f"{index + 1}/{len(reader.tensors)} {tensor.name}", flush=True)
    writer.close()
    record = {
        "purpose": "independent FP32 numerical oracle; never engine runtime weights",
        "source_sha256": digest(source),
        "output_sha256": digest(destination),
        "gguf_version": importlib.metadata.version("gguf"),
        "script_sha256": digest(Path(__file__)),
        "tensors": len(reader.tensors),
        "output_bytes": destination.stat().st_size,
    }
    destination.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)
