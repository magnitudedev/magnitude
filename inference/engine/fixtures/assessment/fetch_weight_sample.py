"""Fetch one GGUF's weight sample for the service's weight-lineage tests,
with ranged GETs only: the header, then the first 64 values of each sampled
normalization tensor.

usage: python3 -I fetch_weight_sample.py SOURCE NAME
  SOURCE is hf:REPO@REVISION:PATH, ollama:MODEL@sha256:DIGEST (a model layer
  blob of registry.ollama.ai/library/MODEL) or a local file path.

Writes weight-samples/NAME.json beside this script: the sample as the
service's `WeightSample` serializes it, tensor name to values. The sampled
tensors are the one-dimensional `output_norm.weight` and `blk.0.*norm.weight`
and `blk.1.*norm.weight` tensors stored as F32, F16 or BF16; constant ones are
left out.
"""
import json
import os
import struct
import subprocess
import sys

VALUES = 64
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def location(source):
    if source.startswith("hf:"):
        repo, rest = source[3:].split("@", 1)
        revision, path = rest.split(":", 1)
        return f"https://huggingface.co/{repo}/resolve/{revision}/{path}"
    if source.startswith("ollama:"):
        model, digest = source[7:].split("@", 1)
        return f"https://registry.ollama.ai/v2/library/{model}/blobs/{digest}"
    return source


def read(where, start, length):
    if not where.startswith("https://"):
        with open(where, "rb") as file:
            file.seek(start)
            return file.read(length)
    command = ["curl", "-sSfL", "--max-time", "300", "-r", f"{start}-{start + length - 1}"]
    token = os.environ.get("HF_TOKEN")
    if token and "huggingface.co" in where:
        command += ["-H", f"Authorization: Bearer {token}"]
    data = subprocess.run(command + [where], check=True, capture_output=True).stdout
    if len(data) != length:
        raise SystemExit("server ignored the Range header")
    return data


class Reader:
    def __init__(self, where):
        self.where, self.data, self.pos = where, b"", 0

    def take(self, fmt):
        size = struct.calcsize(fmt)
        while self.pos + size > len(self.data):
            self.data += read(self.where, len(self.data), max(4 << 20, len(self.data)))
        value = struct.unpack_from("<" + fmt, self.data, self.pos)
        self.pos += size
        return value[0]

    def string(self):
        length = self.take("Q")
        return self.take(f"{length}s").decode("utf-8", "replace") if length else ""

    def value(self, kind):
        if kind in SCALAR:
            return self.take(SCALAR[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            element, count = self.take("I"), self.take("Q")
            return [self.value(element) for _ in range(count)]
        raise SystemExit(f"unknown metadata type {kind}")


def sampled(name):
    if name == "output_norm.weight":
        return True
    for block in ("blk.0.", "blk.1."):
        if name.startswith(block) and name.endswith(".weight"):
            rest = name[len(block):-len(".weight")]
            return rest.endswith("norm") and "." not in rest
    return False


def decode(kind, raw):
    if kind == 0:
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))
    halves = struct.unpack(f"<{len(raw) // 2}H", raw)
    if kind == 1:
        return [struct.unpack("<e", struct.pack("<H", h))[0] for h in halves]
    return [struct.unpack("<f", struct.pack("<I", h << 16))[0] for h in halves]


def main():
    source, name = sys.argv[1], sys.argv[2]
    where = location(source)
    header = Reader(where)
    if header.take("I") != 0x46554747:
        raise SystemExit("not a GGUF")
    header.take("I")
    tensor_count, metadata_count = header.take("Q"), header.take("Q")
    alignment = 32
    for _ in range(metadata_count):
        key = header.string()
        value = header.value(header.take("I"))
        if key == "general.alignment":
            alignment = value
    tensors = []
    for _ in range(tensor_count):
        tensor = header.string()
        dims = [header.take("Q") for _ in range(header.take("I"))]
        kind, offset = header.take("I"), header.take("Q")
        tensors.append((tensor, dims, kind, offset))
    data = (header.pos + alignment - 1) // alignment * alignment
    sample = {}
    for tensor, dims, kind, offset in tensors:
        if not sampled(tensor) or len(dims) != 1 or kind not in (0, 1, 30):
            continue
        width = 4 if kind == 0 else 2
        values = decode(kind, read(where, data + offset, min(dims[0], VALUES) * width))
        if any(value != values[0] for value in values):
            sample[tensor] = values
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weight-samples", f"{name}.json")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as file:
        json.dump(dict(sorted(sample.items())), file, separators=(",", ":"))
        file.write("\n")
    print(f"{output}: {len(sample)} tensors")


main()
