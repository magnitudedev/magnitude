"""Fetch one GGUF header for the assessment fixtures from Hugging Face,
with ranged GETs only (never the whole file), trim it and gzip it.

usage: python3 -I fetch_header.py REPO REVISION PATH

Writes headers/<file name>.gz beside this script. The header is every byte
through the aligned tensor-data offset. Trimming keeps every metadata key,
the vocabulary's length and token kinds, and the tensor directory; it
empties `tokenizer.ggml.merges` and the text and score of every normal token
no special-token key names. Assessment reads none of what is cut.
"""
import gzip
import os
import struct
import subprocess
import sys

HF = "https://huggingface.co"


def request(url, headers=None):
    headers = dict(headers or {})
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    command = ["curl", "-sSfL", "--max-time", "300"]
    for name, value in headers.items():
        command += ["-H", f"{name}: {value}"]
    return subprocess.run(command + [url], check=True, capture_output=True).stdout


def ranged(repo, revision, path, end):
    url = f"{HF}/{repo}/resolve/{revision}/{path}"
    data = request(url, {"Range": f"bytes=0-{end - 1}"})
    if len(data) > end:
        raise SystemExit("server ignored the Range header")
    return data


class Short(Exception):
    pass


def header_length(data):
    """Bytes through the aligned tensor-data offset, or Short."""
    pos = 0

    def take(n):
        nonlocal pos
        if pos + n > len(data):
            raise Short
        chunk = data[pos:pos + n]
        pos += n
        return chunk

    def u32():
        return struct.unpack("<I", take(4))[0]

    def u64():
        return struct.unpack("<Q", take(8))[0]

    def string():
        return take(u64()).decode("utf-8", "replace")

    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def value(kind):
        if kind in sizes:
            return take(sizes[kind])
        if kind == 8:
            return string()
        if kind == 9:
            inner = u32()
            count = u64()
            if inner in sizes:
                take(sizes[inner] * count)
            else:
                for _ in range(count):
                    value(inner)
            return None
        raise SystemExit(f"unknown GGUF value type {kind}")

    if take(4) != b"GGUF":
        raise SystemExit("not a GGUF file")
    u32()
    tensors = u64()
    metadata = u64()
    alignment = 32
    for _ in range(metadata):
        key = string()
        kind = u32()
        result = value(kind)
        if key == "general.alignment":
            alignment = struct.unpack("<I", result)[0]
    for _ in range(tensors):
        string()
        dims = u32()
        take(8 * dims)
        u32()
        u64()
    return (pos + alignment - 1) // alignment * alignment


def fetch(repo, revision, path):
    end = 4 << 20
    while True:
        data = ranged(repo, revision, path, end)
        try:
            length = header_length(data)
        except Short:
            if len(data) < end:
                raise SystemExit("file ends inside its header")
            end *= 2
            continue
        if length > len(data):
            end = length
            continue
        return data[:length]



SIZES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, n):
        chunk = self.data[self.pos:self.pos + n]
        if len(chunk) != n:
            raise SystemExit("truncated header")
        self.pos += n
        return chunk

    def unpack(self, fmt):
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.take(size))[0]

    def string(self):
        return self.take(self.unpack("Q")).decode("utf-8", "surrogateescape")

    def value(self, kind):
        if kind in SIZES:
            return ("scalar", kind, self.unpack(SIZES[kind]))
        if kind == 8:
            return ("scalar", kind, self.string())
        if kind == 9:
            inner = self.unpack("I")
            count = self.unpack("Q")
            if inner in SIZES:
                fmt = SIZES[inner]
                size = struct.calcsize("<" + fmt)
                raw = self.take(size * count)
                items = list(struct.unpack(f"<{count}{fmt}", raw)) if count else []
            elif inner == 8:
                items = [self.string() for _ in range(count)]
            else:
                raise SystemExit(f"nested array type {inner}")
            return ("array", inner, items)
        raise SystemExit(f"unknown value type {kind}")


def write_string(out, text):
    raw = text.encode("utf-8", "surrogateescape")
    out += struct.pack("<Q", len(raw)) + raw


def write_value(out, value):
    shape, kind, payload = value
    if shape == "scalar":
        if kind == 8:
            write_string(out, payload)
        else:
            out += struct.pack("<" + SIZES[kind], payload)
        return
    out += struct.pack("<IQ", kind, len(payload))
    if kind == 8:
        for item in payload:
            write_string(out, item)
    elif payload:
        out += struct.pack(f"<{len(payload)}{SIZES[kind]}", *payload)



def trim(header):
    reader = Reader(header)
    if reader.take(4) != b"GGUF":
        raise SystemExit("not a GGUF file")
    version = reader.unpack("I")
    tensor_count = reader.unpack("Q")
    metadata_count = reader.unpack("Q")
    metadata = []
    for _ in range(metadata_count):
        key = reader.string()
        kind = reader.unpack("I")
        metadata.append([key, kind, reader.value(kind)])
    values = {key: value for key, _, value in metadata}
    alignment = values["general.alignment"][2] if "general.alignment" in values else 32
    directory = Reader(reader.data[reader.pos:])
    for _ in range(tensor_count):
        directory.string()
        directory.take(8 * directory.unpack("I"))
        directory.unpack("I")
        directory.unpack("Q")
    tensors = directory.data[:directory.pos]

    keep = {
        value[2]
        for key, value in values.items()
        if key.startswith("tokenizer.ggml.") and key.endswith("_token_id")
    }
    if "tokenizer.ggml.eos_token_ids" in values:
        keep.update(values["tokenizer.ggml.eos_token_ids"][2])
    kinds = values.get("tokenizer.ggml.token_type")
    normal = lambda index: kinds is not None and kinds[2][index] == 1 and index not in keep
    for entry in metadata:
        key, _, value = entry
        if key == "tokenizer.ggml.merges":
            entry[2] = ("array", 8, [])
        elif key == "tokenizer.ggml.tokens":
            entry[2] = ("array", 8, ["" if normal(i) else t for i, t in enumerate(value[2])])
        elif key == "tokenizer.ggml.scores":
            entry[2] = ("array", value[1], [0.0 if normal(i) else v for i, v in enumerate(value[2])])

    out = bytearray(b"GGUF")
    out += struct.pack("<IQQ", version, tensor_count, metadata_count)
    for key, kind, value in metadata:
        write_string(out, key)
        out += struct.pack("<I", kind)
        write_value(out, value)
    out += tensors
    out += b"\0" * (-len(out) % alignment)
    return bytes(out)


def main():
    repo, revision, path = sys.argv[1:4]
    header = trim(fetch(repo, revision, path))
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "headers", os.path.basename(path) + ".gz")
    with open(output, "wb") as raw, gzip.GzipFile(os.path.basename(path), "wb", 9, raw, mtime=0) as handle:
        handle.write(header)
    print(f"{output}: {len(header)} header bytes")


main()
