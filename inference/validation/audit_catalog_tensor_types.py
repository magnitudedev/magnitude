"""Audit GGUF tensor types in the immutable release catalog.

The audit reads only GGUF headers. Repository trees and bytes are addressed by
the commits in ``inference/catalog/models.lock.json``; no mutable branch is
consulted. The JSON output is generated evidence and belongs under
``validation/results/`` (see ``validation/AGENTS.md``).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import struct
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

GGML_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
    39: "MXFP4",
    40: "NVFP4",
    41: "Q1_0",
    42: "Q2_0",
}

# Encoding::TryFrom in inference-v4/engine/src/weights/gguf.rs at audit time.
ENGINE_ENCODINGS = {"F32", "F16", "Q8_0", "Q4_K", "Q5_K", "Q6_K", "IQ4_XS"}
SHARD = re.compile(r"^(.*)-(\d{5})-of-(\d{5})(\.gguf)$", re.IGNORECASE)


class NeedMore(Exception):
    pass


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def take(self, size: int) -> bytes:
        end = self.offset + size
        if end > len(self.data):
            raise NeedMore
        value = self.data[self.offset : end]
        self.offset = end
        return value

    def unpack(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(size))[0]

    def string(self, endian: str) -> str:
        size = self.unpack(endian + "Q")
        if size > 256 * 1024 * 1024:
            raise ValueError(f"oversized GGUF string ({size} bytes)")
        return self.take(size).decode("utf-8")


def skip_value(reader: Reader, kind: int, endian: str) -> None:
    scalar_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if kind in scalar_sizes:
        reader.take(scalar_sizes[kind])
    elif kind == 8:
        reader.string(endian)
    elif kind == 9:
        element_kind = reader.unpack(endian + "I")
        count = reader.unpack(endian + "Q")
        if count > 256 * 1024 * 1024:
            raise ValueError(f"oversized GGUF array ({count} elements)")
        for _ in range(count):
            skip_value(reader, element_kind, endian)
    else:
        raise ValueError(f"unsupported GGUF metadata type {kind}")


def parse_header(data: bytes) -> dict[str, Any]:
    reader = Reader(data)
    magic = reader.take(4)
    if magic == b"GGUF":
        endian = "<"
    elif magic == b"FUGG":
        endian = ">"
    else:
        raise ValueError("not a GGUF container")
    version = reader.unpack(endian + "I")
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")
    tensor_count = reader.unpack(endian + "Q")
    metadata_count = reader.unpack(endian + "Q")
    if tensor_count > 10_000_000 or metadata_count > 10_000_000:
        raise ValueError("implausible GGUF entry counts")
    for _ in range(metadata_count):
        reader.string(endian)
        skip_value(reader, reader.unpack(endian + "I"), endian)
    types: Counter[int] = Counter()
    for _ in range(tensor_count):
        reader.string(endian)
        dimensions = reader.unpack(endian + "I")
        if dimensions > 16:
            raise ValueError(f"implausible tensor rank {dimensions}")
        reader.take(dimensions * 8)
        types[reader.unpack(endian + "I")] += 1
        reader.take(8)  # data-section-relative offset
    return {
        "version": version,
        "tensorCount": tensor_count,
        "headerBytesParsed": reader.offset,
        "tensorTypes": [
            {
                "id": type_id,
                "name": GGML_TYPES.get(type_id, f"UNKNOWN_{type_id}"),
                "tensorCount": count,
            }
            for type_id, count in sorted(types.items())
        ],
    }


def header_url(repository: str, revision: str, path: str) -> str:
    quoted = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/{repository}/resolve/{revision}/{quoted}"


def inspect_remote_header(repository: str, revision: str, path: str) -> dict[str, Any]:
    limit = 256 * 1024 * 1024
    size = 256 * 1024
    url = header_url(repository, revision, path)
    while size <= limit:
        request = urllib.request.Request(url, headers={"Range": f"bytes=0-{size - 1}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read(size + 1)
            resolved_commit = response.headers.get("x-repo-commit")
        if resolved_commit is not None and resolved_commit != revision:
            raise ValueError(
                f"resolved commit {resolved_commit} does not match lock {revision}"
            )
        try:
            parsed = parse_header(data)
            parsed["bytesFetched"] = len(data)
            return parsed
        except NeedMore:
            size *= 2
    raise ValueError("GGUF header exceeds 256 MiB")


@dataclass(frozen=True, order=True)
class Artifact:
    repository: str
    revision: str
    path: str


def list_gguf(api: HfApi, repository: str, revision: str) -> list[str]:
    return sorted(
        entry.path
        for entry in api.list_repo_tree(
            repository,
            revision=revision,
            repo_type="model",
            recursive=True,
            expand=False,
        )
        if entry.path.lower().endswith(".gguf")
    )


def is_projector(path: str) -> bool:
    return "mmproj" in path.rsplit("/", 1)[-1].lower()


def primary_for(paths: list[str], selector: str) -> str:
    selector = selector.lower()
    matches = []
    for path in paths:
        lowered = path.lower()
        basename = lowered.rsplit("/", 1)[-1]
        shard = SHARD.match(path)
        if (
            selector in lowered
            and not is_projector(path)
            and "imatrix" not in basename
            and (shard is None or shard.group(2) == "00001")
        ):
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(
            f"format {selector} resolved to {len(matches)} primary files: {matches}"
        )
    return matches[0]


def component_paths(paths: list[str], primary: str) -> list[str]:
    match = SHARD.match(primary)
    if match is None:
        return [primary]
    prefix, _, total, suffix = match.groups()
    expected = [
        f"{prefix}-{index:05d}-of-{total}{suffix}" for index in range(1, int(total) + 1)
    ]
    missing = sorted(set(expected) - set(paths))
    if missing:
        raise ValueError(f"primary {primary} is missing shards: {missing}")
    return expected


def resolve_artifacts(
    catalog_path: Path, lock_path: Path
) -> tuple[list[Artifact], dict[Artifact, list[str]], list[dict[str, str]]]:
    catalog = json.loads(catalog_path.read_text())
    lock = json.loads(lock_path.read_text())
    model_ids = {model["id"] for model in catalog["models"]}
    if set(lock) != model_ids:
        raise ValueError("catalog lock does not exactly cover the catalog models")
    api = HfApi()
    trees: dict[tuple[str, str], list[str]] = {}

    def tree(repository: str, revision: str) -> list[str]:
        key = (repository, revision)
        if key not in trees:
            trees[key] = list_gguf(api, repository, revision)
        return trees[key]

    uses: dict[Artifact, list[str]] = defaultdict(list)
    resolutions: list[dict[str, str]] = []
    for model in catalog["models"]:
        model_id = model["id"]
        entry = lock[model_id]
        repository = model["repository"]
        revision = entry["target"]
        paths = tree(repository, revision)
        for variant in model["variants"]:
            primary = primary_for(paths, variant["format"])
            use = f"{model_id}/{variant['variantId']}:target"
            for path in component_paths(paths, primary):
                uses[Artifact(repository, revision, path)].append(use)
            resolutions.append(
                {
                    "modelId": model_id,
                    "variantId": variant["variantId"],
                    "role": "target",
                    "repository": repository,
                    "revision": revision,
                    "primary": primary,
                }
            )
            projector = model.get("projector")
            if projector is not None:
                projector_path = projector["path"]
                if projector_path not in paths:
                    raise ValueError(f"{model_id} projector {projector_path} is absent")
                uses[Artifact(repository, revision, projector_path)].append(
                    f"{model_id}/{variant['variantId']}:projector"
                )
        speculative = model.get("speculativeDecoding")
        if speculative is not None and speculative["draft"]["type"] == "file":
            draft = speculative["draft"]
            draft_repository = draft.get("repository", repository)
            draft_revision = entry["speculativeDraft"]
            draft_paths = tree(draft_repository, draft_revision)
            draft_path = draft["path"]
            if draft_path not in draft_paths:
                raise ValueError(f"{model_id} draft {draft_path} is absent")
            for path in component_paths(draft_paths, draft_path):
                uses[Artifact(draft_repository, draft_revision, path)].append(
                    f"{model_id}:draft"
                )
            resolutions.append(
                {
                    "modelId": model_id,
                    "variantId": "all",
                    "role": "draft",
                    "repository": draft_repository,
                    "revision": draft_revision,
                    "primary": draft_path,
                }
            )
    return sorted(uses), uses, resolutions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repository_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--catalog",
        type=Path,
        default=repository_root / "inference/catalog/models.json",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=repository_root / "inference/catalog/models.lock.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results/catalog-tensor-types.json",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    catalog_bytes = args.catalog.read_bytes()
    lock_bytes = args.lock.read_bytes()
    catalog = json.loads(catalog_bytes)
    artifacts, uses, resolutions = resolve_artifacts(args.catalog, args.lock)
    records: list[dict[str, Any]] = []

    def inspect(artifact: Artifact) -> dict[str, Any]:
        base = {
            "repository": artifact.repository,
            "revision": artifact.revision,
            "path": artifact.path,
            "uses": sorted(uses[artifact]),
        }
        try:
            header = inspect_remote_header(
                artifact.repository, artifact.revision, artifact.path
            )
            return {**base, "status": "verified", **header}
        except (OSError, TimeoutError, UnicodeError, ValueError, struct.error) as error:
            # Preserve partial evidence instead of hiding unavailable artifacts.
            return {
                **base,
                "status": "unavailable",
                "error": f"{type(error).__name__}: {error}",
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for completed, record in enumerate(pool.map(inspect, artifacts), 1):
            records.append(record)
            print(
                f"[{completed}/{len(artifacts)}] {record['status']} "
                f"{record['repository']}::{record['path']}",
                file=sys.stderr,
            )

    observed = sorted(
        {
            tensor_type["name"]
            for record in records
            if record["status"] == "verified"
            for tensor_type in record["tensorTypes"]
        }
    )
    unknown = sorted(name for name in observed if name.startswith("UNKNOWN_"))
    report = {
        "schemaVersion": 1,
        "scope": {
            "catalog": str(args.catalog.relative_to(repository_root)),
            "lock": str(args.lock.relative_to(repository_root)),
            "catalogSha256": hashlib.sha256(catalog_bytes).hexdigest(),
            "lockSha256": hashlib.sha256(lock_bytes).hexdigest(),
            "modelCount": len(catalog["models"]),
            "variantCount": sum(len(model["variants"]) for model in catalog["models"]),
            "resolvedArtifactCount": len(artifacts),
        },
        "method": "immutable Hugging Face repository listings plus HTTP byte-range reads of GGUF headers",
        "engineEncodingBaseline": sorted(ENGINE_ENCODINGS),
        "observedTensorTypes": observed,
        "d11MissingTensorTypes": sorted(set(observed) - ENGINE_ENCODINGS),
        "unknownTensorTypes": unknown,
        "verifiedArtifactCount": sum(
            record["status"] == "verified" for record in records
        ),
        "unavailableArtifactCount": sum(
            record["status"] != "verified" for record in records
        ),
        "resolutions": resolutions,
        "artifacts": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "scope",
                    "observedTensorTypes",
                    "d11MissingTensorTypes",
                    "verifiedArtifactCount",
                    "unavailableArtifactCount",
                )
            },
            indent=2,
        )
    )
    return 1 if report["unavailableArtifactCount"] or unknown else 0


if __name__ == "__main__":
    raise SystemExit(main())
