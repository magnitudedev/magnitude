"""`checkpoint`: the acquisition object of a whole GGUF set the host already holds, for `kl --checkpoint`
(`instruments/kl.py`), written for the run's jobs:

1. the set: the shards `<stem>-NNNNN-of-MMMMM.gguf` of the directory named (a lone `<stem>.gguf` is a set of one),
   every one of them; refused, naming the path, where the directory or a shard is absent. Nothing is fetched,
   converted or copied: the set is read where it lies, streamed by the harness;
2. its identity: each shard's SHA-256 of its bytes, hashed once on the host and recorded beside the set by size
   and mtime (`identity`). Headers alone would not tell two sets of other tensor bytes apart. What the set
   claims of its bytes is verified against them, never taken in their place: a shard's provenance record
   (`identity.recorded_sha256`, catalog-tool's per-file record of a slice) must name what that shard hashes to,
   or the set is refused;
3. the acquisition object names the set as a model of its own content (`model`): the references cache a
   reference by that name, so no reference of one set judges another. Its `target_sha256` is the first shard's
   own, as a catalog acquisition's is (`acquire`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

from .identity import Refused, recorded_sha256, remembered_sha256

# A shard, or a lone file: its set's name, and its place in the set.
SHARD = re.compile(r"^(?P<stem>.+?)(?:-(?P<index>\d{5})-of-(?P<count>\d{5}))?\.gguf$")


def shards(directory: str) -> list[str]:
    """The shards of the set in `directory`, in order, each present; refused where it holds none or lacks one. A
    projector (`mmproj-*`) is no part of a set."""
    if not os.path.isdir(directory):
        raise Refused(f"{directory}: not a directory holding a GGUF set")
    found = {}
    for name in sorted(os.listdir(directory)):
        if (match := SHARD.match(name)) and not name.startswith("mmproj"):
            found.setdefault((match["stem"], int(match["count"] or 1)), {})[int(match["index"] or 1)] = name
    if len(found) != 1:
        raise Refused(f"{directory}: holds {len(found)} GGUF sets, not one")
    ((stem, count), present), = found.items()
    missing = sorted(set(range(1, count + 1)) - set(present))
    if missing:
        raise Refused(f"{directory}: {stem} lacks shards {missing} of {count}")
    return [os.path.join(directory, present[index]) for index in range(1, count + 1)]


def identity(paths: list[str]) -> str:
    """The set's identity: its shards' names, sizes and SHA-256 of their bytes in order, hashed together; refused
    where a shard's provenance record names other bytes than its own."""
    digest = hashlib.sha256()
    for path in paths:
        content = remembered_sha256(path)
        recorded = recorded_sha256(path)
        if recorded not in (None, content):
            raise Refused(f"{path}: its bytes hash to sha256 {content}, but its provenance record names {recorded}")
        digest.update(f"{os.path.basename(path)}\0{os.path.getsize(path)}\0{content}\0".encode())
    return digest.hexdigest()


def model_of(directory: str) -> tuple[str, str, list[str]]:
    """The set in `directory` as a model of its own content: its name, its identity and its shards."""
    paths = shards(directory)
    digest = identity(paths)
    stem = SHARD.match(os.path.basename(paths[0]))["stem"]
    return f"{stem.lower()}:checkpoint-{digest[:16]}", digest, paths


def acquire(directory: str, output: str) -> dict:
    """The set's acquisition object, written to `output`: its target the first shard, which opens the rest, and
    `target_sha256` that file's SHA-256, as a catalog acquisition's is, so the target's own provenance record
    (`identity.recorded_sha256`) tells whether it is a slice; the set's identity is in its model name."""
    model, digest, paths = model_of(directory)
    artifacts = {"model": model, "target": paths[0], "target_sha256": remembered_sha256(paths[0]), "drafter": None,
                 "projector": None}
    with open(output, "w") as file:
        json.dump(artifacts, file)
    print("checkpoint %s: %d shards, sha256 %s" % (directory, len(paths), digest), flush=True)
    return artifacts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="magnitude_rig.host.checkpoint", description=__doc__.splitlines()[0])
    parser.add_argument("--path", required=True)
    parser.add_argument("--output", help="write the acquisition object here; without it, print the set's model name")
    arguments = parser.parse_args(argv)
    try:
        if arguments.output is None:
            print(model_of(arguments.path)[0])
        else:
            acquire(arguments.path, arguments.output)
    except Refused as failure:
        print("checkpoint: %s" % failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
