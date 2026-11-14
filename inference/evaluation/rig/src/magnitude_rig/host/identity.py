"""The identity of a model file on a host: its SHA-256, the one definition of a file's digest for the evaluation
(the references' `digest.sha256_file` is this), and what is recorded of it beside the file. Two files of the same
size and metadata but other tensor bytes are other models, so no identity is less than the bytes."""

from __future__ import annotations

import hashlib
import json
import os
import re

# Beside a set, what an earlier hashing of its files left: each file's size and mtime and the digest it had then.
REMEMBERED = ".sha256.json"
HEX = re.compile(r"^[0-9a-f]{64}$")


class Refused(Exception):
    """A file, or a record of it, cannot be taken as it stands: the message names the path and why."""


def sha256(path) -> str:
    """SHA-256 of the file's bytes, lowercase hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: str) -> dict:
    """The JSON object at `path`; refused, naming it, where it is not one."""
    try:
        with open(path) as file:
            value = json.load(file)
    except (OSError, ValueError) as failure:
        raise Refused(f"{path}: unreadable record: {failure}") from failure
    if not isinstance(value, dict):
        raise Refused(f"{path}: a record is a JSON object, not {type(value).__name__}")
    return value


def remembered_sha256(path: str) -> str:
    """The file's SHA-256, from the record beside it where its size and mtime are those it was hashed at, else
    computed once and recorded there (a directory the host cannot write is hashed again each time: a set of
    hundreds of GB is read once where it can be recorded). A corrupt record is refused, never rewritten over."""
    record = os.path.join(os.path.dirname(path), REMEMBERED)
    known = _record(record) if os.path.exists(record) else {}
    stat, name = os.stat(path), os.path.basename(path)
    found = known.get(name)
    if found is not None:
        if not (isinstance(found, dict) and isinstance(found.get("sha256"), str) and HEX.match(found["sha256"])
                and isinstance(found.get("size"), int) and isinstance(found.get("mtime_ns"), int)):
            raise Refused(f"{record}: its entry for {name} is not {{size, mtime_ns, sha256}}: {found!r}")
        if (found["size"], found["mtime_ns"]) == (stat.st_size, stat.st_mtime_ns):
            return found["sha256"]
    digest = sha256(path)
    known[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}
    try:
        with open(record + ".partial", "w") as file:
            json.dump(known, file)
        os.replace(record + ".partial", record)
    except OSError:
        pass
    return digest


def recorded_sha256(path) -> str | None:
    """The SHA-256 that the provenance record beside the file (`<file>.json`, its extension replaced) names, where
    it has one: `catalog-tool slice` (or its `slice-record`) is the one writer of the format, which tells a layer
    slice by it. Refused where the record is unreadable or names no well-formed SHA-256."""
    record = os.path.splitext(os.fspath(path))[0] + ".json"
    if not os.path.exists(record):
        return None
    named = _record(record).get("sha256")
    if not (isinstance(named, str) and HEX.match(named)):
        raise Refused(f"{record}: its provenance record names no SHA-256: {named!r}")
    return named
