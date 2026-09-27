"""Portable boundary snapshots with explicit restoration against a production trace."""

import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from ..binding import Binding
from ..tensor.graph import _stable
from .evidence import fingerprint
from .fixtures import FixtureTensor, FormulaFixture


def save_boundary(boundary: FormulaFixture, path: Path, *, provenance: dict) -> None:
    """Persist values, not live GPU handles or executable Python objects."""
    payloads = {}
    ports = []
    for identity, tensor in boundary.inputs.items():
        entry = {
            "port": identity,
            "spec": fingerprint(_stable(tensor.spec)),
            "identity": tensor.identity,
        }
        if isinstance(tensor.physical, Binding):
            entry["binding"] = tensor.physical.value_identity
        else:
            data = io.BytesIO()
            np.save(data, tensor.reference, allow_pickle=False)
            key = f"{identity}.npy"
            payloads[key] = data.getvalue()
            entry["reference"] = key
            if isinstance(tensor.physical, bytes):
                key = f"{identity}.bin"
                payloads[key] = tensor.physical
                entry["physical"] = key
        ports.append(entry)
    metadata = {
        "version": 1,
        "boundary": boundary.identity,
        "semantics": boundary.isolated.target.semantic_identity,
        "graph": boundary.isolated.target.graph.fingerprint,
        "occurrence": boundary.isolated.target.call.occurrence,
        "ports": ports,
        "provenance": provenance,
        "hashes": {key: hashlib.sha256(value).hexdigest() for key, value in payloads.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("manifest.json", json.dumps(metadata, allow_nan=False))
            for name, value in payloads.items():
                archive.writestr(name, value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_boundary(fixture, target, path: Path) -> tuple[FormulaFixture, dict]:
    """Restore a fixed experimental fixture; never claim fresh upstream execution."""
    from .fixtures import encode_dense, tensor_identity

    isolated = target.isolate()
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("manifest.json"))
        if (
            metadata["version"] != 1
            or metadata["semantics"] != target.semantic_identity
            or metadata["graph"] != target.graph.fingerprint
            or metadata["occurrence"] != target.call.occurrence
        ):
            raise ValueError("retained boundary does not match the selected production trace")
        for key, digest in metadata["hashes"].items():
            if hashlib.sha256(archive.read(key)).hexdigest() != digest:
                raise ValueError("retained boundary content checksum mismatch")
        entries = {item["port"]: item for item in metadata["ports"]}
        if set(entries) != {port.local for port in isolated.inputs}:
            raise ValueError("retained boundary ports do not match")
        inputs = {}
        for port in isolated.inputs:
            entry = entries[port.local]
            if entry["spec"] != fingerprint(_stable(port.spec)):
                raise ValueError("retained boundary specification changed")
            if "binding" in entry:
                tensor = fixture._tensor(port.original)
                if (
                    not isinstance(tensor.physical, Binding)
                    or tensor.physical.value_identity != entry["binding"]
                ):
                    raise ValueError("retained immutable artifact binding changed")
                if tensor.identity != entry["identity"]:
                    raise ValueError("retained artifact values changed")
            else:
                reference = np.load(
                    io.BytesIO(archive.read(entry["reference"])), allow_pickle=False
                )
                if tuple(reference.shape) != port.spec.shape:
                    raise ValueError("retained tensor shape changed")
                reference.flags.writeable = False
                physical = (
                    archive.read(entry["physical"])
                    if "physical" in entry
                    else encode_dense(reference, port.spec)
                )
                if physical != encode_dense(reference, port.spec):
                    raise ValueError("retained physical bytes disagree with reference values")
                if tensor_identity(port.spec, reference, physical) != entry["identity"]:
                    raise ValueError("retained tensor identity mismatch")
                tensor = FixtureTensor(port.spec, reference, physical, entry["identity"])
            inputs[port.local] = tensor
    expected = hashlib.sha256(
        json.dumps(
            tuple((port.local, inputs[port.local].identity) for port in isolated.inputs),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if expected != metadata["boundary"]:
        raise ValueError("retained boundary identity mismatch")
    return FormulaFixture(isolated, inputs, expected), metadata["provenance"]
