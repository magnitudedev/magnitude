"""Magnitude-owned schedule records for reconstruction before module composition.

TileLang selects the configuration through its existing autotuner. Magnitude
owns the workload and validation identity needed to retain that choice before
constructing a composed module. These records require explicit full
identities and are written only by an explicit, reference-checked calibration.
Reading a selection never constructs, compiles or benchmarks a kernel.
"""

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class SelectionIdentity:
    semantic_kernel: str
    schedule_family: str
    construction: str
    workload: str
    numerical_mode: str
    validation: str
    compiler: str
    target: str
    physical_device: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip() for value in asdict(self).values()):
            raise ValueError("schedule selection requires every identity component")

    @property
    def key(self):
        return sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class SelectedConfiguration:
    identity: SelectionIdentity
    config: dict
    latency_ms: float
    timing_basis: str


def load_selection(directory, identity: SelectionIdentity) -> SelectedConfiguration | None:
    """Read only an exact qualified identity; partial/corrupt records are misses."""
    path = Path(directory) / (identity.key + ".json")
    try:
        record = json.loads(path.read_text())
        if record["version"] != 1 or record["identity"] != asdict(identity):
            return None
        config, latency, basis = record["config"], record["latency_ms"], record["timing_basis"]
        if not isinstance(config, dict) or not config or not isinstance(basis, str) or not basis:
            return None
        if (
            not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or not math.isfinite(latency)
            or latency <= 0
        ):
            return None
        return SelectedConfiguration(identity, config, latency, basis)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def calibrate(
    tuner, directory, identity: SelectionIdentity, **run_options
) -> SelectedConfiguration:
    """Use the existing autotuner, then persist only its qualified configuration.

    The caller's construction identity covers the factory and its dependencies;
    workload covers all geometry, representations and resource facts; validation
    covers reference code, concrete fixtures and numerical acceptance settings.
    physical_device must identify the actual device and runtime/OS provenance.
    Compiler and target are additionally checked against the active tuner.
    """
    from tilelang.cache import compiler_identity

    if tuner.profile_args.skip_check or tuner.profile_args.ref_prog is None:
        raise ValueError("persistent selection requires reference validation of every candidate")
    if identity.compiler != compiler_identity() or identity.target != str(
        tuner.compile_args.target
    ):
        raise ValueError("selection identity does not match the active compiler and target")
    result = tuner.run(**run_options)
    if result.config not in tuner.configs:
        raise ValueError("selected configuration is outside the declared family")
    if result.latency is None or not math.isfinite(result.latency) or result.latency <= 0:
        raise ValueError("selected configuration has no finite positive latency")
    selected = SelectedConfiguration(
        identity, dict(result.config), result.latency, tuner.profile_args.backend
    )
    record = dict(
        version=1,
        identity=asdict(identity),
        config=selected.config,
        latency_ms=selected.latency_ms,
        timing_basis=selected.timing_basis,
    )
    contents = json.dumps(record, sort_keys=True, indent=2, allow_nan=False)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".selection-", dir=root)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / (identity.key + ".json"))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return selected
