"""Validated evidence closures; importing never executes the recorded source."""

import hashlib
import json
import zipfile

from .contracts import Measurement, Source, encoded


def source_closure(measurements):
    sources = {m.source_id for m in measurements if m.source_id is not None}
    for measurement in measurements:
        policy = measurement.workload.get("input_policy")
        if isinstance(policy, dict) and policy.get("kind") == "frozen-shared-boundary":
            sources.add(policy["source_id"])
    return sources


def analytical_closure(store, ids):
    """Resolve declared cross-publication transfer evidence before exporting."""
    from formula_performance.records import Publication

    selected = {key: store.measurement(key) for key in ids}
    publications, owners, captures = {}, {}, {}
    for record in store.records("measurement"):
        measurement = Measurement.model_validate(record)
        artifact = measurement.artifacts.get("formula-performance")
        if artifact is None:
            continue
        publication = Publication.model_validate_json(store.blob(artifact))
        publications[measurement.measurement_id] = publication
        for observation in publication.observations:
            owners[observation.identity] = measurement
        for capture in publication.captures:
            owners[capture.identity] = measurement
            captures[capture.identity] = measurement
    pending = list(selected)
    while pending:
        publication = publications.get(pending.pop())
        if publication is None:
            continue
        for transfer in publication.transfers:
            for reference in (
                transfer.baseline_child,
                transfer.baseline_parent,
                *transfer.evidence,
            ):
                owner = owners.get(reference)
                if owner is None:
                    owner = next(
                        (m for key, m in captures.items() if reference.startswith(key + ":")), None
                    )
                if owner is None:
                    raise ValueError("incomplete analytical evidence closure: " + reference)
                if owner.measurement_id not in selected:
                    selected[owner.measurement_id] = owner
                    pending.append(owner.measurement_id)
    return list(selected.values())


def export_bundle(store, ids, destination):
    measurements = analytical_closure(store, ids)
    sources = {source_id: store.source(source_id) for source_id in source_closure(measurements)}
    blobs = {b for m in measurements for b in m.artifacts.values()}
    blobs.update(f.blob for s in sources.values() for f in s.files)
    manifest = {
        "version": 1,
        "measurements": [m.model_dump() for m in measurements],
        "sources": {key: value.model_dump() for key, value in sources.items()},
        "blobs": sorted(blobs),
        "external_requirements": "model artifacts and compatible toolchains",
    }
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", encoded(manifest))
        for blob in sorted(blobs):
            archive.writestr("blobs/" + blob, store.blob(blob))
    return {"path": str(destination), "measurements": len(measurements), "blobs": len(blobs)}


def import_bundle(store, source):
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("duplicate bundle entries")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["version"] != 1:
            raise ValueError("unsupported bundle version")
        measurements = [Measurement.model_validate(m) for m in manifest["measurements"]]
        sources = {key: Source.model_validate(s) for key, s in manifest["sources"].items()}
        for key, item in sources.items():
            if item.source_id != key:
                raise ValueError("source identity mismatch")
        required = {f.blob for s in sources.values() for f in s.files}
        required.update(b for m in measurements for b in m.artifacts.values())
        if required != set(manifest["blobs"]) or not source_closure(measurements) <= sources.keys():
            raise ValueError("incomplete evidence closure")
        if set(names) != {"manifest.json", *("blobs/" + b for b in required)}:
            raise ValueError("unexpected bundle entries")
        # Validate the entire closure before publishing any index entries.
        for blob in required:
            if hashlib.sha256(archive.read("blobs/" + blob)).hexdigest() != blob:
                raise ValueError("bundle checksum mismatch")
        for kind, values in (
            ("source", sources.items()),
            ("measurement", ((m.measurement_id, m) for m in measurements)),
        ):
            for key, value in values:
                try:
                    previous = store.get(kind, key)
                except KeyError:
                    continue
                if encoded(previous) != encoded(value):
                    raise ValueError("immutable imported identity collision")
        for blob in required:
            store.put_blob(archive.read("blobs/" + blob))
        store.put_many(
            [
                *(("source", key, item) for key, item in sources.items()),
                *(("measurement", item.measurement_id, item) for item in measurements),
            ]
        )
    return {"imported": len(measurements)}


def import_session(store, config, directory):
    from .sessions import import_session as ingest

    return ingest(store, config, directory)
