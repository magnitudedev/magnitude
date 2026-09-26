"""One-time v1 conversion. Originals remain byte-for-byte in archive/schema-1."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict

from magnitude_engine.components import ComponentId, component_id
from magnitude_engine.models.embeddings.resident import ResidentEmbedding
from magnitude_engine.models.experts.computation import ResidentExperts
from performance.facts import (
    Configuration,
    NeuralParameters,
    RecurrentStorage,
    TensorFacts,
    WeightUse,
)
from performance.records import CompositionOrigin, digest
from performance.theory.catalog import parameter_type


def convert(original: dict) -> dict:
    if original["schema_version"] != 1:
        raise ValueError("migration expects schema 1")
    if original.get("checksum") != digest({k: v for k, v in original.items() if k != "checksum"}):
        raise ValueError("original checksum mismatch")
    run = json.loads(json.dumps(original))
    notes = {}
    graph = run["assembly"]
    for path, node in graph["nodes"].items():
        binding = ComponentId(node["implementation"])
        raw = node.get("parameters") or {}
        kind = parameter_type(binding.kind)
        try:
            if kind is Configuration:
                facts = Configuration(
                    settings={
                        k: v
                        for k, v in raw.items()
                        if isinstance(v, (str, int, float, bool)) or v is None
                    }
                )
            elif kind is NeuralParameters:
                use = (
                    WeightUse.EMBEDDING
                    if binding.kind == component_id(ResidentEmbedding).kind
                    else WeightUse.EXPERTS
                    if binding.kind == component_id(ResidentExperts).kind
                    else WeightUse.FULL
                )
                facts = NeuralParameters(
                    arrays=raw.get("arrays", {}), weight_use=use, top_k=raw.get("top_k")
                )
                notes[path] = (
                    "Recorded tensor extents retained; untyped matrix arithmetic not promoted."
                )
            elif kind is RecurrentStorage:
                facts = RecurrentStorage(
                    layouts=tuple(
                        tuple(
                            TensorFacts(identity=f"{path}:{i}:{j}", dtype="recorded", **v)
                            for j, v in enumerate(row)
                        )
                        for i, row in enumerate(raw["layouts"])
                    )
                )
            else:
                facts = kind.model_validate(
                    {k: v for k, v in raw.items() if k in kind.model_fields}
                )
            node["parameters"] = facts.model_dump(mode="json")
        except (ValueError, KeyError, TypeError) as error:
            node["parameters"] = None
            notes[path] = f"Historical only: typed parameters cannot be established: {error}"
        # Old reflective provenance cannot assert equivalence with new typed capture.
        node["source"] = digest({"schema_1_source": node["source"], "original_parameters": raw})
    root = graph["nodes"][graph["root"]]["implementation"]
    model = next(
        (
            n["implementation"].split(":")[1]
            for n in graph["nodes"].values()
            if n["implementation"].split(":")[1] in ("QWEN35", "GEMMA4")
        ),
        "MLX_VLM",
    )
    scope = (
        "engine"
        if root.startswith("ENGINE:")
        else "upstream"
        if root.startswith("MODEL:FORWARD:")
        else "model"
    )
    graph["origin"] = asdict(CompositionOrigin(model, scope, selection="historical"))
    run.update(
        schema_version=2,
        id=digest({"schema_1_run": original["id"], "typed_assembly": graph})[:32],
        migration={
            "source_id": original["id"],
            "source_checksum": original["checksum"],
            "notes": notes,
        },
    )
    run.pop("checksum", None)
    run["checksum"] = digest(run)
    return run


def migrate(store) -> dict:
    from performance.store import atomic, locked, validate_run

    with locked(store.root):
        archive = store.root / "archive" / "schema-1"
        originals = {p.parent.name: p for p in archive.glob("*/run.json")}
        originals.update(
            {
                p.parent.name: p
                for p in (store.root / "runs").glob("*/run.json")
                if json.loads(p.read_text()).get("schema_version") == 1
            }
        )
        prepared = []
        for identity, path in originals.items():
            original = json.loads(path.read_text())
            if original["status"] == "running":
                raise ValueError("recover unfinished schema-1 journals before migration")
            converted = convert(original)
            validate_run(converted)
            prepared.append((identity, path, converted))
        for identity, path, converted in prepared:
            destination = archive / identity
            if path.parent != destination:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path.parent), str(destination))
            out = store.root / "runs" / converted["id"]
            if not out.exists():
                shutil.copytree(destination, out)
                atomic(out / "run.json", converted)
        audit = {
            "schema_version": 2,
            "converted": [
                {
                    "original": str(archive / identity / "run.json"),
                    "run": value["id"],
                    "checksum": value["checksum"],
                    "notes": value["migration"]["notes"],
                }
                for identity, _, value in prepared
            ],
        }
        atomic(archive / "migration.json", audit)
        store._publish()
        return audit
