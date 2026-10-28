"""Models come only from the catalog; comparison engines name their own form of that model.

The catalog reference (``<model>:<variant>``) is the run's model. Its acquisition object
(``magnitude-service-catalog-tool acquire``) gives the target GGUF and the catalog drafter,
which every GGUF engine serves (magnitude, llama.cpp, and Ollama importing the GGUF).

The catalog holds GGUF packages only. MLX engines (mlx-vlm, oMLX) need an MLX snapshot and
Ollama's registry runners need a pulled tag, and neither has a catalog form. Those engines
therefore take one explicit artifact per run, ``--artifact ENGINE=REF`` (a pinned
``hf:owner/repo@<commit>`` snapshot, a local MLX directory, or ``ollama:[namespace/]name:tag``),
recorded in the reproduction command. It names the comparison engine's form of the catalog
model; it is never a model registry, and GGUF engines refuse it.
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Literal

from magnitude_fixtures.canonical import Record
from pydantic import Field

from .policy import INFERENCE

CATALOG = INFERENCE / "catalog" / "models.json"
Engine = Literal[
    "magnitude", "llama.cpp", "ollama", "ollama-mlx", "ollama-registry", "mlx-vlm", "omlx"
]
#: Engines that serve the catalog's GGUF target and drafter.
GGUF_ENGINES = ("magnitude", "llama.cpp", "ollama")
#: Engines that serve a model pulled into Ollama's own registry store.
REGISTRY_ENGINES = ("ollama-mlx", "ollama-registry")
#: The stored format each registry engine serves; Ollama picks its runner from it.
REGISTRY_FORMATS = {"ollama-mlx": "safetensors", "ollama-registry": "gguf"}
GGUF_MODEL_LAYER = "application/vnd.ollama.image.model"
OLLAMA_REGISTRY = "registry.ollama.ai"


def catalog_reference(reference: str) -> str:
    """Check ``<model>:<variant>`` against the catalog; model ids hold no colon."""
    model, separator, variant = reference.partition(":")
    if not separator or not variant:
        raise ValueError("--model takes a catalog reference <model>:<variant>")
    entries = {entry["id"]: entry for entry in json.loads(CATALOG.read_text())["models"]}
    if model not in entries:
        raise ValueError(f"{model} is not a catalog model")
    variants = [
        item["variantId"] for source in entries[model]["sources"] for item in source["variants"]
    ]
    if variant not in variants:
        raise ValueError(f"{model} has no variant {variant}; choose from {', '.join(variants)}")
    return reference


class Drafter(Record):
    method: str
    path: Path
    sha256: str


class Projector(Record):
    path: Path
    sha256: str


class Acquisition(Record):
    """The catalog acquisition object (evaluation contract "Model reference")."""

    model: str
    target: Path
    target_sha256: str
    drafter: Drafter | None
    projector: Projector | None

    @property
    def embedded_drafter(self) -> bool:
        return self.drafter is not None and self.drafter.path == self.target


class Target(Record):
    """One engine of the run and the artifact it serves."""

    engine: Engine
    #: A local path, ``hf:owner/repo@<commit>`` or ``ollama:[namespace/]name:tag``.
    reference: str
    #: Known content digest of a single-file artifact (the catalog's), when there is one.
    sha256: str | None = None

    @property
    def kind(self) -> Literal["mlx", "gguf", "registry"]:
        if self.engine in REGISTRY_ENGINES:
            return "registry"
        return "gguf" if self.engine in GGUF_ENGINES else "mlx"

    @property
    def id(self) -> str:
        return self.engine


def targets(
    engines: list[str], artifacts: dict[str, str], acquisition: Acquisition
) -> list[Target]:
    """The run's engines in order: GGUF engines serve the catalog target, the others the
    explicit artifact named for them."""
    if len(set(engines)) != len(engines):
        raise ValueError("each engine is measured once per run")
    unused = set(artifacts) - set(engines)
    if unused:
        raise ValueError(f"--artifact names engines that are not selected: {sorted(unused)}")
    selected = []
    for engine in engines:
        if engine in GGUF_ENGINES:
            if engine in artifacts:
                raise ValueError(f"{engine} serves the catalog's GGUF; it takes no --artifact")
            selected.append(
                Target(engine=engine, reference=str(acquisition.target),
                       sha256=acquisition.target_sha256)
            )
            continue
        if engine not in artifacts:
            raise ValueError(
                f"{engine} needs --artifact {engine}=REF: the catalog has no "
                f"{'Ollama registry' if engine in REGISTRY_ENGINES else 'MLX'} form of a model"
            )
        selected.append(Target(engine=engine, reference=normalize_reference(artifacts[engine])))
    return selected


class ArtifactFile(Record):
    path: str
    size: int
    sha256: str
    mtime_ns: int


class Artifact(Record):
    reference: str
    path: Path
    kind: Literal["mlx", "gguf", "registry"]
    context_limit: int = Field(gt=0)
    metadata: dict
    files: tuple[ArtifactFile, ...]

    def verify_unchanged(self) -> None:
        if self.path.is_dir() and {
            str(p.relative_to(self.path)) for p in artifact_files(self.path)
        } != {item.path for item in self.files}:
            raise ValueError(f"artifact files changed after preparation: {self.path}")
        for item in self.files:
            path = self.path / item.path if self.path.is_dir() else self.path
            stat = path.stat()
            if stat.st_size != item.size or stat.st_mtime_ns != item.mtime_ns:
                raise ValueError(f"artifact changed after preparation: {path}")


def normalize_reference(reference: str) -> str:
    if reference.startswith("hf:"):
        parse_hub(reference)
        return reference
    if reference.startswith("ollama:"):
        parse_registry(reference)
        return reference
    # Normalize without following symlinks: a Hub snapshot's name links to a
    # content-addressed blob, and the engine opens the named path.
    return os.path.abspath(Path(reference).expanduser())


def parse_hub(reference: str) -> tuple[str, str]:
    match = re.fullmatch(r"hf:([^/@]+/[^/@]+)@([0-9a-f]{40})", reference)
    if match is None:
        raise ValueError("Hub snapshots must be hf:owner/repository@<40-character-commit>")
    repository, revision = match.groups()
    return repository, revision


def parse_registry(reference: str) -> tuple[str, str, str]:
    """Namespace, name and tag of an ``ollama:[namespace/]name:tag`` reference."""
    match = re.fullmatch(
        r"ollama:(?:([A-Za-z0-9][\w.-]*)/)?([A-Za-z0-9][\w.-]*):([\w.-]+)", reference
    )
    if match is None:
        raise ValueError("Ollama references must be ollama:[namespace/]name:tag")
    namespace, name, tag = match.groups()
    return namespace or "library", name, tag


def registry_manifest(reference: str) -> Path:
    if "OLLAMA_MODELS" not in os.environ:
        raise ValueError("Ollama registry targets require --ollama-models or OLLAMA_MODELS")
    namespace, name, tag = parse_registry(reference)
    store = Path(os.environ["OLLAMA_MODELS"]).expanduser()
    return store / "manifests" / OLLAMA_REGISTRY / namespace / name / tag


def gguf_description(path: Path) -> tuple[int, dict]:
    """Declared context limit and identifying metadata of a GGUF file."""
    from gguf import GGUFReader

    reader = GGUFReader(str(path), "r")

    def value(name):
        field = reader.get_field(name)
        return field.contents() if field else None

    architecture = value("general.architecture")
    context = value(f"{architecture}.context_length")
    if not isinstance(context, int) or context <= 0:
        raise ValueError("GGUF has no declared context limit")
    return context, {"architecture": architecture, "file_type": value("general.file_type")}


def prepare_registry(target: Target) -> Artifact:
    """Evidence for a pulled Ollama model: its manifest pins every layer by digest."""
    reference = target.reference
    path = registry_manifest(reference)
    if not path.is_file():
        raise ValueError(f"Ollama model is not in the store (pull it first): {path}")
    manifest = json.loads(path.read_text())
    blobs = path.parents[4] / "blobs"

    def blob(digest: str) -> Path:
        return blobs / digest.replace(":", "-")

    configuration = json.loads(blob(manifest["config"]["digest"]).read_text())
    layers = manifest["layers"]
    stored = configuration.get("model_format")
    if stored != REGISTRY_FORMATS[target.engine]:
        raise ValueError(f"{target.engine} cannot serve an Ollama model stored as {stored}")
    if stored == "gguf":
        weights = [item for item in layers if item.get("mediaType") == GGUF_MODEL_LAYER]
        if len(weights) != 1:
            raise ValueError("Ollama GGUF model must have exactly one model layer")
        context, description = gguf_description(blob(weights[0]["digest"]))
        description["model_sha256"] = weights[0]["digest"].removeprefix("sha256:")
        description["model_bytes"] = weights[0]["size"]
    else:
        model = next((item for item in layers if item.get("name") == "config.json"), None)
        if model is None:
            raise ValueError("Ollama model has no config.json layer declaring a context limit")
        config = json.loads(blob(model["digest"]).read_text())
        text = config.get("text_config", config)
        context = text.get("max_position_embeddings") or text.get("max_sequence_length")
        if not isinstance(context, int) or context <= 0:
            raise ValueError("model configuration does not declare a context limit")
        description = {"model_type": text.get("model_type", config.get("model_type"))}
    stat = path.stat()
    return Artifact(
        reference=reference,
        path=path,
        kind="registry",
        context_limit=context,
        metadata={
            **description,
            "model_format": stored,
            "quantization": configuration.get("file_type"),
            "renderer": configuration.get("renderer"),
            "parser": configuration.get("parser"),
            "layers": len(layers),
            "bytes": sum(item["size"] for item in layers),
        },
        files=(
            ArtifactFile(
                path=path.name, size=stat.st_size, sha256=file_hash(path), mtime_ns=stat.st_mtime_ns
            ),
        ),
    )


def file_hash(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_files(path: Path) -> list[Path]:
    return sorted(
        p
        for p in path.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(path).parts)
    )


def prepare(target: Target) -> Artifact:
    reference = target.reference
    if target.kind == "registry":
        if not reference.startswith("ollama:"):
            raise ValueError(f"{target.engine} requires an ollama:[namespace/]name:tag reference")
        return prepare_registry(target)
    if reference.startswith("ollama:"):
        raise ValueError(f"{target.engine} cannot serve an Ollama registry model")
    if target.kind == "gguf":
        path = Path(reference)
        if not path.is_file() or target.sha256 is None:
            raise ValueError(f"{target.engine} serves the catalog's acquired GGUF: {path}")
        context, metadata = gguf_description(path)
        stat = path.stat()
        files = (
            ArtifactFile(
                path=path.name, size=stat.st_size, sha256=target.sha256, mtime_ns=stat.st_mtime_ns
            ),
        )
        return Artifact(
            reference=reference, path=path, kind="gguf", context_limit=context,
            metadata=metadata, files=files,
        )
    if reference.startswith("hf:"):
        from huggingface_hub import snapshot_download

        repository, revision = parse_hub(reference)
        path = Path(snapshot_download(repository, revision=revision))
    else:
        path = Path(reference)
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError(f"MLX artifact must be a model directory: {path}")
    config = json.loads((path / "config.json").read_text())
    text = config.get("text_config", config)
    context = text.get("max_position_embeddings") or text.get("max_sequence_length")
    if not isinstance(context, int) or context <= 0:
        raise ValueError("model configuration does not declare a context limit")
    files = artifact_files(path)
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("MLX artifact contains no safetensors weights")
    return Artifact(
        reference=reference,
        path=path,
        kind="mlx",
        context_limit=context,
        metadata={
            "model_type": text.get("model_type", config.get("model_type")),
            "quantization": config.get("quantization", config.get("quantization_config")),
        },
        files=tuple(
            ArtifactFile(
                path=str(p.relative_to(path)),
                size=p.stat().st_size,
                sha256=file_hash(p),
                mtime_ns=p.stat().st_mtime_ns,
            )
            for p in files
        ),
    )


def main() -> None:
    """Disposable artifact preparation worker, so cancellation can stop downloads and hashing."""
    import sys

    artifact = prepare(Target.model_validate_json(sys.argv[1]))
    Path(sys.argv[2]).write_text(artifact.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
