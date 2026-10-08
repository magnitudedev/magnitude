"""Project-local aliases and immutable artifact evidence; no checked-in model catalog."""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from .policy import ENGINES
from .sessions import Record, digest

GGUF_ENGINES = ("magnitude", "llama.cpp", "ollama")
# Engines that serve a model from Ollama's own registry store rather than a local artifact.
REGISTRY_ENGINES = ("ollama-mlx", "ollama-registry")
# The stored format each registry engine serves; Ollama picks its runner from it.
REGISTRY_FORMATS = {"ollama-mlx": "safetensors", "ollama-registry": "gguf"}
GGUF_MODEL_LAYER = "application/vnd.ollama.image.model"
OLLAMA_REGISTRY = "registry.ollama.ai"


class Alias(Record):
    mlx: str | None = None
    gguf: str | None = None


class Target(Record):
    engine: Literal[
        "magnitude", "mlx-vlm", "omlx", "llama.cpp", "ollama", "ollama-mlx", "ollama-registry"
    ]
    reference: str

    @property
    def kind(self) -> Literal["mlx", "gguf", "registry"]:
        """The container this engine accepts: the native engine, llama.cpp and Ollama's
        import read GGUF; the registry engines serve a model pulled into Ollama's store."""
        if self.engine in REGISTRY_ENGINES:
            return "registry"
        return "gguf" if self.engine in GGUF_ENGINES else "mlx"

    @property
    def id(self) -> str:
        return f"{self.engine}-{digest(self.reference)[:10]}"


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


def aliases(root: Path) -> dict[str, Alias]:
    path = root / "models.local.json"
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object mapping aliases to MLX/GGUF references")
    return {key: Alias.model_validate(item) for key, item in value.items()}


def normalize_reference(reference: str, root: Path) -> str:
    if reference.startswith("hf:"):
        parse_hub(reference)
        return reference
    if reference.startswith("ollama:"):
        parse_registry(reference)
        return reference
    # Normalize without following symlinks: a Hub snapshot's GGUF name links to an
    # extensionless content-addressed blob, and the engine opens the named file.
    return os.path.abspath(root / Path(reference).expanduser())


def parse_hub(reference: str) -> tuple[str, str, str | None]:
    match = re.fullmatch(r"hf:([^/@]+/[^/@]+)@([0-9a-f]{40})(?:#(.+))?", reference)
    if match is None:
        raise ValueError(
            "Hub references must be hf:owner/repository@<40-character-commit>[#filename]"
        )
    repository, revision, filename = match.groups()
    if filename and (Path(filename).is_absolute() or ".." in Path(filename).parts):
        raise ValueError("unsafe Hub filename")
    return repository, revision, filename


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


def prepare_registry(target: Target) -> "Artifact":
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


def select(root: Path, models: list[str], engines: list[str], targets: list[str]) -> list[Target]:
    if targets and (models or engines):
        raise ValueError("use --target pairs or --model/--engine, not both")
    selected = []
    if targets:
        for item in targets:
            engine, separator, reference = item.partition("=")
            if not separator:
                raise ValueError("--target requires ENGINE=ARTIFACT")
            selected.append(
                Target.model_validate(
                    {
                        "engine": engine,
                        "reference": normalize_reference(reference, root),
                    }
                )
            )
    else:
        if not models:
            raise ValueError(
                "--model is required (aliases live in inference/benchmarks/models.local.json)"
            )
        local = aliases(root)
        for model in models:
            for engine in engines or ["magnitude"]:
                if engine not in ENGINES:
                    raise ValueError(f"unknown engine: {engine}")
                alias = local.get(model)
                if alias:
                    reference = alias.gguf if engine in GGUF_ENGINES else alias.mlx
                    if not reference:
                        raise ValueError(f"alias {model!r} has no artifact for {engine}")
                elif model.startswith("hf:") or (root / Path(model).expanduser()).exists():
                    reference = model
                else:
                    raise ValueError(
                        f"unknown model alias {model!r}; edit {root / 'models.local.json'}"
                    )
                selected.append(
                    Target(engine=engine, reference=normalize_reference(reference, root))
                )
    unique = {target.id: target for target in selected}
    return list(unique.values())


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
    if reference.startswith("hf:"):
        from huggingface_hub import hf_hub_download, snapshot_download

        repository, revision, filename = parse_hub(reference)
        if target.kind == "gguf" and not filename:
            raise ValueError(f"{target.engine} requires an explicit #GGUF-filename")
        if target.kind == "mlx" and filename:
            raise ValueError("MLX engines require a complete snapshot, not one file")
        path = Path(
            hf_hub_download(repository, filename, revision=revision)
            if filename
            else snapshot_download(repository, revision=revision)
        )
    else:
        path = Path(reference)
    if target.kind == "mlx":
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
        metadata = {
            "model_type": text.get("model_type", config.get("model_type")),
            "quantization": config.get("quantization", config.get("quantization_config")),
        }
    else:
        if not path.is_file():
            raise ValueError(f"{target.engine} requires a GGUF file: {path}")
        context, metadata = gguf_description(path)
        files = [path]
    evidence = tuple(
        ArtifactFile(
            path=str(p.relative_to(path)) if path.is_dir() else path.name,
            size=p.stat().st_size,
            sha256=file_hash(p),
            mtime_ns=p.stat().st_mtime_ns,
        )
        for p in files
    )
    return Artifact(
        reference=reference,
        path=path,
        kind=target.kind,
        context_limit=context,
        metadata=metadata,
        files=evidence,
    )


def main() -> None:
    """Disposable artifact preparation worker, so cancellation can stop downloads and hashing."""
    import sys

    artifact = prepare(Target.model_validate_json(sys.argv[1]))
    Path(sys.argv[2]).write_text(artifact.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
