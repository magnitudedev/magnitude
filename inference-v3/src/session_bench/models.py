"""Project-local aliases and immutable artifact evidence; no checked-in model catalog."""

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from .policy import ENGINES
from .sessions import Record, digest


class Alias(Record):
    mlx: str | None = None
    gguf: str | None = None


class Target(Record):
    engine: Literal["magnitude", "mlx-vlm", "omlx", "llama.cpp"]
    reference: str

    @property
    def kind(self) -> Literal["mlx", "gguf"]:
        return "gguf" if self.engine == "llama.cpp" else "mlx"

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
    kind: Literal["mlx", "gguf"]
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
    path = Path(reference).expanduser()
    return str((root / path).resolve())


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


def select(root: Path, models: list[str], engines: list[str], targets: list[str]) -> list[Target]:
    if targets and (models or engines):
        raise ValueError("use --target pairs or --model/--engine, not both")
    selected = []
    if targets:
        for item in targets:
            engine, separator, reference = item.partition("=")
            if not separator:
                raise ValueError("--target requires ENGINE=ARTIFACT")
            selected.append(Target.model_validate({
                "engine": engine, "reference": normalize_reference(reference, root),
            }))
    else:
        if not models:
            raise ValueError("--model is required (aliases live in inference-v2/models.local.json)")
        local = aliases(root)
        for model in models:
            for engine in engines or ["magnitude"]:
                if engine not in ENGINES:
                    raise ValueError(f"unknown engine: {engine}")
                alias = local.get(model)
                if alias:
                    reference = alias.gguf if engine == "llama.cpp" else alias.mlx
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
    if reference.startswith("hf:"):
        from huggingface_hub import hf_hub_download, snapshot_download

        repository, revision, filename = parse_hub(reference)
        if target.kind == "gguf" and not filename:
            raise ValueError("llama.cpp requires an explicit #GGUF-filename")
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
            raise ValueError(f"llama.cpp requires a GGUF file: {path}")
        from gguf import GGUFReader

        reader = GGUFReader(str(path), "r")

        def value(name):
            field = reader.get_field(name)
            return field.contents() if field else None

        architecture = value("general.architecture")
        context = value(f"{architecture}.context_length")
        if not isinstance(context, int) or context <= 0:
            raise ValueError("GGUF has no declared context limit")
        metadata = {"architecture": architecture, "file_type": value("general.file_type")}
        files = [path]
        del reader
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
