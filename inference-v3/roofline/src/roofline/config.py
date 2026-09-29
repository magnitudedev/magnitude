"""Folder-local definitions; there is no product catalog resolution."""

import os
from pathlib import Path

from .contracts import Models, Targets


def project_root() -> Path:
    if value := os.environ.get("ROOFLINE_ROOT"):
        root = Path(value).expanduser().resolve()
    else:
        root = next(
            (
                p
                for p in (Path.cwd(), *Path.cwd().parents)
                if (p / "roofline/models.json").is_file()
            ),
            None,
        )
        if root is None:
            root = next(
                (
                    p / "inference-v3"
                    for p in (Path.cwd(), *Path.cwd().parents)
                    if (p / "inference-v3/roofline/models.json").is_file()
                ),
                None,
            )
        if root is None:
            raise ValueError("run inside inference-v3 or set ROOFLINE_ROOT to its path")
    return root


class Configuration:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.models = Models.model_validate_json(
            (root / "roofline/models.json").read_bytes()
        ).models
        self.targets = Targets.model_validate_json(
            (root / "roofline/targets.json").read_bytes()
        ).targets
        for name, model in self.models.items():
            if unknown := model.locations.keys() - self.targets.keys():
                raise ValueError(f"model {name} has undefined targets: {sorted(unknown)}")
        self.workspace = Path(os.environ.get("ROOFLINE_WORKSPACE", root / "runs/performance"))

    def validate(self, experiment):
        if experiment.model not in self.models:
            raise ValueError(f"unknown model variant: {experiment.model}; use roofline models")
        for name in (
            *experiment.targets,
            *([experiment.input_target] if experiment.input_target else []),
        ):
            if name not in self.targets:
                raise ValueError(f"unknown target: {name}; use roofline workers list")
            if name not in self.models[experiment.model].locations:
                raise ValueError(f"target {name} has no artifact location for {experiment.model}")
