"""Evidence publication used by the common benchmark and carried-over session tool."""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, JsonValue

from magnitude_engine.platform.publication import publish


def atomic(path: Path, value: JsonValue) -> None:
    publish(path, json.dumps(value, indent=2, allow_nan=False).encode("utf-8"))


class Store:
    """Content-addressed immutable evidence; existing assessments are not updated."""

    def __init__(self, root: Path):
        self.root = root

    def put(self, record: BaseModel) -> tuple[str, bool]:
        payload = record.model_dump_json().encode("utf-8")
        identity = hashlib.sha256(payload).hexdigest()
        path = self.root / "runs" / identity / "run.json"
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError("stored evidence does not match its content identity")
            return identity, False
        publish(path, payload)
        return identity, True
