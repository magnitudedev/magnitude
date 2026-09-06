"""Desktop's authenticated, host-scoped door to the native companion.

Hermes loads dashboard modules independently from agent modules. Load our package
relative to this file, without changing sys.path or relying on its install folder
name. Merely mounting routes must not connect to or start Magnitude.
"""

import importlib
import importlib.util
from pathlib import Path
import sys
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field


def load_companion():
    name = __name__ + ".companion"
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        commands = importlib.import_module(name + ".commands")
        return module, commands.ModelCommands()
    except BaseException:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                sys.modules.pop(key, None)
        raise


companion, commands = load_companion()
observations = importlib.import_module(companion.__name__ + ".observations")
reader = observations.ObservationReader()
router = APIRouter()


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["status", "load", "stop"]
    args: str = Field(default="", max_length=2048)


@router.get("/setup")
def setup():
    return {"prompt": companion.SETUP_PROMPT}


@router.get("/progress")
def progress(session_id: str):
    if not session_id or len(session_id) > 256:
        return {"available": False, "observations": []}
    snapshot = reader.read(observations.session_group(session_id))
    return {"available": snapshot is not None, "observations": snapshot or []}


@router.post("/command")
def command(body: Command):
    # A sync route runs in FastAPI's worker pool, not the gateway event loop.
    # Only an explicit user action reaches the lazy service-starting client.
    return {"message": getattr(commands, body.operation)(body.args)}
