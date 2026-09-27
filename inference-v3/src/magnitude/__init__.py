"""Public Python interface to Magnitude's existing inference engine.

Exports are loaded on access. Importing magnitude never loads a model or opens
an execution device; implementation packages retain their existing names.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from engine.data import TokenId as TokenId
    from engine.inputs.tokenizer import SpecialTokens as SpecialTokens
    from engine.inputs.tokenizer import Tokenizer as Tokenizer
    from engine.loading import LoadedModel as LoadedModel
    from engine.loading import load_model as load_model
    from engine.models.sequence import LogitsSelection as LogitsSelection
    from engine.models.sequence import ModelAdvance as ModelAdvance
    from engine.models.sequence import ModelBatch as ModelBatch
    from engine.models.sequence import ModelCheckpoint as ModelCheckpoint
    from engine.models.sequence import ModelExecutor as ModelExecutor
    from engine.models.sequence import ModelInput as ModelInput
    from engine.models.sequence import ModelRequest as ModelRequest
    from engine.models.sequence import ModelSequence as ModelSequence
    from engine.platform.backend import Backend as Backend
    from ops.runtime.resources import CapacityError as CapacityError

__all__ = [
    "Backend",
    "CapacityError",
    "LoadedModel",
    "LogitsSelection",
    "ModelAdvance",
    "ModelBatch",
    "ModelCheckpoint",
    "ModelExecutor",
    "ModelInput",
    "ModelRequest",
    "ModelSequence",
    "SpecialTokens",
    "TokenId",
    "Tokenizer",
    "load_model",
]

_EXPORTS = {
    "Backend": "engine.platform.backend",
    "CapacityError": "ops.runtime.resources",
    "LoadedModel": "engine.loading",
    "load_model": "engine.loading",
    "SpecialTokens": "engine.inputs.tokenizer",
    "TokenId": "engine.data",
    "Tokenizer": "engine.inputs.tokenizer",
    **{
        name: "engine.models.sequence"
        for name in (
            "LogitsSelection",
            "ModelAdvance",
            "ModelBatch",
            "ModelCheckpoint",
            "ModelExecutor",
            "ModelInput",
            "ModelRequest",
            "ModelSequence",
        )
    },
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
