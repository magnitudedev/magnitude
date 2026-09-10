"""Immutable validated data contracts, independent of runtime ownership."""

from typing import NewType

from pydantic import BaseModel, ConfigDict

TokenId = NewType("TokenId", int)


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
