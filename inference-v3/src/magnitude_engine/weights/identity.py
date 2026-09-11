"""Content identity shared by ingestion, model binding, and encoded operations."""

from typing import NewType

ArtifactIdentity = NewType("ArtifactIdentity", str)
