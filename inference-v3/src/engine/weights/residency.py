"""Nominal weight-provider boundary used by the existing blueprint type system."""

from abc import ABC, abstractmethod

import ops

from .descriptor import WeightDescriptor
from .identity import ArtifactIdentity


class WeightResidency(ABC):
    @property
    @abstractmethod
    def identity(self) -> ArtifactIdentity: ...

    @abstractmethod
    def bind(self, descriptor: WeightDescriptor, dtype: ops.DType) -> ops.Binding: ...

    @abstractmethod
    def close(self) -> None: ...
