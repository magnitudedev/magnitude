from typing import Literal

from benchmarks.contracts import Subject, SubjectBlueprint
from magnitude_engine.composition import component


@component
class KVAppend(SubjectBlueprint):
    granularity: Literal["runs", "pages"]
    prefix_tokens: int
    append_tokens: int

    @staticmethod
    def implementation() -> type[Subject]:
        from .runtime import AppendTrace

        return AppendTrace


@component
class KVBranch(SubjectBlueprint):
    page_size: int
    prefix_tokens: int
    branch_tokens: int

    @staticmethod
    def implementation() -> type[Subject]:
        from .runtime import BranchTrace

        return BranchTrace
