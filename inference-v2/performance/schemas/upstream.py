"""The upstream call and parameter owner are explicit library bindings."""

from magnitude_engine.models.architectures.mlx_vlm.program import LibraryForward, LibraryProgram
from performance.bindings import Fields, schema
from performance.facts import OpaqueParameters


@schema(LibraryProgram)
def library(a: LibraryProgram, _: None) -> Fields[OpaqueParameters]:
    if not isinstance(a.call, LibraryForward):
        raise TypeError("library capture requires an explicitly bound LibraryForward")
    return Fields(OpaqueParameters(), operands={"model": a.call.model}, sources=(a.call,))
