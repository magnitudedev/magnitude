# Native template extraction

This directory owns the C++ template renderer and semantic-output parser linked into
`magnitude-templates`. Cargo compiles the explicit translation-unit list in `build.rs`; building
the crate does not run CMake, Python, `patch`, or consult another Magnitude source tree.

`source/` is the required subset of the patched extraction produced from llama.cpp revision
`930e2fa5995789efbf249a8bf61325bb626e417b`. The original file hashes and upstream repository are
recorded in `provenance/manifest.json`, and `provenance/patches/` records the ordered transformations
applied to those sources. `LICENSE.llama.cpp` is the upstream license. `src/` and `include/` are
Magnitude's owned C ABI boundary. Model-template fixtures used to verify the generic API live under
`tests/assets/`; they are test inputs, not runtime configuration or numerical-model dependencies.

To refresh the extraction, use the provenance record and patch series in a maintainer workflow,
then check in the resulting patched sources. Source preparation is deliberately not part of a
consumer build.
