# Tensor system

The normative execution contract lives in
[Formula-defined execution and measurement](../../design/inference/formula-execution.md).

Engine composes typed formulas. Ops implements those formulas through operations
and owns physical execution. TileLang compiles and realizes portable kernel work.
Formula composition retains typed occurrences and effects for independent reference,
isolation and stable measurement. It does not force host submission boundaries.

Execution planning resolves dependencies, storage and maximal native submission
units; it does not enumerate or rank competing implementations. An operation may
compose children or directly implement its parent formula. Streaming dependencies
and required host observations are explicit parts of that execution.

The compiled callable retains static resources and executable units. Warm calls
bind dynamic inputs and submit prepared work, with resources retained until
completion. Logical acceptance remains an engine decision. Public ops APIs expose
neither TileLang nor TVM; no source generation or direct TIR manipulation is allowed.
