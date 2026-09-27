# Operations

The normative formula/operation contract now lives in
[Formula-defined execution and measurement](../../design/inference/formula-execution.md).

A formula defines stable mathematics, effects and useful quantities. An operation
is its complete physical implementation: I/O, allocation, transfer, conversion,
kernels and completion, expressed through shared ops execution facilities.
Runtime instrumentation supplies physical observations; formulas supply useful
units. There is no competing implementation registry or cost-ranked graph cover.

Attention's isolated body and its enclosing gated-output composition share one
streaming schedule definition. Composition removes redundant publication and
adds the output projection; it does not independently widen the attention tile.
The same storage-typed Q/K and FP32 probabilities/PV body therefore benefits from
isolated optimization in both contexts. A parent's measured duration remains its
own observation, never a sum inferred from child measurements.

See [tensor-system.md](tensor-system.md) for execution and
[performance.md](performance.md) for the measurement boundary.
