# Performance

The normative measurement system lives in
[Formula-defined execution and measurement](../../design/inference/formula-execution.md).

Formula identities define stable performance boundaries across operation rewrites.
One persistent Lab and observation store serve isolated development and the required
interactive TUI. Runtime observations and formula-derived useful units remain
separate from justified theoretical bounds and empirical resource ceilings.

## Complete boundaries

Initial resident import and compilation are preparation costs. Recurring streaming
I/O, allocation, conversion, transfers, kernels and required completion belong to
the measured invocation. Host durations are not GPU timestamps; source API bytes
are not disk counters. Parent latency is measured, never inferred by adding isolated
children. Cache/residency, device, precision and input conditions identify comparisons.

## Integration evidence

Prefill covers the complete admitted prompt computation and required state publication.
Decode begins from an identical retained history without prefix construction in the
interval. Samples must not share a mutable tail: complete and abort each measured
advance before reuse. Reference logits come independently from the same artifact.
Time to first token is client-observed admission through availability of the first
output, including serving, packing, sampling and publication costs.

Long histories, supported non-ideal shapes and correctness remain integration gates.
V2 is a reference point, not the optimization target. Neither an isolated kernel win
nor a favorable aggregate excuses an unexplained production regression.

Session bench remains opaque external serving evidence using the
[benchmark fixtures](benchmark-fixtures.md); native service spans do not substitute
for client-observed latency. Raw observations retain provenance; design documents
carry no benchmark numbers. Existing runs/results remain historical evidence, not
a second source of truth for formula measurement definitions.
