---
applies_to:
  - inference-v4/seismic/crates/seismic-accounting/src/schedule.rs
  - inference-v4/seismic/crates/seismic-accounting/src/algebra.rs
  - inference-v4/seismic/crates/seismic-accounting/src/schedule/**
  - inference-v4/seismic/crates/seismic-accounting/src/objective.rs
  - inference-v4/seismic/crates/seismic-accounting/tests/independent_solver.rs
  - inference-v4/seismic/crates/seismic-metal/examples/solver_family_inventory.rs
---

# Independent scheduling boundary

The Seismic accounting adapter appends execution constraints to one caller-owned
mathematical model. Source legality, workload identity, units, hardware
assumptions and native authority remain owned by Seismic. The compiler owns one
common solver search over the complete export. Accounting owns no child search,
structured refinement frontier or algorithm-specific scheduling coordinator.

All concurrent work contributes to one enclosing resource boundary. Serial and
parallel composition, repeated occurrences and scope lifetimes preserve their
original dependency and resource semantics. Finite expansion is one exact
encoding of that structure. When compact occurrence semantics cannot yet be
expressed in the shared model, the original region and a typed analysis
obligation remain retained. Missing vocabulary does not establish infeasibility,
and a restricted repeated schedule family does not establish complete coverage.

A schedule validated by the original model can supply a sufficient finite horizon.
Dependency-earliest and serial schedules are candidate seeds only; every resource,
lifetime and static-order constraint must hold before either bounds the horizon.
Without a feasible seed, the sum of operation latencies supplies the exhaustive
horizon: removing idle gaps preserves operation reservations and contracts
resident lifetimes. A serial seed is checked, never assumed feasible.
Every schedule that could improve that witness fits within the horizon, because
the objective is the maximum completion of nonnegative-time operations. No guessed
horizon or arbitrary candidate cutoff narrows improvement coverage. Unsupported
integer ranges and missing mappings retain typed analysis obligations. Invalid
requests and reconstruction failures remain errors.

The translation preserves operation latency, result and issue dependencies,
offset resource reservations, event-to-event lifetimes and one common static
instruction order across dynamic visits. Completion is the actual maximum end
event, including in incomplete incumbents. Every reconstructed schedule is
independently validated against the original model and objective before returning
it. A solver optimum cannot promote an optimistic machine relaxation into an
executable witness or establish physical qualification.

Budget exhaustion retains the compiler session's single generic solver state.
An incomplete incumbent is diagnostic and never constructs Tuned IR or calls
native compilation. An infeasible translated result after a valid original
witness indicates a defect; it cannot establish infeasibility of the original
execution family.

Read-only family inspection reports unresolved compiler choices and their actual
constraints separately from fully selected schedule models. Diagnostic choices
used to prepare an inspected family are stated explicitly. Hypothetical hardware
models remain identified as hypothetical. A fixed-execution translation does not
establish unresolved-family coverage or practical compiler integration.

Acceptance requires small-model agreement between the public scheduling boundary, the generic solver and direct bounded
schedule enumeration, with tests for
resource offsets, lifetimes, common static orders, interruption and authority.
Unresolved-family coverage and compact repetition require their own checked
correspondence; fixed-schedule validation cannot establish either.

The symbolic scheduling boundary collects all conditional activities, offset
reservations and lifetimes for each shared resource before constructing cumulative
constraints. Whole-activity reservations also expose duration/demand coupling.
Completion is an exact maximum over active end events; inactive events contribute
zero even if their private time variables are nonzero. These joint relations
preserve alternatives that are slower in isolation but compose better.

Fixed schedule fragments use the same translation inside this shared boundary.
Their complete activation condition guards dependencies, service offsets,
lifetimes and static orders; private inactive assignments are canonical. A
fragment supplies constraints and reconstruction correspondence, never an
isolated cost or a family-coverage claim. Alternatives that exceed a justified
improvement horizon are excluded by their active constraints, without preventing
other alternatives from being constructed. The enclosing family owns compatible
workload and timebase conditions and validates the complete solver assignment.

Objective interpretation is separate from selection. A reconstructed flat or
structured schedule must pass its original-model validator, retain its physical
time unit, and have the same completion as the solver objective. The global
lower bound is supplied by the enclosing search; feasibility never implies a
child optimum. Only the compiler's globally optimal outcome may authorize native
selection under the current acceptance policy.

Budgets count solver work units. Resumption retains the compiler session's model
identity and solver state. Progress distinguishes missing construction or
analysis from an exhausted work budget; no such obligation becomes infeasibility.
