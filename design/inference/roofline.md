---
applies_to:
  - inference-v3/formula-performance/**
  - inference-v3/roofline/**
---

# Roofline

Roofline owns performance measurement orchestration, worker operation, persistent
evidence, queries and a read-only Textual browser. Engine integrations own workload
execution and numerical checking. Magnitude execution uses production Ops and
TileLang mechanisms.

## Test definitions

Local, gitignored model and target definitions configure the testing setup without
depending on product catalogs or sibling projects. A model is declared once under
its `model:format:quantization` ID, with an artifact checksum and absolute file
locations keyed by target. Format and quantization are not repeated as metadata.
Targets define connections, devices and an optional worker directory; they contain
no model mappings or repository paths. Architecture comes from artifact contents.

Checksums identify model bytes independently of location. Workers verify the
selected copy before measurement. Missing locations, unsupported formats and
checksum mismatches are explicit failures; no similar model is substituted.
Accepted definitions are immutable measurement provenance even when the local
configuration later changes.

## Worker ownership

Each worker owns a user directory, defaulting to `~/.local/share/roofline/` on its
host. Control installations, managed Python, execution environments, received
source, compilation and dependency caches, journals and logs live there. Existing
repositories and their virtual environments are neither installation destinations
nor execution dependencies. External model artifacts are read from their configured
paths.

Setup bootstraps an isolated control environment from the copied Roofline package
and its dependency lock using host Python 3 and uv. It respects SSH host verification,
requires no privileged service or exposed listening port, and reports missing host
prerequisites. Package updates refuse to replace a worker with accepted or active
work. Idle workers restart from the new installation.

Execution and scope discovery use received source snapshots and dependency locks.
Submissions include content-verified fixture bytes, independent of changes to the
upstream download. Numerical checking dependencies are locked execution inputs.
Dependencies and native compiler libraries are built inside worker-owned execution
environments, with no fallback to a checkout interpreter or compiler installation.
Dependency-build reuse requires matching compiler and native-library source, build
hooks, dependency locks, build
settings and Python version, and can reference only this worker's own prepared
environments. Engine Python code always comes from the requested snapshot. Cached preparation never substitutes for new measurement samples.

Discovery and preflight do not build execution environments, transfer compiler
sources or load model weights. They check each requested target independently and
trace metadata using a matching existing environment. Missing preparation remains
explicit; ordinary measurement performs it automatically. Metadata inspection does
not claim full artifact verification or device execution qualification.

Explicit shared-input requests name a producer source and target. The coordinator
completes and collects that boundary before dispatching consumer measurements.
Consumers verify its workload provenance, exact production graph, ports, numerical
representation and immutable weight bindings. They independently check and sample
their own implementations. Frozen inputs remain distinguished from native
production inputs; a failed producer leaves every consumer outcome explicit.

The local coordinator captures source and collects evidence. Remote workers accept
stable attempt IDs, journal execution, and survive SSH disconnection. Reconnection
collects the same attempt rather than submitting a duplicate. Worker shutdown must
end its owned execution processes. Environment preparation and numerical execution
share the declared deadline. Restart reaps only a recorded process whose identity
still matches. Permanent protocol failures terminate the attempt; transport loss
reconciles accepted work by its existing identity. Completed ordinary samples are
checkpointed before optional diagnostics and independent checking.

## Evidence and browsing

Measurements publish automatically. Numerical correctness, execution outcome,
collection state and timing conclusiveness are distinct. Failed numerical checks
remain queryable and cannot enter best-correct history. Comparisons require the
relevant artifact, workload, component-input, hardware and protocol identities;
model labels and target aliases alone cannot establish compatibility.

Explicit session import preserves artifact checksums, original chronology, timer
boundaries and response validation. It matches a unique configured artifact without
inferring missing checksums or source. Imported response validation cannot qualify
numerical correctness. Unmatched records remain addressable as original artifacts.

The analytical unit is the model's existing formula composition. Ops publishes the numerical graph and execution evidence. The device-free
`formula-performance` package owns mathematical derivation and evidence interpretation;
Roofline owns publication, indexing and the shared model report. Each measurable formula declares
its meaningful primary metric. Missing declarations fail supported-formula validation;
no display layer substitutes another available quantity.

All workload observations contribute to the applicable component relations over
parameters, hardware and implementation revisions. Exact experimental comparisons
retain their strict compatibility requirements; applicability to a parameterized
performance relation is separately justified. Workload names identify evidence,
not independent model views. Different operating points remain distinguishable
within one consolidated report, without a misleading universal average.

The numerical graph supplies obligations and composition relationships for
mathematical ceilings. Each formula has one mathematical roofline parameterized by
workload geometry and hardware capabilities. Hardware profiles bind that function;
they do not define disconnected formula models. Every measurement is normalized
against its own applicable hardware-relative ceiling and enters the shared relation.
Changes to common derivations reevaluate evidence across hardware bindings.
Recorded physical execution supplies measured attribution
and implementation predictions. Child evidence updates dependent ancestor
predictions or constraints; enclosing evidence updates mapped regions or joint
constraints on unresolved terms. Independent isolated timings cannot be relabeled
as actual parent contributions. A total-only observation does not identify every
child's cost. Derived predictions, direct measurements and theoretical bounds
remain distinct and retain their evidence dependencies.

Publication updates affected derivations through a dependency index and exposes a
coherent analytical snapshot. Original observations remain immutable. Each conditional prediction preserves the
inputs available when it was made; later enclosing measurements validate it without
rewriting that earlier prediction. Ordinary workload timings compose the actual
sequence of model forwards, with additive useful quantities and certified completion
barriers. They contribute alongside separately instrumented observations. Independent
whole-model checking evaluates production operations at their actual inputs,
loading and releasing independent weight references within each checked boundary.
It verifies arithmetic before compression and encoded bytes afterward, then checks
ordinary execution against that qualified replay. It never expands the whole
weight archive at once. Reanalysis
is pure computation over recorded graph/evidence contracts and must work without
an engine, compiler, numerical runtime or worker. Import preserves this evidence
closure. Discovery publishes actual structure, not another model definition.

The Textual browser requires only a model selection. One structural component tree
shows accumulated performance knowledge across all recorded workloads and hardware.
Rows name the declared formula unit and show hardware-normalized performance against
the common roofline, supporting variation/coverage and contribution information.
Hardware-specific rates and evaluated ceilings belong to the supporting evidence.
Phase, context, batch and hardware parameters are coordinates of the evidence,
available in component details; they are not prerequisite selectors or primary
hardware partitions. Details expose derivations, operating points, history and
prediction validation. Raw provenance belongs behind explicit named-evidence navigation.

No representative workload or latest case stands in for the whole model. Missing
evidence remains an identified term of the same analytical model. Empirical resource
references cannot substitute for mathematical ceilings. Browsing never launches
measurements or traces source. Unknown current-code freshness remains explicit.
Model analysis runs in the background so the tree displays loading state immediately
and remains responsive while stored evidence is consolidated.

## Acceptance

- Two differently named copies with the same checksum preserve artifact identity;
  a changed copy fails verification.
- Unknown targets and missing model locations fail before dispatch.
- Setup and execution succeed with no remote engine checkout configured or present.
- Worker environments and caches remain under the worker directory even when the
  invoking shell points at another virtual environment or compiler installation.
- Updating an idle worker preserves its journal; updating a busy worker is rejected.
- Queries work offline without engine or GPU imports, preserving failed observations.
- A model-only query and TUI selection consolidate all recorded workload domains
  in one formula graph without requiring a workload or hardware selection. Each
  observation is normalized using its own hardware binding of the same roofline.
- New child or enclosing evidence updates dependent analytical results, preserving
  direct observation history and distinguishing predictions from measurements.
- Formula metrics are explicitly declared and dimensionally checked. Ceilings
  derive from justified constraints; empirical references are identified separately.
- Shared/fused/overlapping work is accounted for through execution relationships,
  never by summing independent child timings or inventing exclusive attribution.
