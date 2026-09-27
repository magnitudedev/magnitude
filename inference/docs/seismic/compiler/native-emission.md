# Native emission

Native emission forms one specified request for a fixed physical contract and reconciles the resulting artifact's actual requirements. It does not choose a different source computation, schedule or numerical policy.

| Input | Output |
| --- | --- |
| Closed candidate physical structure, selected emission request, effective target and toolchain | Compatible native artifacts and reconciled executable bindings/resources |

## Decision boundary

Physical construction fixes operations, contribution structure, storage, mapping, communication, numerical modes and execution protocol. Residual emission choices describe permitted ways to form that contract, including lowering recipes and toolchain options that preserve its behavior and constraints.

These alternatives belong to the same [candidate domain](candidate-domain.md) and participate in the same [evaluation](evaluation.md) as physical choices. The backend forms the specified request; it has no separate tuning policy. Derived facts are not independent choices, and uncontrolled toolchain decisions are not selectable dimensions. A compiler flag that changes reassociation, FTZ or contraction is a numerical choice and cannot be enabled after applicability was derived.

Optimal emission requires both formation routes that can obtain an optimum within the fixed contract and sufficient cost comparisons to select it. Delegating to a native toolchain or measuring a finite set of variants does not establish this requirement.

## Backend generation

CPU, Metal and CUDA consume the same operation and storage meanings while lowering to their native facilities. Backend recipes preserve explicit rounding, participant protocols, logical address arithmetic and publication. Segmented and direct storage share the selected ABI; a backend cannot accept a segmented binding while continuing to emit direct pointer access.

Source operations and programmable target lowerings remain visible through typed construction. Opaque native text belongs only to the explicit authored direct-native route, not a way for an optional compiler factory to hide another algorithm.

The pure `PhysicalDialect` owns typed launch descriptors as well as kernel operations. Each target-parameterized launch contains its selected descriptor and common geometry exactly once. CPU and Metal use ordinary launches; CUDA currently supports independent and cooperative-grid launches. Semantic participation requirements are resolved using target facts during construction. Ordinary region context is inherited, with no extra deployment or event object.

The same descriptor survives normalization, import, specialization and executable lowering. Native reflection describes which descriptors and limits an artifact admits; reconciliation checks the constructed request without translating or replacing it. Several launches may share one native artifact while retaining distinct descriptors. The registry does not supply a second launch-mode mapping.

## Native reconciliation

Compilation and pipeline creation may reveal artifact-specific layout or resource facts. Formation binds these to the exact emitted artifact and fixed physical request before exposing an executable. Any resulting resource projections used by evaluation and runtime must agree with that artifact.

Native reflection is a legitimate external boundary: it observes what the toolchain actually produced. It does not retroactively grant source correspondence to arbitrary code. A contradiction between a guaranteed target contract and the result is a compiler/toolchain integration defect. Genuine toolchain exhaustion, device loss or service failure is reported separately.

Actual argument ordinals can be cached with a native artifact. Semantic destinations, value versions and schedule publications remain owned by the executable launch. Reusing an artifact cannot import another candidate's destinations.

## Reuse and publication

Multiple candidates may share a native kernel while having different whole-entry control or applicability. Request identity includes the complete specialized lowering input, ABI/storage layout, numerical environment, linkage, options and toolchain/target facts. A source hash or matching reflection alone is insufficient.

A request identifies what to form; an artifact instance identifies what was actually formed and measured. Measurements remain attached to that instance, and publication retains it. Recompiling the same request does not automatically transfer its predecessor's timings.
The request key preserves explicit descriptor settings even when their reflected limits equal an implicit default. Equal instruction text or reflected resource numbers do not establish that two formation requests, retained artifacts, or objective costs are interchangeable.

The current Metal pipeline handle retains its exact submitted MSL source and entry name; the resulting machine image remains opaque. The CUDA handle retains the exact cubin submitted to its module loader, its entry name and the configured function handle. These formation witnesses are charged to host metadata storage and do not by themselves certify final loaded instructions or latency.

Shared preparation can explicitly form a fresh outcome for an already resident physical coordinate. It returns a distinct candidate and native instance identities; ordinary coordinate lookup keeps its cached candidate. A search policy must request this action when its formation model requires a retry, and must account for the new native work. Neither the fresh-call API nor memoization alone proves that every useful native outcome is reachable.

Preparation-local realization caches distinguish materialized family instances and exact ordered active physical-choice/value pairs even when their structural or assignment digests agree. Those digests still label semantic subjects elsewhere; the local cache guarantee does not establish global semantic identity or native-universe coverage.
Structural materialization also keeps each completed construction path's own family rather than sharing an earlier one by structural digest alone.

Publication manifests bind complete native descriptions to ordered formed-instance IDs. A fresh outcome may have the same semantic implementation digest as an earlier outcome and different reflection; both remain independently resolvable. Rebinding the same exact IDs with conflicting descriptions is rejected.

Selection policies also retain exact preparation candidate IDs in selector order. Finalization verifies that each ID names the corresponding retained executable in its own preparation before attaching the already formed handles. The published executable variant exposes that same preparation candidate ID, so later evidence can name its exact formation outcome. Equal semantic labels cannot authorize swapping fresh outcomes between selector operands.

Feedback observation requests carry the preparation candidate ID of the executable being measured. Observers can use it to bind a sample to one retained formation outcome even when different outcomes share a semantic digest.

If a later kernel in one candidate fails to form, successful earlier kernels remain resident. The failure carries their aggregate native metrics back to preparation so the budget records that work, and an ordinary retry can reuse those instances. A failed native call has no successful-artifact metrics in the current native compiler contract.

Private compilation can be memoized while construction or applicability is incomplete. Only the complete applicable and natively reconciled result can be timed or retained. Compilation requests occur during [evaluation](evaluation.md). [Prepared-kernel finalization](prepared-kernel.md) packages selected artifacts and cannot introduce another compilation or replanning phase.
