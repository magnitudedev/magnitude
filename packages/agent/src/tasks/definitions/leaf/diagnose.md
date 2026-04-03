---
id: diagnose
label: Diagnose
description: Hypothesis-driven root-cause isolation for bugs, failures, or unexpected behavior.
allowedAssignees: [debugger]
---

<!-- @lead -->

## Inputs to provide the worker
- Symptom description with concrete observed behavior (what failed, where, and how often).
- Reproduction steps if known, including commands, inputs, environment, and setup details.
- Error outputs, stack traces, logs, screenshots, or other failure artifacts.
- Relevant code areas, recent changes, and any prior debugging attempts or hypotheses.
- Scope boundaries for diagnosis and any time/risk constraints.

## Output to expect from the worker
- Confirmed reproduction status with exact command/sequence used.
- Root-cause identification supported by an explicit evidence chain.
- Causal explanation from triggering conditions to failure point and defect mechanism.
- Explicit list of tested and falsified hypotheses.
- Clear statement of knowns, unknowns, and blockers that prevent stronger conclusions.

## Coordination loop
1. Provide the best available reproduction context and failure artifacts up front.
2. If reproduction fails, resolve environment/setup gaps before deeper diagnosis.
3. Review each hypothesis test for falsifiability and evidence quality; reject speculation.
4. Require a complete symptom-to-mechanism causal chain before accepting a diagnosis.
5. Route only confirmed root-cause findings into implementation work.

<!-- @worker -->

## Objective
- Isolate the true root cause of the failure using reproducible evidence and falsifiable hypothesis testing, with a diagnosis precise enough to guide implementation.

## Procedure
1. Reproduce the failure with a concrete command/sequence and capture exact outputs and conditions.
2. Generate specific, falsifiable hypotheses tied to observable behavior, then prioritize by diagnostic value.
3. Run targeted tests to confirm or falsify hypotheses, changing one variable at a time when possible.
4. Trace data/control flow from triggering inputs to failure point and identify where behavior diverges from expectation.
5. Confirm the defect mechanism (logic, state, contract mismatch, race, configuration, etc.) with direct evidence.
6. Document rejected hypotheses, residual unknowns, and any blockers separately from the primary cause statement.

## Output contract
- Return a reproduction section with exact steps/commands and observed results.
- Return an evidence chain linking symptom → intermediate findings → root cause.
- Return a root-cause statement with precise mechanism and triggering conditions.
- Return a ruled-out hypotheses section with rejection rationale.
- Return remaining unknowns/risks and concrete unblock requests if verification is limited.
- Do not present unverified hypotheses as confirmed root causes.

<!-- @criteria -->

## Completion criteria
- [ ] Root cause is identified with concrete evidence, not hypothesis alone.
- [ ] Reproduction steps are confirmed, or inability to reproduce is documented with explicit blockers.
- [ ] Evidence chain from symptom to cause is documented and test-backed.
- [ ] Causal explanation is specific enough to guide implementation of a fix.
- [ ] Known unknowns and ruled-out alternatives are explicitly documented.