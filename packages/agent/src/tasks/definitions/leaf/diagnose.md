---
id: diagnose
label: Diagnose
description: Hypothesis-driven root-cause isolation for bugs, failures, or unexpected behavior.
allowedAssignees: [debugger]
---

<!-- @lead -->

## Scope and inputs

Symptom description with concrete observed behavior — what failed, where, how often. Reproduction steps if known, including commands, inputs, environment, and setup. Error outputs, stack traces, logs, screenshots. Relevant code areas, recent changes, prior debugging attempts or hypotheses. Scope boundaries and time/risk constraints.

## Output and coordination

Confirmed reproduction status with exact steps used. Root-cause identification supported by an evidence chain: symptom → intermediate observations → cause mechanism. Tested and falsified hypotheses listed explicitly. Clear statement of knowns, unknowns, and blockers.

If reproduction fails, resolve environment or setup gaps before deeper diagnosis. Require a complete causal chain before accepting a diagnosis — speculation without evidence creates false confidence. Only confirmed root causes should flow into implementation work.

<!-- @worker -->

## Approach

Reproduce the failure first with a concrete command or sequence — capture exact outputs and conditions. A stable reproduction is the foundation for everything else.

Generate specific, falsifiable hypotheses tied to observable behavior. Test them by changing one variable at a time. Each observation narrows the cause space; each falsified hypothesis permanently eliminates a path. Trace data and control flow from triggering inputs to failure point — identify where behavior diverges from expectation.

The diagnosis is complete when you can name the defect mechanism (logic error, state corruption, contract mismatch, race condition, configuration issue) with direct evidence. Document rejected hypotheses, residual unknowns, and blockers separately from the root-cause statement. Unverified hypotheses are not confirmed root causes.

<!-- @criteria -->

## Completion criteria

- [ ] Root cause is identified with concrete evidence, not hypothesis alone.
- [ ] Reproduction steps are confirmed, or inability to reproduce is documented with blockers.
- [ ] Evidence chain from symptom to cause is documented and test-backed.
- [ ] Causal explanation is specific enough to guide implementation of a fix.
- [ ] Known unknowns and ruled-out alternatives are explicitly documented.
