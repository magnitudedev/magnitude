---
name: debug
description: When hypothesis-driven root-cause isolation is needed for bugs, failures, or unexpected behavior.
---

# Debug

Hypothesis-driven root-cause isolation for bugs, failures, or unexpected behavior.

## Approach

The goal is to understand *why* something is broken, not to fix it. Diagnosis is done when you can name the defect mechanism with direct evidence — not before.

**Reproduce first** — Get the failure to happen reliably with a concrete command or sequence. Capture exact outputs and conditions. A stable reproduction is the foundation for everything else. If reproduction is inconsistent, that tells you something: timing, environment, data shape, or shared state.

**Form hypotheses** — Based on the symptom and relevant code, what could cause this? Generate specific, falsifiable hypotheses tied to observable behavior.

**Test hypotheses** — Change one variable at a time. Run things. Read logs. Inspect runtime state. Don't just read code and guess. Each observation narrows the cause space; each falsified hypothesis permanently eliminates a path.

**Trace to root cause** — Follow data and control flow from triggering inputs to failure point. Identify where behavior diverges from expectation. The diagnosis is complete when you can name the defect mechanism (logic error, state corruption, contract mismatch, race condition, configuration issue) with direct evidence.

## Delegation

When assigning a worker to debug, share in your spawn message:

- Symptom description with concrete observed behavior — what failed, where, how often
- Reproduction steps if known, including commands, inputs, environment, and setup
- Error outputs, stack traces, logs, screenshots
- Relevant code areas, recent changes, prior debugging attempts or hypotheses
- Scope boundaries and time/risk constraints

Expect back: confirmed reproduction status with exact steps, root-cause identification with an evidence chain (symptom → intermediate observations → cause mechanism), tested and falsified hypotheses listed explicitly, and clear statement of knowns, unknowns, and blockers.

Require a complete causal chain before accepting a diagnosis — speculation without evidence creates false confidence. Only confirmed root causes should flow into implementation work.

## Worker Guidance

Reproduce the failure first. A stable reproduction is the foundation for everything else.

Focus on *why*, not *what to do about it*. Your job is diagnosis, not implementation. If you make temporary changes for debugging (added logging, test scripts), note them in your report.

If you discover something critical or unexpected beyond the original issue, message the lead immediately.

Deliver a structured report: symptom, root cause with file:line references, evidence (what you ran/checked), recommended fix direction. Write to `$M/reports/` and link in your message.

## Quality Bar

- Root cause is identified with concrete evidence, not hypothesis alone.
- Reproduction steps are confirmed, or inability to reproduce is documented with blockers.
- Evidence chain from symptom to cause is documented.
- Causal explanation is specific enough to guide implementation of a fix.
- Known unknowns and ruled-out alternatives are explicitly documented.

## Skill Evolution

Update this skill when:
- A class of bugs recurs — add notes on where to look and what patterns to test.
- Debugging techniques prove particularly effective for this codebase.
- The user has preferences about diagnosis depth or report format.
