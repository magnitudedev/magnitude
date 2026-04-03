---
id: bug
label: Bug
description: Unexpected behavior, errors, test failures, or regressions from previously working functionality.
allowedAssignees: []
---

<!-- @lead -->

## Suggested decomposition

```
- bug: {id}
  - research: {id}-research (debugger)
    OR
    group: {id}-research
      - research: {id}-research-{area} (debugger) +
  - implement: {id}-impl (builder)
  - review: {id}-review (reviewer)
```

## Orchestration procedure

1. Build an evidence chain from symptom to root cause before any implementation begins.
   - Deploy debuggers to gather evidence: reproduce the bug, trace code paths, and collect exact error output.
   - Ask the user for additional context: exact reproduction steps, when it started, and environment details.
2. Require a concrete bug-fix plan grounded in proof.
   - Document exact reproduction steps.
   - Document the evidence chain from input to failure.
   - Document the specific root cause with proof.
   - Document the proposed fix and expected blast radius.
3. Execute a minimal root-cause fix, not a symptom patch.
   - Apply the smallest change that addresses the demonstrated cause.
   - If proposed changes rely on untested assumptions, pause and gather more evidence.
   - If the same root-cause pattern appears elsewhere, address those instances deliberately.
4. Verify the fix end-to-end.
   - Confirm the reproduction case flips from failure to success.
   - Confirm baseline checks still pass and no regressions are introduced.

## Oversight responsibilities

- Enforce evidence discipline throughout:
  - Reproduce first; if reproduction is not possible, require more information.
  - Require exact error messages, return values, and stack traces.
  - Require code-path tracing with concrete checks at each step.
  - Require specific, falsifiable hypotheses and explicit prove/disprove outcomes.
- Reject guesses and speculative fixes; require every claim to be backed by observation.
- Ensure implementation remains anchored to the proven root cause.
- Ensure verification includes both the original failure mode and regression coverage.

<!-- @criteria -->

## Completion criteria

- [ ] Root cause is identified with explicit supporting evidence.
- [ ] A minimal fix addressing the root cause is applied.
- [ ] The reproduction case flips from failure to success.
- [ ] Baseline validation confirms no regressions were introduced.
