---
id: bug
label: Bug
description: Unexpected behavior, errors, test failures, or regressions from previously working functionality.
allowedAssignees: [self]
---

# Bug

You are diagnosing and fixing a bug. Your methodology is evidence-driven: every claim must be backed by observation, and the fix must be proven.

## Suggested Task Decomposition

```
- bug: {id} (self)
  - research: {id}-research (debugger)
    OR
    group: {id}-research (self)
      - research: {id}-research-{area} (debugger) +
  - implement: {id}-impl (builder)
  - review: {id}-review (reviewer)
```

## Procedure

### Research

Build an evidence chain from symptom to root cause. Do not guess.

1. Read the code areas mentioned in the bug report, error messages, or stack traces.
2. Deploy workers to gather evidence in parallel: reproduce the bug, trace the code path, collect exact error output.
3. While workers run, ask the user for additional context — exact reproduction steps, when it started, environment details.

Evidence discipline:
- **Reproduce first.** Get a concrete command or sequence that reliably demonstrates the failure. If you cannot reproduce it, you cannot fix it — ask for more information.
- **Observe precisely.** Collect exact error messages, return values, and stack traces. "It doesn't work" is not evidence.
- **Narrow with proof.** Trace the code path from input to failure. At each step, verify your assumption with a concrete check before moving deeper.
- **Hypothesize and test.** Form specific, falsifiable hypotheses. Prove or disprove each one with evidence. Do NOT make changes based on untested hypotheses.

Research is complete when you can state: "The bug is caused by [X], here is the evidence: [Y]."

### Plan

Document:
- Reproduction steps (exact commands/sequence)
- Evidence chain: what you observed at each step
- Root cause: what specifically causes the bug, with proof
- Proposed fix: what to change and why
- Blast radius: what else could the fix affect

### Implement

Apply the minimal fix that addresses the root cause, not the symptom. If you find yourself adding special cases or try/catch blocks around the symptom, stop and reconsider whether you've found the actual cause.

After fixing, ask: does this same pattern exist elsewhere in the codebase? If so, fix those too.

### Verify

The reproduction case must flip from failure to success. No regressions — baseline tests still pass.

## Completion criteria

- Root cause identified with evidence.
- Fix applied and reproduction case flips from red to green.
- No regressions introduced.
