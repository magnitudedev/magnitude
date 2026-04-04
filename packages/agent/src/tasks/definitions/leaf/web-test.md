---
id: web-test
label: Web Test
description: Browser-based verification of web application behavior and visual state.
allowedAssignees: [browser]
---

<!-- @lead -->

## Scope and inputs

Target URL and required setup context: authentication state, seed data, feature flags, environment. Concrete verification points with expected behavior for each. Reference artifacts when available: specs, screenshots, recordings, design references. Required edge-case coverage: viewport modes, interaction paths, state permutations.

## Output and coordination

Explicit pass/fail for every verification point. Evidence for each: screenshots and observation notes. Observed-vs-expected comparison per point. Sequence notes for dynamic behavior — timing, transitions, state changes. Unexpected findings and blocked checks with details.

Setup or access blockers need resolution before requesting final verdicts. Weak or ambiguous evidence needs retest. Failed checks route into implementation follow-ups with reproducible details. Re-run targeted verification after fixes.

<!-- @worker -->

## Approach

Navigate to the target URL and establish required setup — login, data, flags, viewport mode. Confirm baseline state before executing verification points.

Pass/fail is only meaningful against explicit expected outcomes. Capture screenshots and sequence notes for dynamic behavior — static snapshots miss timing and transition issues. Edge-case checks (alternate viewports, interaction paths, state permutations) raise confidence in interaction-heavy systems.

Report one entry per verification point: expected behavior, observed behavior, verdict, evidence reference. Flag unexpected behavior even outside explicit test points — it can be high-impact. If expectations are ambiguous, surface the ambiguity rather than guessing a verdict.

<!-- @criteria -->

## Completion criteria

- [ ] All specified verification points are executed with concrete evidence.
- [ ] Pass/fail status is explicitly provided for each verification point.
- [ ] Observed behavior is compared directly against expected behavior for each point.
- [ ] Unexpected behavior is documented with reproducible detail.
- [ ] Any unverified checks are explicitly listed with blockers.
