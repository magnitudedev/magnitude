---
id: web-test
label: Web Test
description: Browser-based verification of web application behavior and visual state.
allowedAssignees: [browser]
---

<!-- @lead -->

## Inputs to provide the worker
- Target URL/page and required setup context (authentication state, seed data, feature flags, environment).
- Concrete verification points covering behaviors and visual states to test.
- Expected behavior for each verification point, plus known regression context where relevant.
- Reference artifacts (specs, screenshots, recordings, design references) when available.
- Required edge-case coverage (viewport/device modes, interaction paths, state permutations).

## Output to expect from the worker
- Explicit pass/fail result for every verification point.
- Evidence for each result (screenshots and concrete observation notes).
- Observed-versus-expected comparison per point.
- Sequence-level notes for dynamic behavior (timing, transitions, state changes).
- Unexpected findings and any blocked checks with reproducible details.

## Coordination loop
1. Assign with URL, prerequisites, and an enumerated verification checklist.
2. Resolve setup/access blockers before requesting final pass/fail conclusions.
3. Review evidence quality per point; require retest when proof is weak or ambiguous.
4. Route failed checks into implementation follow-ups with reproducible repro details.
5. Re-run targeted web-test validation after fixes and close only when all criteria are satisfied.

<!-- @worker -->

## Objective
- Verify specified web behaviors and visual states in-browser and return evidence-backed outcomes for each requested check.

## Procedure
1. Prepare the test context by navigating to the target URL and establishing required setup (login, data, flags, viewport/device mode).
2. Confirm baseline state, then execute each verification point exactly as specified.
3. Determine pass/fail for each point against expected behavior using concrete, reproducible observations.
4. Capture screenshots and sequence notes for dynamic behavior, including timing, transitions, and state changes.
5. Execute required edge-case checks and compare outcomes to expected behavior.
6. Compile final results with per-point verdicts, evidence references, and unexpected findings.

## Output contract
- Return a verification checklist with one entry per point containing: verification point, expected behavior, observed behavior, pass/fail, and evidence reference.
- Return an edge-case section for requested alternate modes/paths/states.
- Return an unexpected behavior section even when outside explicit test points.
- Return blockers/limitations for any unverified checks, including exact failing step and missing prerequisites.
- If expectations are ambiguous, flag uncertainty and request clarification instead of guessing a verdict.

<!-- @criteria -->

## Completion criteria
- [ ] All specified verification points are executed with concrete evidence.
- [ ] Pass/fail status is explicitly provided for each verification point.
- [ ] Observed behavior is compared directly against expected behavior for each point.
- [ ] Unexpected behavior is documented with reproducible detail.
- [ ] Any unverified checks are explicitly listed with blockers and acceptance status.