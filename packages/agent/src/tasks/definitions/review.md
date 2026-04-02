---
id: review
label: Review
description: Completed work that needs verification for correctness, quality, and requirement coverage.
allowedAssignees: [self, reviewer]
---

# Review

You are evaluating work with adversarial rigor.

## Deliverable

A review with:
- Pass/fail verdict
- Concrete findings (not vague suggestions)
- Evidence for each finding (file paths, test results, specific code references)

## Approach

1. Check requirement coverage — does the work do what was asked?
2. Check correctness — edge cases, error handling, data flow.
3. Check for regressions — did anything break?
4. Check code quality — patterns, naming, structure.
5. Verify claims with evidence (read the code, run the tests) — do not take assertions at face value.
6. Iterate with the implementer until issues are resolved.

## Done when

- All findings are resolved or explicitly accepted.
- Work meets requirements and quality bar.
