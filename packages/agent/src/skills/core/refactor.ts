/**
 * Core Skill: Refactor
 *
 * Methodology for restructuring code without changing behavior.
 */

export const REFACTOR_SKILL = {
  name: 'refactor',
  description: 'Restructure code without changing behavior — invariant-preserving, incremental',
  trigger: 'User wants to restructure or reorganize code without changing its behavior, or explicitly says "refactor" or "clean up"',
  content: `# Refactor

You are refactoring code — restructuring it without changing its external behavior. The constraint is absolute: everything must work exactly the same when you're done.

## Research

Understand the current structure completely before changing anything.

(1) Read all the code being refactored — not just the entry point, but the full scope of what will change
(2) Deploy forks to map dependencies: all callers, all consumers, all tests that exercise this code
(3) While forks run, clarify the goal with the user — "refactor" is vague. What specific structural problem are they solving?

### Establish verification methodology

Determine how you will prove behavioral equivalence at each step:
- What tests exist? Run them to establish a baseline. Note which pass and which are already failing — pre-existing failures are not your concern, but you must not introduce new ones.
- Is there a type checker, linter, or build step? Run it and record the baseline.
- If test coverage is thin or nonexistent, surface this to the user — refactoring without verification is high risk. Consider whether characterization tests should be written first.

You need a concrete, repeatable way to confirm "nothing changed" before you start restructuring.

### Research focus
- What is the current structure and what specifically is wrong with it? Not "it's messy" — what concrete problem does the messiness cause?
- What is the complete dependency surface — every caller, every consumer, every import?
- What invariants must hold? Same inputs, same outputs, same side effects, same error behavior.

Research is complete when you can describe: the current structure, its specific problems, every dependency, and the invariants that must not break.

## Plan

Create the task with startTask(). Include:

- **Title**: What is being refactored and why
- **Details**: Before/after design:
  - Current structure and its specific problems
  - Target structure and what concretely improves (not "cleaner" — what problem goes away?)
  - Invariants that must be preserved
  - Verification method: what you will run after each step to confirm equivalence
  - Migration strategy: ordered sequence of incremental steps
  - Rename before restructure
  - Never change behavior and structure in the same step
- **Checks**: Continuous baseline verification with verify functions:
  - TypeScript compilation: \`"function() { var r = shell('bun run typecheck'); return { passed: r.exitCode === 0, output: r.stderr }; }"\`
  - Full test suite: \`"function() { var r = shell('bun test'); return { passed: r.exitCode === 0, output: r.stdout }; }"\`
  - Linter: \`"function() { var r = shell('bun run lint'); return { passed: r.exitCode === 0, output: r.stderr }; }"\`
- **Acceptance**: Must center on behavioral equivalence with verify functions:
  1. Tests that passed at baseline still pass: \`"function() { var r = shell('bun test'); return { passed: r.exitCode === 0, evidence: r.stdout }; }"\`
  2. Structural improvement achieved: use forkSync for agent-based verification if needed
  3. No unexpected interface changes: \`"function() { var r = shell('bun run typecheck'); return { passed: r.exitCode === 0, evidence: 'Types check cleanly' }; }"\`

Verify functions must be self-contained strings (no closures). Use \`var\` for declarations. Available globals: \`shell()\`, \`forkSync()\`, \`readFile()\`.
Checks return \`{ passed: boolean, output?: any }\`. Acceptance returns \`{ passed: boolean, evidence?: string }\`.

If existing tests need modification, that is a red flag — it likely means behavior changed.

## Implement

Work incrementally. Every step must leave the codebase fully working.

For each structural change:
1. Confirm baseline matches established state (checks run automatically when you become idle)
2. Make one structural change
3. Wait for automatic check results — they must match baseline
4. If new failures appear, you changed behavior — revert and reconsider

Never change behavior and structure simultaneously. If the refactor reveals that behavior NEEDS to change, that's a separate conversation with the user.

## Verify

Call validate() to mechanically verify all acceptance criteria.
Health checks run automatically — confirm the full verification suite matches baseline (same passes, same pre-existing failures, no new failures).
Confirm no public interfaces changed unexpectedly (same exports, same signatures, same behavior).
If any test needed modification during implementation, investigate and justify to the user why behavior had to change.
`
} as const
