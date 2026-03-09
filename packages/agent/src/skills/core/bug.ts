/**
 * Core Skill: Bug
 *
 * Methodology for diagnosing and fixing defects.
 */

export const BUG_SKILL = {
  name: 'bug',
  description: 'Diagnose and fix a defect — evidence-based, root-cause-proven',
  trigger: 'User reports unexpected behavior, errors, test failures, or something that worked before but doesn\'t now',
  content: `# Bug

You are diagnosing and fixing a bug. Your methodology is evidence-driven: every claim must be backed by observation, and the fix must be proven with a red/green cycle.

## Research

Build an evidence chain from symptom to root cause. Do not guess.

(1) Read the code areas mentioned in the bug report, error messages, or stack traces
(2) Deploy forks to gather evidence in parallel: reproduce the bug, trace the code path, collect exact error output
(3) While forks run, ask the user for additional context — exact reproduction steps, when it started, environment details

### Establish verification methodology

Early in research, determine how you will objectively verify the fix:
- Are there existing unit/integration tests? Run them to establish a baseline — note which pass and which are already failing (pre-existing failures are not your responsibility).
- Is there a test runner, linter, type checker, or other tooling you can use?
- Can you write a minimal reproduction as a test case?
- If none of the above, what shell command or manual check can serve as your red/green signal?

You need a concrete, repeatable verification method before you start fixing anything.

### Evidence discipline

At each step, record what you tested, what you observed, and what it means. This prevents circular debugging.

- **Reproduce first.** Get a concrete command or sequence that reliably demonstrates the failure. This is your red test. If you cannot reproduce it, you cannot fix it — ask for more information.
- **Observe precisely.** Collect exact error messages, return values, and stack traces. "It doesn't work" is not evidence.
- **Narrow with proof.** Trace the code path from input to failure. At each step, verify your assumption with a concrete check before moving deeper.
- **Hypothesize and test.** Form specific, falsifiable hypotheses. Prove or disprove each one with evidence. Do NOT make changes based on untested hypotheses.

Research is complete when you can state: "The bug is caused by [X], here is the evidence: [Y]."

## Plan

Create the task with startTask(). Include:

- **Title**: Brief description of the defect
- **Details**: Structured root cause analysis:
  - Reproduction steps (exact commands/sequence that demonstrate the failure)
  - Evidence chain: what you observed at each step of narrowing
  - Root cause: what specifically causes the bug, with proof
  - Proposed fix: what to change and why it addresses the root cause
  - Blast radius: what else could the fix affect?
  - Verification method: how you will prove the fix works (test command, script, manual check)
- **Checks**: Continuous health checks with verify functions:
  - TypeScript compilation: \`"function() { var r = shell('bun run typecheck'); return { passed: r.exitCode === 0, output: r.stderr }; }"\`
  - Related test suite: \`"function() { var r = shell('bun test src/auth/'); return { passed: r.exitCode === 0, output: r.stdout }; }"\`
- **Acceptance**: Must always include verify functions for:
  1. The reproduction case flips from failure to success: \`"function() { var r = shell('bun test reproduction.test.ts'); return { passed: r.exitCode === 0, evidence: r.stdout }; }"\`
  2. No regressions — baseline tests still pass: \`"function() { var r = shell('bun test'); return { passed: r.exitCode === 0, evidence: r.stdout }; }"\`

Verify functions must be self-contained strings (no closures). Use \`var\` for declarations. Available globals: \`shell()\`, \`forkSync()\`, \`readFile()\`.
Checks return \`{ passed: boolean, output?: any }\`. Acceptance returns \`{ passed: boolean, evidence?: string }\`.

The first acceptance criterion — the red/green flip — is non-negotiable. If you can't express the bug as a failing check that becomes a passing check, you don't understand the bug well enough.

## Implement

### Red/green cycle
1. **Red**: Call validate() to confirm the reproduction case fails
2. **Fix**: Make the minimal change to address the root cause
3. **Green**: Call validate() again — the reproduction case must now pass

Fix the root cause, not the symptom. If you find yourself adding special cases or try/catch blocks around the symptom, stop and reconsider whether you've found the actual cause.

### Systemic check
Once the fix is proven, ask: does this same issue exist elsewhere?

Characterize the root cause as a pattern and search the codebase for other instances. If the bug was caused by a recurring pattern — missing error handling on a particular API, incorrect assumption about a data format, unsafe access pattern — other call sites likely have the same problem. Fix them too.

## Verify

Call validate() to mechanically verify all acceptance criteria.
Health checks run automatically when you finish a work chain — if any fail, fix the issues.
If the systemic check found and fixed additional instances, verify those too.
`
} as const
