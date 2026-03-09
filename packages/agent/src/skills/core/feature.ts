/**
 * Core Skill: Feature
 *
 * Methodology for implementing new functionality.
 */

export const FEATURE_SKILL = {
  name: 'feature',
  description: 'Implement new functionality — requirements-driven, acceptance-criteria-verified',
  trigger: 'User requests new functionality that doesn\'t exist yet, or wants to add new behavior or capabilities',
  content: `# Feature

You are implementing a new feature — introducing novel behavior into the codebase.

## Research

Understand the codebase and the problem space before designing anything.

(1) Read files directly that are clearly relevant — entry points, related modules, types, config
(2) Deploy forks to explore broader context: existing patterns, conventions, similar features, integration points
(3) While forks run, ask the user to clarify requirements and scope — what exactly should this do, and what should it NOT do?

If the feature involves unfamiliar libraries, external APIs, or technology choices — use webSearch() to research them. Do not guess at API surfaces or design around assumptions about how a library works. Look it up.

Research is complete when you can answer:
- What does the feature do, precisely?
- Where does it integrate with existing code?
- What existing patterns and conventions must it follow?
- What is explicitly out of scope?

## Plan

Create the task with startTask(). Include:

- **Title**: What the feature is
- **Details**: Design document covering functional behavior, integration design, and scope boundaries (what it does NOT do)
- **Checks**: Continuous health checks with verify functions — things that should always hold true while you work:
  - TypeScript compilation: \`"function() { var r = shell('bun run typecheck'); return { passed: r.exitCode === 0, output: r.stderr }; }"\`
  - Test suite: \`"function() { var r = shell('bun test'); return { passed: r.exitCode === 0, output: r.stdout }; }"\`
- **Acceptance**: Final proof criteria with verify functions — specific, observable outcomes that prove the feature works:
  - \`"function() { var r = shell('curl -s http://localhost:3000/api/projects'); return { passed: r.stdout.includes('['), evidence: r.stdout }; }"\`
  - \`"function() { var r = shell('bun test src/new-feature/'); return { passed: r.exitCode === 0, evidence: r.stdout }; }"\`

Verify functions must be self-contained strings (no closures). Use \`var\` for declarations. Available globals: \`shell()\`, \`forkSync()\`, \`readFile()\`.
Checks return \`{ passed: boolean, output?: any }\`. Acceptance returns \`{ passed: boolean, evidence?: string }\`.

Criteria should be things that can be mechanically checked by running a command:
- "GET /api/projects returns a list" — verified by shell command
- "Existing tests pass without modification" — verified by test runner
- "The new component renders correctly" — verified by forkSync with agent inspection

NOT: "The code is clean" or "The feature works correctly"

Iterate with the user until they agree with the approach.

## Implement

Deploy forks for parallelizable work.
For sequential or small work, implement directly via shell.
Follow the patterns and conventions you identified during research — the feature should look like it belongs in this codebase.
Message the user when forks complete or when you encounter something unexpected.

Checks run automatically when you finish a work chain — if any fail, you'll receive the results and should fix the issues before continuing.

## Verify

Call validate() to mechanically execute all acceptance criteria verify functions.
If any fail, diagnose why, fix, and re-validate.
`
} as const
