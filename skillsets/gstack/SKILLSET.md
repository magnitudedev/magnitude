# GStack Skillset

You have access to a comprehensive engineering workflow skillset covering the full sprint lifecycle. Each skill provides a detailed, step-by-step methodology for a specific phase of work.

## The Sprint Lifecycle

These skills follow the natural rhythm of a sprint:

**Think → Plan → Build → Review → Test → Ship → Reflect**

Each skill feeds into the next. `office-hours` produces a design doc that `plan-ceo-review` reads. `plan-eng-review` produces a test plan that `qa` picks up. `review` catches bugs that `ship` verifies are fixed. When working through a multi-phase workflow, carry forward the artifacts from each skill into the next — nothing should fall through the cracks.

---

## Available Skills

### Think & Plan

| Skill | Specialist Role | Use when... |
|-------|----------------|-------------|
| `office-hours` | YC Office Hours | Starting a new feature or project. Runs six forcing questions to reframe the problem before any code is written. Produces a design doc that feeds into every downstream skill. |
| `plan-ceo-review` | CEO / Founder | You need to challenge scope and ambition. Finds the 10-star product hiding inside the request. Four modes: Expansion, Selective Expansion, Hold Scope, Reduction. |
| `plan-eng-review` | Eng Manager | You need to lock in architecture, data flow, edge cases, and test strategy. Forces hidden assumptions into the open before building begins. |
| `plan-design-review` | Senior Designer | You need to score and refine design quality. Rates each design dimension 0–10, explains what a 10 looks like, then edits the plan to get there. Detects AI slop. |
| `plan-devex-review` | DX Lead | The work targets developers (API, CLI, SDK, docs). Benchmarks against competitor TTHW, designs the magical moment, traces friction points. Three modes: DX Expansion, DX Polish, DX Triage. |

### Design

| Skill | Specialist Role | Use when... |
|-------|----------------|-------------|
| `design-consultation` | Design Partner | Building a design system from scratch. Researches the landscape, proposes creative risks, generates realistic product mockups. |
| `design-shotgun` | Design Explorer | The user wants to explore visual options. Generates 4–6 mockup variants, presents them for comparison, collects feedback, and iterates. |
| `design-html` | Design Engineer | Turning a mockup or design into production HTML. Uses Pretext computed layout for dynamic, shippable output. Detects React/Svelte/Vue. |

### Review & Audit

| Skill | Specialist Role | Use when... |
|-------|----------------|-------------|
| `review` | Staff Engineer | Code is ready for review. Finds bugs that pass CI but blow up in production. Auto-fixes obvious issues. Includes 7 specialist review passes (security, performance, testing, API contracts, etc.). |
| `design-review` | Designer Who Codes | UI/design changes need auditing. Same methodology as `plan-design-review`, but applied to live code — fixes what it finds with atomic commits. |
| `devex-review` | DX Tester | Developer-facing changes need validation. Actually tests onboarding: navigates docs, tries the getting started flow, times TTHW, screenshots errors. Compares against `plan-devex-review` scores if available. |
| `cso` | Chief Security Officer | Security review is needed. OWASP Top 10 + STRIDE threat model. Zero-noise methodology with high confidence gate and independent finding verification. |
| `investigate` | Debugger | A bug needs root-cause analysis. Iron Law: no fixes without investigation. Traces data flow, tests hypotheses, stops after 3 failed fixes to reassess. |

### Test

| Skill | Specialist Role | Use when... |
|-------|----------------|-------------|
| `qa` | QA Lead | The application needs testing. Finds bugs, fixes them with atomic commits, re-verifies, and auto-generates regression tests. |
| `qa-only` | QA Reporter | You need a bug report without code changes. Same methodology as `qa`, report-only output. |
| `benchmark` | Performance Engineer | Performance matters. Baselines page load times, Core Web Vitals, and resource sizes. Use before and after changes to measure impact. |

### Ship & Deploy

| Skill | Specialist Role | Use when... |
|-------|----------------|-------------|
| `ship` | Release Engineer | Code is ready to ship. Syncs main, runs tests, audits coverage, pushes, and opens a PR. Bootstraps test frameworks if needed. |
| `land-and-deploy` | Release Engineer | A PR is approved and ready to merge. Handles merge, CI, deploy, and production health verification. |
| `setup-deploy` | DevOps | Deployment infrastructure needs configuring. Supports Fly.io, Render, Vercel, Netlify, GitHub Actions, and custom setups. |
| `canary` | SRE | A deploy just landed. Monitors for console errors, performance regressions, and page failures. |

### Reflect & Document

| Skill | Specialist Role | Use when... |
|-------|----------------|-------------|
| `document-release` | Technical Writer | A feature just shipped. Updates all project docs to match, catches stale READMEs. |
| `retro` | Eng Manager | It's time for a retrospective. Team-aware weekly analysis with per-person breakdowns, shipping streaks, test health trends, and growth opportunities. |
| `health` | Tech Lead | You need a codebase health assessment. Scores across multiple dimensions and tracks trends over time. |

---

## Choosing the Right Review Skill

| The work targets... | Before code (plan review) | After code (live audit) |
|--------------------|--------------------------|------------------------|
| End users (UI, web app, mobile) | `plan-design-review` | `design-review` |
| Developers (API, CLI, SDK, docs) | `plan-devex-review` | `devex-review` |
| Architecture (data flow, perf, tests) | `plan-eng-review` | `review` |
| Security | — | `cso` |

---

## Workflow Patterns

### Full feature sprint
`office-hours` → `plan-ceo-review` → `plan-eng-review` → `plan-design-review` → *build* → `review` → `qa` → `ship` → `land-and-deploy` → `canary` → `document-release`

Not every feature needs every step. Use judgment — a small backend refactor doesn't need `plan-design-review`, and a copy change doesn't need `plan-eng-review`. Match the rigor to the risk.

### Bug fix
`investigate` → *fix* → `review` → `ship` → `land-and-deploy` → `canary`

### Design exploration
`design-consultation` → `design-shotgun` → `design-html` → `design-review`

### Security audit
`cso` — standalone, run at any time

### Weekly check-in
`retro` + `health` — run together for a complete picture of team velocity and codebase quality trends


