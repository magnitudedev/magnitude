---
name: plan
description: When a concrete, implementation-ready design or strategy is needed before building begins.
---

# Plan

Produce a concrete, implementation-ready design or strategy before building begins.

## Approach

The plan is the contract that builders will follow — it must be specific enough to implement without ambiguity.

The lead should ensure the plan names the files that will change, the interfaces involved, the ordered steps, and how to verify the result. Plans that stay abstract — not naming files or interfaces — aren't ready for implementation.

**Resolve ambiguities before presenting to the user.** Planning inevitably reveals open questions and ambiguities — the lead's job is to recursively answer these, not leave them as holes in the plan. Spin up research, scan, or diagnose tasks to fill gaps. Each answer may surface new questions; keep going until the plan is concrete end-to-end. A plan with unresolved ambiguities forces the user to do the planning work themselves, which defeats the purpose.

**Present a clean view with key decisions visible.** When the plan is ready for user review, the user should see a coherent proposal — not a raw working document. Surface the approach, the key decisions made and their rationale, and any remaining tradeoffs the user needs to weigh. The user's job is to evaluate direction and shape consequential choices, not to read implementation details or fill in blanks. Make those choices prominent and easy to act on.

The approved plan becomes the baseline for both implementation and review.

## Delegation

Planning is collaborative, not fire-and-forget. The lead may delegate the detailed planning work to a worker, but stays involved throughout — reviewing, identifying gaps, iterating.

**Why delegate the drafting:** A worker can do the detailed work of researching options, identifying files, sequencing steps. The lead provides oversight and catches what the worker misses.

**Why the lead stays involved:** The lead reviews the draft plan against the user's actual requirements. Gaps, ambiguities, and wrong assumptions are easier to catch early — before any building starts — than after. The lead may also spin up sub-tasks (research, scan) to fill gaps the planner identified. And the lead shapes the raw planner output into a clean proposal for the user — don't present raw working documents to the user, present polished proposals with key decisions surfaced.

**The iteration cycle:**
1. Worker produces initial plan draft, writes to `$M/plans/`
2. Lead reviews for gaps, ambiguities, missing edge cases
3. Lead sends back for revision, or spins up sub-tasks to fill gaps
4. Repeat until the plan is concrete end-to-end
5. Lead shapes the final plan into a clean user-facing proposal

**What to give the worker:** Problem statement and desired outcomes, requirements and constraints, relevant codebase context (files, modules, existing patterns, prior decisions, known limitations), prior research or scan findings, decision priorities and acceptable tradeoffs. Tell them to recursively resolve open questions — spin up their own sub-tasks for research/scan as needed.

**What to expect back:** A concrete plan with scope boundaries, recommended approach with rationale, ordered steps with specific files and modules, verification strategy tied to requirements. Critically: no unresolved ambiguities that could be answered with available information. Remaining open questions should be genuine user decisions, not research gaps the worker didn't bother to close.

## Completion

The planning is complete when:
- The plan provides a clear, ordered, implementation-ready path
- Scope, approach rationale, and verification strategy are concrete and internally consistent
- Affected files/components and sequencing dependencies are identified
- Ambiguities have been recursively resolved — the plan has no holes that could be filled with available information
- Remaining open questions are genuine user decisions, not research gaps
- The user is presented a clean view with key decisions and tradeoffs prominently visible
- The lead confirms the plan is actionable for implementation without additional discovery
