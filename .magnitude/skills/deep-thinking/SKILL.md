---
name: deep-thinking
description: When complex problems require extended, iterative reasoning that benefits from externalization and refinement.
---

# Deep Thinking

Extended, iterative reasoning for complex problems that benefit from externalization and refinement.

## Approach

Thinking improves through externalization. Writing reasoning down in `$M/thoughts/` makes it inspectable, revisable, and iterable. You can revisit earlier reasoning, refine it, spot contradictions, and develop ideas through back-and-forth rather than trying to hold everything in working memory.

Deep thinking is a tool, not a phase — it can happen at any point during planning, debugging, ideation, or implementation when complexity demands it. Use it when: weighing multiple interacting constraints, designing non-obvious architectures, tracing complex causal chains, or any situation where "let me think about this more carefully" produces better results than a quick answer.

The workspace at `$M/thoughts/` is shared across all agents — thoughts written there can be read and built upon by workers, making collaborative reasoning possible.

## Delegation

Deep-thinking is rarely a standalone delegated task. It's a technique — the lead uses it directly, or instructs workers to use it within larger tasks.

**When the lead uses it directly:** Write your reasoning to `$M/thoughts/` as you work through a complex problem. This is for the lead's own benefit — externalizing helps you think more clearly, and the written trail lets you revisit and revise as understanding evolves.

**When to instruct a worker to use it:** For tasks where the reasoning trail matters as much as the conclusion — complex planning, large option spaces, subtle tradeoffs, problems where the answer depends on how you frame the question. Tell the worker to write their reasoning to `$M/thoughts/` as they work, not after — the point is to externalize incrementally.

**Collaborative reasoning:** Because `$M/thoughts/` is shared, the lead can read a worker's in-progress reasoning and course-correct early, or the worker can build on the lead's earlier thinking. This is especially useful in ideation and planning — the lead and worker can develop ideas together through the shared workspace rather than exchanging final deliverables.

**What to expect back:** Structured reasoning in `$M/thoughts/` that shows the iterative process — not just final conclusions, but how they arrived at them. Plus a summary of conclusions, open questions, and recommended next steps in their message to the lead.

## Completion

The deep thinking is complete when:
- The problem has been reasoned through with sufficient depth
- Key conclusions are supported by the reasoning trail in `$M/thoughts/`
- Open questions and remaining uncertainties are explicitly identified
- The reasoning is accessible for downstream work (planning, ideation, implementation)
