---
name: diagnose
description: When hypothesis-driven root-cause isolation is needed for bugs, failures, or unexpected behavior.
---

# Diagnose

Hypothesis-driven root-cause isolation for bugs, failures, or unexpected behavior.

## Approach

The goal is diagnosis, not implementation. Diagnosis is done when you can name the defect mechanism with direct evidence — not before.

The lead should ensure the worker reproduces the failure first, then follows a hypothesis-driven process: generate specific falsifiable hypotheses, test them against observable behavior, and trace the causal chain from symptom to root cause. Require a complete causal chain before accepting a diagnosis — speculation without evidence creates false confidence.

## Delegation

Diagnosis is a research task — the worker investigates and reports back, the lead evaluates the evidence quality. The lead doesn't typically diagnose things itself; that's what workers are for.

**What to give the worker:** The more context you provide upfront, the faster they'll converge. Share: symptom description with concrete observed behavior (what failed, where, how often), reproduction steps if known, error outputs, stack traces, logs, relevant code areas, recent changes, any prior debugging attempts or hypotheses you already have. Also share scope boundaries and time/risk constraints — some bugs are worth a deep investigation, some just need a quick fix.

**What to expect back:** A structured report (written to `$M/reports/`) with: confirmed reproduction status and exact steps, root cause identification with an evidence chain tracing from symptom through intermediate observations to the defect mechanism, hypotheses they tested and which ones they ruled out (falsified hypotheses are valuable — they prevent going in circles), and a clear statement of what's known, what's unknown, and what's blocking further progress.

**Evaluating the diagnosis:** A diagnosis without a complete causal chain is not done — send it back. If the worker can't name the defect mechanism with direct evidence, they haven't diagnosed it yet, they've just narrowed the search space. That's progress, but it's not a complete diagnosis. Only confirmed root causes should flow into implementation work.

## Completion

The diagnosis is complete when:
- Root cause is identified with concrete evidence, not hypothesis alone
- Reproduction steps are confirmed, or inability to reproduce is documented with blockers
- Evidence chain from symptom to cause is documented
- Causal explanation is specific enough to guide implementation of a fix
- Known unknowns and ruled-out alternatives are explicitly documented
