---
id: scan
label: Scan
description: Quick file reading or information gathering — the lightweight default for exploration.
allowedAssignees: [explorer]
---

<!-- @lead -->

## Inputs to provide the worker
- Exact files, directories, symbols, commands, or patterns to inspect.
- The concrete question to answer or facts to extract.
- Scope boundaries: what is in scope, out of scope, and depth limits.
- Any preferred output shape (for example: callsite list, config matrix, dependency map).

## Output to expect from the worker
- A direct answer to the requested question.
- Concise findings limited to requested scope.
- Evidence for each key point with file paths and relevant line ranges/sections.
- Explicit unknowns, ambiguities, or places where scope blocked certainty.
- A workspace report when findings are broad, reused by other tasks, or decision-critical.

## Coordination loop
1. Verify the response answers the asked question directly and stays within scope.
2. If findings are incomplete, unsupported, or noisy, send a tighter follow-up scan with narrowed targets.
3. Route validated findings into next tasks (plan, implement, review, research) with references preserved.
4. If repeated scans keep surfacing ambiguity, escalate to a deeper research or diagnose task.

<!-- @worker -->

## Objective
- Gather targeted codebase information quickly and accurately, with traceable evidence, so the lead can make immediate next-step decisions.

## Procedure
1. Read the provided scope and question carefully before inspecting files.
2. Inspect the specified files/paths/patterns first; only expand scope when necessary to answer correctly.
3. Extract facts relevant to the request; avoid unrelated architecture commentary.
4. Attach evidence to each important finding (path + line range/section).
5. Flag uncertainty explicitly when the available code does not support a definitive answer.
6. If results become substantial or cross-cutting, capture them in a workspace document and link it.

## Output contract
- Start with the direct answer in 1–3 bullets.
- Provide supporting findings as concise bullets mapped to evidence.
- Include explicit blockers/unknowns and the minimal next step to resolve each.
- When applicable, include a workspace document link containing organized findings.

<!-- @criteria -->

## Completion criteria
- [ ] The requested question is answered directly.
- [ ] Findings are scoped, concrete, and tied to source evidence.
- [ ] Unknowns or blockers are explicitly identified with suggested next steps.
- [ ] Substantial findings are preserved in a workspace document when appropriate.
