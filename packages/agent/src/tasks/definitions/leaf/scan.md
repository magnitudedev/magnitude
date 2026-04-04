---
id: scan
label: Scan
description: Quick file reading or information gathering — the lightweight default for exploration.
allowedAssignees: [explorer]
---

<!-- @lead -->

## Scope and inputs

Scans work best with specific targets: exact files, directories, symbols, or patterns. A concrete question to answer. Clear scope boundaries and depth limits. Preferred output shape if relevant (callsite list, dependency map, config matrix).

## Output and coordination

Direct answer to the question with evidence — file paths and line ranges for each key finding. Explicit unknowns rather than gaps papered over with hedging. Workspace document when findings are broad or reused by other tasks.

Repeated ambiguity from scans signals the question needs a deeper research or diagnose task.

<!-- @worker -->

## Approach

The value of a scan is speed and precision. Read specified targets first — expanding scope costs time and dilutes signal. Findings without source references (file path + line range) can't be verified and tend to get sent back. Uncertainty flagged explicitly is more useful than confident-sounding guesses.

Lead your response with the direct answer, then supporting evidence, then unknowns and what would resolve them. Substantial or cross-cutting findings belong in a workspace document.

<!-- @criteria -->

## Completion criteria

- [ ] The requested question is answered directly.
- [ ] Findings are scoped, concrete, and tied to source evidence.
- [ ] Unknowns or blockers are explicitly identified with suggested next steps.
- [ ] Substantial findings are preserved in a workspace document when appropriate.
