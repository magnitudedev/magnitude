---
name: scan
description: When doing quick, targeted information gathering from specific files, directories, or patterns.
---

# Scan

Quick, targeted information gathering from specific files, directories, or patterns.

## Approach

Use this skill for fast, bounded lookups — not broad investigation. The value of a scan is speed and precision. When the question needs deeper investigation or the scope is unclear, use **research** or **explore-codebase** instead.

Scans work best with specific targets: exact files, directories, symbols, or patterns. A concrete question to answer. Clear scope boundaries and depth limits.

## Delegation

When assigning a worker to scan, share in your spawn message:

- Specific targets: exact files, directories, symbols, or patterns
- A concrete question to answer
- Clear scope boundaries and depth limits
- Preferred output shape if relevant (callsite list, dependency map, config matrix)

Expect back: direct answer to the question with evidence (file paths and line ranges for each key finding), explicit unknowns rather than gaps papered over with hedging, and a workspace document when findings are broad or reused by other tasks.

Repeated ambiguity from scans signals the question needs a deeper research or debug task.

## Worker Guidance

Read specified targets first — expanding scope costs time and dilutes signal. Findings without source references (file path + line range) can't be verified and tend to get sent back. Uncertainty flagged explicitly is more useful than confident-sounding guesses.

Lead your response with the direct answer, then supporting evidence, then unknowns and what would resolve them. Substantial or cross-cutting findings belong in a workspace document.

## Quality Bar

- The requested question is answered directly.
- Findings are scoped, concrete, and tied to source evidence.
- Unknowns or blockers are explicitly identified with suggested next steps.
- Substantial findings are preserved in a workspace document when appropriate.

## Skill Evolution

Update this skill when:
- Certain scan patterns recur — add guidance on where to look first.
- The user has preferences about scan depth or output format.
