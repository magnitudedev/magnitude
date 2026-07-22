# Design documents

The documents in this directory are the high-level source of truth for Magnitude's architecture and behavior. They describe the system that should exist, independently of the current implementation.

A design document should be sufficient for a capable engineer to reproduce the intended architecture, invariants, behavior, boundaries, and failure semantics without first reading the code. It should remain durable as files, types, functions, and internal implementation details change.

## What belongs here

Design documents capture durable decisions such as:

- system responsibilities and ownership boundaries;
- externally observable behavior and contracts;
- lifecycle, state-transition, concurrency, and data-flow rules;
- correctness and safety invariants;
- failure classification and recovery behavior;
- persistence, caching, identity, and invalidation rules;
- deliberate limitations and excluded behavior;
- acceptance criteria that distinguish a conforming implementation.

Include enough detail to make the intended system unambiguous. Prefer concepts, guarantees, and stable domain terminology over descriptions of the current call graph.

Avoid unnecessary implementation details such as line numbers, private helper names, incidental type layouts, or lists of files that merely reflect the current code organization. Mention a concrete API, type, algorithm, or implementation constraint only when it is itself part of the design or materially clarifies a guarantee.

Design documents are normative unless they explicitly label a section as historical context, an example, or a future possibility. Code is evidence of the current implementation, not authority to silently contradict a design document.

## Applicability front matter

Every design document must begin with YAML front matter containing `applies_to`:

```yaml
---
applies_to:
  - packages/example/**
  - packages/shared/src/contract.ts
---
```

Each entry is a glob pattern relative to the project root. Use forward slashes. `*` matches within one path segment and `**` matches across directories. Prefer a YAML list, including when only one pattern is needed, and prefer multiple clear patterns over brace expansion or other glob-dialect-specific syntax.

Patterns identify the code and configuration governed by the document, not files that happen to mention the same subject. Keep them broad enough to survive ordinary refactoring but narrow enough that a match means the document is materially relevant.

Multiple design documents may apply to the same path. In that case all matching documents apply; overlap is expected where architectural concerns intersect.

`AGENTS.md` files are repository instructions and do not require `applies_to` front matter.

applies_to should have no more than ~10 patterns. If the number of patterns becomes large consolidate to broader patterns even if less precise.

## Required workflow

Before changing a file, identify and read every design document whose `applies_to` pattern matches that project-root-relative path. Follow all applicable invariants and acceptance criteria.

Use the repository helper to find applicable documents:

```bash
bun design-docs path/to/file-or-directory
bun design-docs --changed
bun design-docs --all
```

Add `--explain` to show the matched paths and patterns.

When a change affects architecture or observable behavior:

1. Update the applicable design document in the same change.
2. Describe the intended durable behavior rather than narrating the patch.
3. Update `applies_to` when ownership moves, new governed areas are added, or old areas cease to be governed.
4. Add or revise acceptance criteria when the change introduces a new guarantee.
5. Verify the implementation and tests against the updated design.

When adding a new subsystem or durable behavior with no adequate design document, create one in the most relevant subdirectory and give it accurate `applies_to` patterns. Extend an existing document when the behavior belongs to one coherent design; create a separate document when it has an independent contract or lifecycle.

Refactoring that does not intentionally change behavior must preserve the design. If the implementation and a design document disagree, determine whether the implementation is defective or the intended design has changed. Do not edit the document merely to rationalize accidental implementation behavior.

## Maintenance and enforcement

Changes to governed code should be reviewed for both implementation correctness and design conformance. Reviewers and agents must check that:

- all matching design documents were considered;
- the implementation satisfies their invariants and acceptance criteria;
- intentional design changes update the documents;
- `applies_to` patterns still cover the implementation's real ownership boundaries;
- new transient implementation states or limitations have not leaked into durable product contracts without an explicit design decision;
- documents remain internally consistent and do not contradict one another.

When paths move, audit the front matter of all design documents rather than relying on the old patterns to keep matching. When a document is superseded, update references and applicability atomically; remove or clearly mark obsolete authority so two conflicting sources of truth do not remain active.

Automated tooling may use `applies_to` to select required design context for changed files and to flag governed changes whose design documents were not considered. Such tooling is an enforcement aid; it does not replace judgment about indirect architectural impact that falls outside an existing glob.

## Quality standard

A strong design document answers:

- What does this system own, and what does it not own?
- What must always be true in the normal case?
- What inputs and events cause state to change?
- Which outcomes are valid domain results, and which indicate failure?
- How do concurrency, persistence, caching, and recovery remain correct?
- What may callers rely on?
- What limitations are intentional?
- How can an implementation be shown to conform?

Keep documents concise enough to remain usable, but never omit a necessary guarantee merely to shorten them. The goal is a stable specification of the correct system, not a summary of the current code.
