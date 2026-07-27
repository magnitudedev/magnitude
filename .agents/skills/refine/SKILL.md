---
name: refine
description: Refine changes to ensure correctness and high code quality
---

# Overview

This skill documents a generic process for iterative code refinement.
It is intented to be used when code changes are in place, but their correctness and quality is not yet guaranteed.

The objective of this skill's use is to prescribe an iterative loop that can be followed to ensure correctness and optimize for code quality.

## Phase 1: Grounding

Before reviewing the code, it is important to understand the changes, the context behind them, and any requirements or constraints they are intended to follow.

If you are the agent which made these changes, you may already have some of this context, however you still may be missing some.

Gather the following information:
- The actual code changes, staged and unstaged (use git diff and/or read files directly as appropriate)
- Any plan the changes are based on
- Any relevant design docs
- Related code and architecture

Use the code changes to inform what other code and design docs you need to read to get the full picture.

### Plan documents

Plan documents represent some point-in-time intended delta.
Consider that a plan may prescribe changes that were adapted in-flight.
The plan may not serve as an up-to-date source of truth for what was meant to be implemented, and may be missing information or contain outdated information.
The plan should serve as one piece of evidence in your understanding of the changes being undergone and what they may or may not be doing correctly, rather than some ultimate source of truth.

## Design documents

Design documents serve as a semantic basis for the intented project state.
These documents, however, may not be fully up to date.
Relevant design documents should be read, but their content should be treated as evidence rather than ultimate truth.

## Related code

Use findings from the code changes and design documents to identify related code - code that touches or integrates with the changes, as well as code which follows similar patterns or is architectually related. This will give you a better understanding of whether the current code is following established practices, whether there are duplicated patterns, or whether architectural changes or new abstractions may be necessary.

## Phase 2: Requirement mapping

Use the documents, established code patterns, and any other pieces of evidence to establish clear standards for the changes.
- What code quality practices are established and are applicable?
- What archiectural patterns are relevant?
- What are the user's explicit requirements?
- What requirements were in the plan?

These requirements should guide your search for any issues that conflict with them.

## Phase 3: Issue mapping

### Issue refinement

When you identify issues, you must carefully audit each one rather than assuming they should be fixed without thinking through the actual conditions under which each issue is relevant and the conseque in the most obvious way.

For correctness issues, ask the following:
- Under what conditions does this issue actually arise? Could it ever actually happen in practice?
- How complex is it to solve this issue properly?
  - Solving an issue that never occurs in practice and requires a complex solution for example, is not worthwhile
- Does this issue hint at a broader architectural issue?
  - A correctness issue that is better solved with a broader architectural adjustment should not be haphazardly patched, but should rather serve as justification for the broader architectural change.

For quality issues, ask the following:
- Can this be unambiguously resolved, or does it demand a higher architectural decision

For architectural issues:
- What is the breadth of the issue, with respect to the current changes, as well as existing code?
- Once the scope is understood, you must redesign existing patterns as needed and **formulate abstractions**

### Correctness issues

### Quality issues

Identified as:
(1) **failure** to use or integrate with a cleaner, better pattern that already exists
(2) **failure** to uphold an established abstraction boundary
(3) **failure** to uphold language-specific code patterns (proper types, Effect usage, idiomatic Rust)

If the quality issue is systemic rather than one-off, it must be audited in full.

### Architectural issues

The hardest yet most important issue to identify is when the existing architecture is inherently flawed and needs to be reworked, or demands the formation of a new pattern.

Architectural issues are more broadly understable as one of:
(1) **failure** to adopt a unified abstraction for a common pattern
(2) **failure** to create abstractions that actually make sense and have clear boundaries

Creating the wrong abstractions, abstractions without clear boundaries, or indirect/redundant/unnecessary abstractions, is an architectural issue.
Failing to identify a pattern or complexity that justifies the creation of an abstraction is an architectural issue.

Such issues can only be resolved through careful system analysis and abstraction formulation

## Phase 4: Issue resolution

If there are architectural issues, solve these first via **abstraction formulation**.
Once the broad shapes are rebuilt correctly, resolve any correctness and quality issues.

### Abstraction formulation

The highest form of re-assessing foundational issues is the analy

To formulate abstractions:
(1) Analyze the desired behaviors
(2) Design and iterate on the clearest semantic entities and their relationships with one another
(3) Discarding any preconceptions seeded by the current state of the codebase
(4) Work backwards from the clear semantic organization into architectural shapes that also adopt the correct language-specific mechanisms and patterns
(4) Delta these pattterns to what currently exists to understand the required work

## Workflow

Repeat 1-4 until the code has no correctness issues, meets a high quality standard, and is optimally architected.
