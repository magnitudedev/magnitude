---
name: formulate-abstractions
description: Formulate clear semantic abstractions from desired behavior before committing to implementation. Use when designing or revising terminology, entities, boundaries, responsibilities, relationships, high-level architecture, or a plan whose foundational shapes are unclear, disputed, or constrained by the current implementation.
---

# Overview

Formulate the broad semantic shapes of a system before designing their implementation. Use this process either within a design conversation or while creating or revising an existing plan; adapt the depth and presentation to the surrounding task.

## Phase 1: Reset the frame

Bracket the current implementation and any terminology or structure it suggests. Preserve genuine requirements and constraints, but do not treat existing code, plans, or names as the natural shape of the problem.

Start from what the system must mean and do.

## Phase 2: Formulate the broad shapes

Identify:

- Core behaviors and guarantees
- Semantic entities and precise terminology
- Responsibilities and boundaries
- Relationships, ownership, and information flow
- The smallest high-level architecture that makes these concepts coherent

Prefer one meaningful concept over several overlapping ones. Avoid abstractions that merely rename implementation details, introduce indirection, or anticipate cases without evidence.

Present the proposal at the semantic level first: terminology, behavior, boundaries, and broad architecture. Keep it concise, high-signal, and easy to scan. Do not descend into files, APIs, schemas, classes, or migration details yet.

## Phase 3: Align with the user

Surface ambiguity only when it affects the broad shapes or represents a meaningful product or architectural choice. State the proposed interpretation and its consequences so the user can respond to a concrete model rather than an open-ended question.

When the user appears to want a design conversation, pause at natural decision points and offer a compact proposal for feedback. Resolve questions that can be answered through reasoning or research autonomously; reserve user attention for key semantic decisions.

Treat agreement as provisional when important evidence remains unexplored.

## Phase 4: Ground and challenge

Once the broad shapes are sufficiently aligned, investigate the relevant code, design documents, plans, constraints, and adjacent systems. Work from the proposal into progressively specific aspects, using research to:

- Fill genuine unknowns
- Test behaviors and boundaries against real integration points
- Discover missing requirements or constraints
- Challenge terminology, ownership, and assumptions
- Determine how the desired model differs from the current state

Do not let implementation friction alone distort a clearer abstraction. Do let real constraints and newly discovered behavior change it.

## Phase 5: Revise and converge

When evidence challenges the proposal, revise it at the highest level affected. Step back as far as necessary rather than patching details around a flawed premise. Confer with the user again when the revision changes a key behavior, term, boundary, or architectural choice.

Repeat formulation, alignment, and grounding until the model remains coherent under detailed inspection. Only then translate it into implementation architecture or a concrete plan, preserving an explicit connection from each lower-level choice to the agreed semantic model.
