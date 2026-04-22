# Parser Architecture

## Overview

The xml-act parser is a **context-sensitive streaming XML parser**. It processes a protocol where the same tag name can be structural or literal text depending on context. Designed for streaming LLM output.

## Pipeline

```mermaid
graph LR
    A[Raw Text Chunks] --> B[Tokenizer]
    B -->|Tokens| C[Parser / Resolve]
    C -->|Bound Handlers| D[Handler.open/close]
    D -->|ParserOp array| E[Stack Machine]
    E -->|Events| F[Consumers]
```

- **Tokenizer** — pure XML syntax. Emits tokens (Open, Close, SelfClose, Content). No semantic awareness.
- **Resolve** — context-sensitive layer. Determines which tags are structural vs literal text based on the current stack frame. Returns a typed bound handler, or `undefined` (literal text).
- **Handlers** — pure functions. Receive the narrowed parent frame and attributes, return `ParserOp[]` describing stack changes and events.
- **Stack Machine** — generic state manager. Applies ops (push/pop/replace/emit/observe/done) to a frame stack.

## Context-Sensitive Resolution

The core problem: the same tag name means different things in different contexts. `<parameter>` is structural inside an invoke, but literal text inside a reason block or prose.

Resolution is split into three functions that check the current top frame:

- **`resolveOpenHandler(tagName, top)`** — switches on `tagName`, then checks `top.type`. Returns a `BoundOpenHandler` or `undefined`.
- **`resolveCloseHandler(tagName, top)`** — switches on `top.type`, then verifies `tagName` matches. Returns a `BoundCloseHandler` or `undefined`.
- **`resolveSelfCloseHandler(tagName, top)`** — handles yield tags in prose context only.

This is **correct by construction**: resolution checks the frame type before returning a handler. A `<parameter>` token can only resolve as structural when `top.type === 'invoke'`. In any other context, it returns `undefined` → literal text.

### Tag Visibility by Context

| Context (top frame) | Structural tags | Everything else |
|---------------------|----------------|-----------------|
| Prose | reason, message, invoke, yield_* | Literal text |
| Reason | — | Literal text |
| Message | — | Literal text |
| Invoke | parameter, filter | Literal text |
| Parameter | — | Literal text |
| Filter | — | Literal text |

This table is encoded as `VALID_CHILDREN` in `nesting.ts` — a single constant shared by both the grammar builder and the parser.

## Grammar–Parser Lockstep

The grammar (GBNF for constrained decoding) and the parser must agree on what tags are valid in what contexts. Both import `VALID_CHILDREN` from `nesting.ts` as their single source of truth.

```mermaid
graph TD
    N[nesting.ts — VALID_CHILDREN] --> G[Grammar Builder]
    N --> P[Parser — resolveOpenHandler]
    G -->|GBNF rules| C[Constrained Decoding]
    P -->|Structural vs literal| S[Stack Machine]
    C -.->|Guarantees valid structure| S
```

The grammar constrains the model during inference so it can only produce structurally valid output. The parser then interprets that output using the same nesting rules. Because both derive from the same constant:

- Adding a new tag to `VALID_CHILDREN` automatically generates grammar rules for it AND makes the parser resolve it structurally in the right context.
- Removing a tag removes it from both layers simultaneously.
- Compile-time type assertions catch any divergence between `VALID_CHILDREN` and the handler implementations.

## Handler Lifecycle

```mermaid
sequenceDiagram
    participant T as Tokenizer
    participant R as Resolve
    participant H as Handler
    participant M as Stack Machine

    T->>R: Token (Open "parameter")
    R->>R: top.type === 'invoke'? Yes
    R->>R: bindOpen(parameterOpenHandler, top as InvokeFrame)
    R-->>H: BoundOpenHandler
    H->>H: handler.open(attrs, ctx) → ParserOp[]
    H->>M: apply([push(ParameterFrame), emit(ToolInputFieldChunk)])
    Note over M: ParameterFrame now on top<br/>Only </parameter> resolves structurally
```

When resolution binds a handler, it captures the narrowed parent frame in a closure. The handler receives it as a typed parameter — TypeScript enforces the parent/child relationship at compile time via `OpenHandler<InvokeFrame, ParameterFrame>`.

## Type Safety

The handler pattern provides compile-time guarantees:

- **Generic handlers** — `OpenHandler<TParent, TChild>` enforces that a parameter handler can only be called with an InvokeFrame parent and can only push a ParameterFrame. TypeScript errors at definition time if the types don't match.
- **Bound handlers** — `bindOpen`/`bindClose` capture the narrowed frame from resolution. The parser loop calls `handler.open(attrs, ctx)` without knowing frame types — the binding is localized to resolution.
- **Zero unsafe casts** — no `as InvokeFrame`, `as ParameterFrame`, etc. in the dispatch path. TypeScript narrows through discriminant checks in resolution.
- **Exhaustive content dispatch** — a `ContentHandlers` mapped type ensures every frame type has a content handler. Adding a new frame type without one is a compile error.

## Pure Handlers

All handlers return `ParserOp[]` — an array of stack machine operations. No side effects. Events, errors, and stack changes are all expressed as ops:

```
push(frame) | pop | replace(frame) | emit(event) | observe | done
```

Helper constructors (`emitEvent`, `emitStructuralError`, `emitToolError`) produce emit ops. The parser loop applies the returned ops to the machine.

## Stack Machine Modes

The machine has three modes:

- **`active`** — normal parsing. Ops are applied to the frame stack.
- **`observing`** — entered when a yield tag fires. Tracks whether non-whitespace content appears after the yield (runaway detection). No ops applied.
- **`done`** — terminal. Emits `TurnEnd` with termination reason (`natural` vs `runaway`).

## Frame Mutability

Most frame fields are `readonly`. Three exceptions are mutable by design:

- `ParameterFrame.rawValue` — accumulated character-by-character during streaming
- `FilterFrame.query` — same
- `InvokeFrame.seenParams` / `fieldStates` — mutable Set and Map

These are mutated in-place rather than through `replace` ops because they accumulate per-character content where cloning would be O(n) allocations for O(n) content with no benefit — these frames are never snapshotted or replayed.

Parameter and filter frames store a reference to their parent `InvokeFrame` at open time, eliminating stack traversal in close handlers.
