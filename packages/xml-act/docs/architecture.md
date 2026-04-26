# xml-act Architecture

xml-act is a streaming parser for LLM agent output in a constrained XML response format. It converts raw model output into structured events that drive the agent framework.

## Pipeline

```
Grammar (GBNF) --> Tokenizer (streaming XML) --> Parser (handler pattern) --> Events
```

Each layer has a single responsibility:

1. **Grammar** — constrains LLM output during inference so every generated turn conforms to the format. Targets GBNF for llama.cpp and compatible providers.
2. **Tokenizer** — streaming character-level XML tokenizer. Emits `Open`, `Close`, `SelfClose`, and `Content` tokens. Handles attribute parsing and CDATA. Known structural close tags are emitted immediately; unknown close tags become Content.
3. **Parser** — converts token stream into structured events using a typed handler pattern. Manages a frame stack via a stack machine. Implements tentative close confirmation with greedy last-match semantics.
4. **Events** — `LensStart`/`LensEnd`, `MessageStart`/`MessageEnd`, `ToolInputStarted`/`ToolInputFieldChunk`, `ProseEnd`, `TurnEnd`, etc.

## Response Format

A turn consists of reasoning blocks, then messages and tool invocations, then a yield:

```xml
<think about="analysis">thinking here</think>
<message to="user">response text</message>
<invoke tool="shell">
  <parameter name="command">ls -la</parameter>
  <filter>$.stdout</filter>
</invoke>
<yield_user/>
```

Phase ordering (reasons before messages/invocations) is enforced structurally by the grammar — the model cannot produce a reasoning block after a message under constrained decoding.

## Grammar

All body types use a single **greedy last-match** strategy:

```gbnf
param-body ::= buc (CLOSE buc)* CLOSE continuation
buc        ::= ([^<] | "<" [^/] | "</" [^p] | ...)*
```

At each close tag, the GBNF engine offers both "content" and "structural close" paths. The model's token choice selects the interpretation. The last close tag is structural — greedy last-match emerges naturally from how the model generates. No lookahead, no confirmation windows, no state explosion. The CFG stack handles unbounded depth.

This applies to all body types: `think`, `message`, `parameter`, and `filter`. The continuation rules differ per tag type. Top-level bodies (`think`, `message`) confirm on the next top-level tag or yield. Inside invoke blocks, the last parameter uses **deep confirmation** — `</parameter>` is confirmed through `</invoke>` all the way to the next top-level tag as a single grammar unit. Non-last parameters confirm on the next parameter, filter, or invoke close.

Tool names and parameter names are **constrained per-tool** in the grammar. The grammar enumerates valid tool names and, for each tool, the valid sequence of parameter names. Grammar size scales with the number of tools and their parameter schemas.

Whitespace between structural tags is unbounded — the grammar uses an unconstrained `ws` rule rather than a bounded whitespace window.

See [grammar.md](grammar.md) for full details.

## Tokenizer

A streaming character-level state machine (`tokenizer.ts`). Key features:

- **Immediate close-tag emission** — known structural close tag names (`invoke`, `parameter`, `filter`, `think`, `message`) are emitted as `Close` tokens immediately. Unknown close tags (e.g. `</div>`) are emitted as `Content`. No confirmation, no lookahead, no buffering for close tags.
- **Attribute parsing** — 7-phase parser for `key="value"`, `key='value'`, `key=value`, and boolean attributes.
- **CDATA** — `...` emitted as Content tokens, chunk-boundary safe.
- **Chunk-boundary safety** — `pendingLt` for `<` at boundaries.

## Parser

Uses a **handler pattern** where resolution returns typed handlers that narrow their own context. See [parser.md](parser.md) for details.

Key properties:

- **Zero unsafe frame casts** in the dispatch path. TypeScript narrows through discriminant checks and generic handler types.
- **One layer** — resolution and dispatch are unified. `resolveOpenHandler` returns a `BoundOpenHandler` that knows both whether a token is structural AND how to handle it.
- **Pure handlers** — all handlers return `ParserOp[]` (stack operations + events). No side effects.
- **Exhaustive content dispatch** — a mapped type ensures every frame type has a content handler at compile time.
- **Tentative close confirmation** — `dispatch.ts` maintains a `pendingCloseStack`. When a `Close` token matches the current frame, it is held tentatively rather than applied immediately. The next token resolves it: whitespace Content extends the buffer; valid structural continuation (Open, SelfClose, or Content starting with `<`) confirms; non-whitespace non-`<` Content rejects (the close tag and buffered whitespace are emitted as content). Cascade is supported — confirming a `</parameter>` may immediately make `</invoke>` tentative if no more parameters follow. This implements greedy last-match at the parser level, using nesting context and schema info that the tokenizer does not have.

## Confirmation Layers

Close-tag confirmation operates at two levels:

1. **Grammar** (primary) — under constrained decoding, the greedy last-match pattern (`buc (close buc)* close continuation`) is the primary confirmation mechanism. The continuation rule forces the last close tag to be structural. The model cannot produce a false close under grammar constraint.
2. **Parser** (fallback) — when constrained decoding is not active, the parser's `pendingCloseStack` tentative close mechanism implements equivalent logic. It holds close tags tentatively and confirms or rejects based on what follows.

The tokenizer is a simple structural scanner — it emits close tags immediately with no confirmation. This keeps the tokenizer small and puts confirmation logic where nesting context and schema info are available (grammar and parser).

## Grammar-Parser Lockstep

`nesting.ts` exports `VALID_CHILDREN` — the single source of truth for valid tag nesting:

```ts
VALID_CHILDREN = {
  prose:     ['think', 'message', 'invoke'],
  invoke:    ['parameter', 'filter'],
  think:    [],
  message:   [],
  parameter:    [],
  filter:    [],
}
```

Both the grammar builder and the parser import this constant. Compile-time type assertions catch divergence. Adding a new structural tag to `VALID_CHILDREN` automatically propagates to grammar output and forces a corresponding parser handler.

## Stack Machine

The parser drives a stack machine with typed frames and three modes:

- **`active`** — normal parsing, ops applied to the frame stack
- **`observing`** — entered on yield, tracks post-yield content for runaway detection
- **`done`** — terminal state

Operations: `push`, `pop`, `replace`, `emit`, `observe`, `done`, `popUntil`.

## Output

The parser emits events consumed by the turn engine:

| Event | When |
|-------|------|
| `LensStart` / `LensEnd` | Reasoning block open/close |
| `MessageStart` / `MessageEnd` | Message open/close |
| `ToolInputStarted` | Invoke open |
| `ToolInputFieldChunk` | Parameter content streaming |
| `ProseEnd` | Top-level prose content |
| `TurnEnd` | Stream end, with termination reason |

## Key Files

| File | Purpose |
|------|---------|
| `grammar/grammar-builder.ts` | GBNF grammar generation |
| `tokenizer.ts` | Streaming XML tokenizer |
| `nesting.ts` | Shared valid-children constant |
| `parser/resolve.ts` | Token -> handler resolution |
| `parser/dispatch.ts` | Parser loop |
| `parser/handler.ts` | Handler type definitions |
| `parser/handlers/` | Per-tag handler implementations |
| `parser/content.ts` | Exhaustive content routing |
| `parser/flush.ts` | EOF frame cleanup |
| `parser/types.ts` | Frame type definitions |
| `parser/ops.ts` | Op helper constructors |
| `machine.ts` | Stack machine |
| `output/renderer.ts` | Result rendering (XML) |
