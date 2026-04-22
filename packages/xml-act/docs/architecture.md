# xml-act Architecture

xml-act is a streaming parser for LLM agent output in a constrained XML response format. It converts raw model output into structured events that drive the agent framework.

## Pipeline

```
Grammar (GBNF) --> Tokenizer (streaming XML) --> Parser (handler pattern) --> Events
```

Each layer has a single responsibility:

1. **Grammar** — constrains LLM output during inference so every generated turn conforms to the format. Targets GBNF for llama.cpp and compatible providers.
2. **Tokenizer** — streaming character-level XML tokenizer. Emits `Open`, `Close`, `SelfClose`, and `Content` tokens. Handles attribute parsing, CDATA, and close-tag confirmation.
3. **Parser** — converts token stream into structured events using a typed handler pattern. Manages a frame stack via a stack machine.
4. **Events** — `LensStart`/`LensEnd`, `MessageStart`/`MessageEnd`, `ToolInputStarted`/`ToolInputFieldChunk`, `ProseEnd`, `TurnEnd`, etc.

## Response Format

A turn consists of reasoning blocks, then messages and tool invocations, then a yield:

```xml
<reason about="analysis">thinking here</reason>
<message to="user">response text</message>
<invoke tool="shell">
  <parameter name="command">ls -la</parameter>
  <filter>$.stdout</filter>
</invoke>
<yield_user/>
```

Phase ordering (reasons before messages/invocations) is enforced structurally by the grammar — the model cannot produce a reasoning block after a message under constrained decoding.

## Grammar

The grammar uses two strategies for close-tag handling, chosen per tag type:

### Eager Confirmation (reason, message)

Uses a **chain architecture** with bounded lookahead. Each tag's body DFA terminates by flowing into shared continuation rules for the next element. When `<` confirms a close tag, that `<` has been consumed, so continuation rules match the next tag name without its leading `<`. When confirmation is via newline, continuation rules expect the full tag.

Close-tag confirmation uses bounded lookahead — after matching a close tag, the grammar enters trailing-whitespace states that confirm on `
` or `<`, reject on anything else, with a bounded whitespace window.

### Greedy Last-Match (parameter, filter)

Uses **recursive repetition** to achieve parallel-state behavior within standard GBNF:

```gbnf
param-body ::= buc (CLOSE buc)* CLOSE continuation
buc        ::= ([^<] | "<" [^/] | "</" [^p] | ...)*
```

At each close tag, the GBNF engine offers both "content" and "structural close" paths. The model's token choice selects the interpretation. The last close tag is structural — greedy last-match emerges naturally from how the model generates. No lookahead, no confirmation windows, no state explosion. The CFG stack handles unbounded depth.

This eliminates false commits for parameter and filter bodies, which are the most likely to contain their own close tags (shell commands, code, XML output).

See [grammar.md](grammar.md) for full details.

## Tokenizer

A streaming character-level state machine (`tokenizer.ts`). Key features:

- **Close-tag confirmation** — for `reason`/`message` close tags: mirrors the grammar's eager confirmation. After a close tag, enters `pendingClose` state, buffers up to 4 horizontal whitespace characters, confirms on `
` or `<`, rejects otherwise. For `parameter`/`filter` close tags: uses deep confirmation with continuation-prefix matching, buffering whitespace and validating the next structural element before committing. See [lenience.md](lenience.md).
- **Attribute parsing** — 7-phase parser for `key="value"`, `key='value'`, `key=value`, and boolean attributes.
- **CDATA** — `...` emitted as Content tokens, chunk-boundary safe.
- **Chunk-boundary safety** — `pendingLt` for `<` at boundaries, `pendingClose` across boundaries.

## Parser

Uses a **handler pattern** where resolution returns typed handlers that narrow their own context. See [parser.md](parser.md) for details.

Key properties:

- **Zero unsafe frame casts** in the dispatch path. TypeScript narrows through discriminant checks and generic handler types.
- **One layer** — resolution and dispatch are unified. `resolveOpenHandler` returns a `BoundOpenHandler` that knows both whether a token is structural AND how to handle it.
- **Pure handlers** — all handlers return `ParserOp[]` (stack operations + events). No side effects.
- **Exhaustive content dispatch** — a mapped type ensures every frame type has a content handler at compile time.

## Grammar-Parser Lockstep

`nesting.ts` exports `VALID_CHILDREN` — the single source of truth for valid tag nesting:

```ts
VALID_CHILDREN = {
  prose:     ['reason', 'message', 'invoke'],
  invoke:    ['parameter', 'filter'],
  reason:    [],
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
