# Grammar Strategy

The grammar constrains LLM output during inference so that every generated turn conforms to the XML response format. It targets GBNF (llama.cpp and compatible providers). This document describes the design at a strategic level — the concepts and tradeoffs behind the grammar, not the implementation details of specific rules.

## Response Format

A turn consists of zero or more reasoning blocks, followed by zero or more messages or tool invocations, followed by exactly one yield token. Reasoning blocks must come before messages and invocations — once a message or invocation appears, no further reasoning blocks are allowed.

<reason about="...">...</reason>
<message to="...">...</message>
<invoke tool="...">
  <parameter name="...">value</parameter>
  <filter>$.query</filter>
</invoke>
<yield_user/>

Reasoning blocks carry an `about` attribute identifying the lens. Messages carry a `to` attribute. Invocations carry a `tool` attribute and contain `<parameter` children (each with a `name` attribute and free-text value) and an optional `<filter>` child. Yield tokens are self-closing tags that terminate the turn.

## The Close-Tag Problem

LLMs occasionally mention their own tags inside body content — for example, a shell command that outputs XML, or documentation explaining the response format itself. The grammar must distinguish real close tags from incidental mentions.

This is hard because the close tag sequence is just bytes — there's nothing syntactically special about the "real" one versus a mention in content. The grammar must decide, at each close tag occurrence, whether it's structural or content.

## Greedy Last-Match

All body types — `reason`, `message`, `parameter`, `filter` — use a single unified strategy: **greedy last-match via recursive repetition**.

The grammar defines every body as:

```gbnf
body ::= buc (CLOSE buc)* CLOSE continuation
```

Where:
- **`buc`** (body-until-close) is a DFA matching any string that does NOT contain the close tag sequence
- **`CLOSE`** is the literal close tag (e.g. `</parameter>`)
- **`continuation`** is the rule for what comes next after this body closes

The `*` repetition is the key. At each close tag, the GBNF engine faces two valid alternatives:

1. **Take another iteration** of the `*`: this close tag is content. Match more body, then another close tag.
2. **Exit the `*`** and match the final `CLOSE continuation`: this close tag is structural.

The engine computes the **union** of valid next tokens for both paths. Both alternatives remain live. The **model** picks a token — and its choice implicitly selects the interpretation.

### Why Greedy Last-Match Is Emergent

The model generates content as long as it has content to express. Each close tag encountered along the way gets absorbed by the `*` (content path). When the model is done, it stops generating body characters. The engine exits the `*` and the final close tag chains to the continuation.

The **last** close tag is structural because it's the one where the model chose to stop. All earlier ones were content because the model kept going. No explicit "greedy" logic is needed — it falls out naturally from how the model generates tokens under grammar constraints.

### The BUC DFA

The `buc` (body-until-close) rule uses a standard "match everything except a specific substring" pattern:

```gbnf
buc ::= ([^<] | "<" [^/] | "</" [^p] | "</p" [^a] | ... | "</paramete" [^r] | "</parameter" [^>])*
```

Each alternative matches a prefix of the close tag followed by a character that breaks the pattern. When the full close tag sequence appears, none of the alternatives match, so the `*` terminates. The parent rule's `CLOSE` literal then matches the complete close tag.

This is a standard KMP-style substring exclusion automaton expressed as GBNF alternatives. It has O(L) alternatives where L is the close tag length.

### Properties

- **Grammar size**: 2 rules per body type — `buc` (O(L) alternatives) and the body rule itself. No state explosion.
- **Streaming compatible**: token mask computable at each step as the union of both paths.
- **Unbounded depth**: handles any number of embedded close tags via the `*` repetition.
- **No false commits**: the grammar never commits a close tag as structural until the model stops generating content.

## Continuation Rules

The grammar is structured as a **recursive chain** rather than a loop. Each body rule's `continuation` hands off directly to shared continuation rules that determine what valid next tags look like. Each body rule ends by naming its continuation rule directly — there is no outer loop.

At the top level, continuations match the full next tag (including `<`). Inside an invoke block, per-tool sequence rules offer `<parameter`, `<filter>`, or `</invoke>` as valid continuations. The only variation between body types is which continuation rule each chains to.

### Confirmation in the Grammar

The `continuation` rule is what confirms a close tag as structural. The grammar's `buc (close buc)* close continuation` pattern means a close tag is only structural when followed by a valid continuation — another tag open, a yield, or end of turn. Earlier close tags are absorbed by the `(close buc)*` repetition as content. Under constrained decoding, this is the primary confirmation mechanism.

### Parser Fallback: Tentative Close

When constrained decoding is not active (cloud providers, etc.), the parser implements equivalent logic via a **tentative close** mechanism. When a `Close` token arrives, the parser holds it tentatively and watches subsequent tokens:

- **Whitespace** — buffered, decision deferred
- **Valid structural continuation** (`<`, next open tag) — close confirmed, frame popped
- **Non-whitespace content** — close rejected, emitted as content
- **Another close tag for the same frame** — greedy last-match: prior close rejected as content, new close becomes tentative
- **EOF** — close confirmed

This mirrors the grammar's `(CLOSE buc)*` repetition at the semantic level. See [parser.md](parser.md) for details on the `pendingCloseStack` implementation.

## Phase Enforcement

The ordering constraint — reasons before messages and invocations — is enforced structurally through two sets of continuation rules:

- **Lens phase** continuations allow `<reason>`, `<message>`, `<invoke`, and yield as valid next elements. Reason bodies chain back to the lens phase, so multiple consecutive reasons are allowed.
- **Post-lens phase** continuations allow only `<message>`, `<invoke`, and yield. Once a message or invocation appears, all subsequent continuations use this phase. `<reason>` is simply not a valid option.

This means the ordering constraint is a structural guarantee of the grammar, not a runtime check. Under constrained decoding, the model physically cannot produce a reasoning block after a message.

## Invoke Nesting and Deep Confirmation

Inside an invoke block, the grammar generates **per-tool sequential parameter chains**. Each tool gets a sequence of rules (`tool-seq-N` down to `tool-seq-1`) that enforce constrained parameter names and bounded parameter count.

At each position in the chain, three continuations are valid: another `<parameter` (with a constrained name from the tool's schema), a `<filter>`, or `</invoke>`. Non-last parameter bodies chain to the next position in the sequence.

The **last parameter** (or filter) uses **deep confirmation**: its body rule confirms `</parameter>` through `</invoke>` all the way to the next top-level tag as a single unit:

```gbnf
last-body ::= buc (CLOSE buc)* CLOSE ws "</invoke>" turn-next-post
```

This means the grammar never commits `</parameter>` independently — it only accepts `</parameter>` when followed by `ws </invoke> ws <next-top-level-tag>`. This eliminates false commits at the invoke boundary and ensures the entire invoke block closes cleanly.

## Scaling

Grammar size is driven by the number of tools and their parameters. Each tool contributes:

- One `invoke` alternative with its tool name as a literal string constant in the `tool` attribute
- A sequential parameter chain (`seq-N` through `seq-1`) with constrained parameter names
- Per-position body rules for non-last and last parameters

Adding a tool with N parameters adds O(N) rules to the grammar. Grammar size scales linearly with total parameter count across all tools.

BUC DFAs are **shared** across all tools — there are only four: `param-buc`, `filter-buc`, `reason-buc`, and `msg-buc`. Each has O(L) alternatives where L is the close tag length.

Top-level tag types (`reason`, `message`) each contribute one body rule chaining to their respective continuation. Yield tags contribute one alternative each. The `maxLenses` protocol option (bounded lens counting) adds one body rule per lens slot, which is the only source of grammar growth independent of tool count.

## Whitespace Strategy

The grammar uses a single whitespace strategy: **unbounded whitespace** (`[ \t\n]*`) between block elements. This is safe because whitespace rules are always followed by a required specific token (a tag open or yield), so the grammar cannot get stuck producing whitespace indefinitely.

Whitespace after a close tag and before the next element is handled by the `ws` rule in the continuation. There are no bounded whitespace windows or confirmation-window states.

## Protocol Configuration

The grammar entry point varies based on protocol options:

- **`minLenses: 0`** — the chain starts in the lens phase; the model may immediately produce a message, invocation, or yield.
- **`minLenses: 1`** — the first element is forced to be a reasoning block. After it, the lens phase continues normally.
- **`requiredMessageTo`** — a forced-message phase is inserted between the lens phase and the post-lens phase. The model must produce a specific message before any free messages or invocations.
- **`maxLenses`** — generates multiple lens-phase variants, each allowing one fewer reasoning block, counting down to zero. This is the only protocol option that significantly increases grammar size independent of tool schema.

## Limitations

### GBNF Engine Performance on Long Bodies

Each embedded close tag in body content adds one iteration to the `*` repetition, which adds one level to the GBNF engine's internal state. For typical content this is negligible. For pathologically long content with many embedded close tags (hundreds), some GBNF engines exhibit quadratic backtracking behavior.

In practice, parameter bodies rarely contain more than a handful of embedded close tags, and the cost is negligible. This is a property of the GBNF engine implementation, not of the grammar design.
