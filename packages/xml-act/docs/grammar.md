# Grammar Strategy

The grammar constrains LLM output during inference so that every generated turn conforms to the XML response format. It targets GBNF (llama.cpp and compatible providers). This document describes the design at a strategic level — the concepts and tradeoffs behind the grammar, not the implementation details of specific rules.

## Response Format

A turn consists of zero or more reasoning blocks, followed by zero or more messages or tool invocations, followed by exactly one yield token. Reasoning blocks must come before messages and invocations — once a message or invocation appears, no further reasoning blocks are allowed.

```xml
<reason about="...">...</reason>
<message to="...">...</message>
<invoke tool="...">
  <parameter name="...">value</parameter>
  <filter>$.query</filter>
</invoke>
<yield_user/>
```

Reasoning blocks carry an `about` attribute identifying the lens. Messages carry a `to` attribute. Invocations carry a `tool` attribute and contain `<parameter` children (each with a `name` attribute and free-text value) and an optional `<filter>` child. Yield tokens are self-closing tags that terminate the turn.

## The Close-Tag Problem

LLMs occasionally mention their own tags inside body content — for example, a shell command that outputs XML, or documentation explaining the response format itself. The grammar must distinguish real close tags from incidental mentions.

This is hard because the close tag sequence is just bytes — there's nothing syntactically special about the "real" one versus a mention in content. The grammar must decide, at each close tag occurrence, whether it's structural or content.

## Two Strategies: Eager Confirmation and Greedy Last-Match

The grammar uses two different strategies depending on the tag type. The choice is driven by a fundamental tradeoff: **greedy last-match provides maximum content freedom but zero structural constraint inside the body, while eager confirmation maintains structural constraint throughout but risks false commits on close tags in content.**

The rule: **greedy for opaque content, eager for structured content.**

- **Parameter and filter bodies** contain opaque content — shell commands, code, file contents, arbitrary text. The model should be free to write anything, including its own close tags. Structural constraint only matters at the invoke level (what comes after the body closes). Greedy last-match is the right choice.
- **Reason and message bodies** contain structured prose. The grammar should guide the model throughout generation, not just at the boundaries. Eager confirmation provides that ongoing constraint.

The greedy `*` repetition creates a window where both the "content" and "structural close" paths are live. Inside this window, the union token mask is dominated by the content path, which accepts any character. This means the grammar provides **no constraint** while the body is being generated — the model is completely free. Constraint resumes only when the model exits the `*` at the final close tag, where the continuation rule takes over.

For parameter bodies this is exactly what you want. For reason/message bodies it would defeat the purpose of constrained decoding — the model could generate arbitrary bytes and the grammar would accept them, because the content path always accepts everything.

### Eager Confirmation (reason, message)

For top-level body tags (`reason`, `message`), the grammar uses **bounded lookahead confirmation**. When the body content matches a close tag, the grammar enters a short window examining what follows:

- If a **newline** appears, the close tag is confirmed.
- If a **`<`** appears (start of a known next tag), the close tag is confirmed.
- If **any other character** appears, the close tag is rejected and treated as body content.

The window allows a bounded number of horizontal whitespace characters (spaces and tabs) before requiring a confirmation signal. This bound prevents the model from getting stuck producing spaces indefinitely.

**At every point in this window, the grammar can escape back to body content.** The model is never constrained during the lookahead.

This strategy works well for `reason` and `message` bodies because their close tags are rarely mentioned in content, and the confirmation signals (newline, `<`) are reliable indicators of structural boundaries.

### Greedy Last-Match (parameter, filter)

For `parameter` and `filter` bodies inside `invoke` blocks, the grammar uses a fundamentally different strategy: **greedy last-match via recursive repetition**.

Parameter bodies are the most likely to contain their own close tags — shell commands, code snippets, XML output, and documentation frequently include the literal close tag sequence. Eager confirmation would false-commit in many of these cases.

The greedy approach ensures the **last** close tag is structural and all earlier ones are content. It achieves this without any lookahead, confirmation windows, or state explosion.

#### How It Works

The grammar defines a parameter body as:

```gbnf
param-body ::= buc (CLOSE buc)* CLOSE continuation
```

Where:
- **`buc`** (body-until-close) is a DFA matching any string that does NOT contain the close tag sequence
- **`CLOSE`** is the literal close tag (e.g. `</parameter>`)
- **`continuation`** is the rule for what comes next at the invoke level (another `<parameter`, a `<filter>`, or `</invoke>`)

The `*` repetition is the key. At each close tag, the GBNF engine faces two valid alternatives:

1. **Take another iteration** of the `*`: this close tag is content. Match more body, then another close tag.
2. **Exit the `*`** and match the final `CLOSE continuation`: this close tag is structural.

The engine computes the **union** of valid next tokens for both paths. Both alternatives remain live. The **model** picks a token — and its choice implicitly selects the interpretation.

#### Why Greedy Last-Match Is Emergent

The model generates content as long as it has content to express. Each close tag encountered along the way gets absorbed by the `*` (content path). When the model is done, it stops generating body characters. The engine exits the `*` and the final close tag chains to the continuation.

The **last** close tag is structural because it's the one where the model chose to stop. All earlier ones were content because the model kept going. No explicit "greedy" logic is needed — it falls out naturally from how the model generates tokens under grammar constraints.

#### The BUC DFA

The `buc` (body-until-close) rule uses a standard "match everything except a specific substring" pattern:

```gbnf
buc ::= ([^<] | "<" [^/] | "</" [^p] | "</p" [^a] | ... | "</paramete" [^r] | "</parameter" [^>])*
```

Each alternative matches a prefix of the close tag followed by a character that breaks the pattern. When the full close tag sequence appears, none of the alternatives match, so the `*` terminates. The parent rule's `CLOSE` literal then matches the complete close tag.

This is a standard KMP-style substring exclusion automaton expressed as GBNF alternatives. It has O(L) alternatives where L is the close tag length.

#### Properties

- **Grammar size**: 2 rules per body type — `buc` (O(L) alternatives) and the body rule itself. No state explosion.
- **Streaming compatible**: token mask computable at each step as the union of both paths.
- **Unbounded depth**: handles any number of embedded close tags via the `*` repetition.
- **No false commits**: the grammar never commits a close tag as structural until the model stops generating content.

## Chain Architecture

A naive grammar structure would use a loop: match reasoning blocks, then loop over messages and invocations, then match a yield. This fails because of how close-tag confirmation interacts with what comes next.

When a close tag is confirmed by seeing `<` (in eager confirmation), the grammar has already consumed that `<`. An outer loop would then try to match the next element starting with `<`, but that character is gone.

The solution is to structure the grammar as a **recursive chain** rather than a loop. Each body rule's confirmation window, upon seeing `<`, hands off directly to shared **continuation rules** that know what valid next tags look like — matching the tag name without the leading `<` (since it was already consumed). When confirmation is via newline, the continuation rules expect the full tag including `<`.

This chain architecture applies to `reason` and `message` bodies (which use eager confirmation). Parameter and filter bodies use the greedy approach and chain to `invoke-next` after the final structural close tag.

## Phase Enforcement

The ordering constraint — reasons before messages and invocations — is enforced structurally through two sets of continuation rules:

- **Lens phase** continuations allow `<reason>`, `<message>`, `<invoke`, and yield as valid next elements. Reason bodies chain back to the lens phase, so multiple consecutive reasons are allowed.
- **Post-lens phase** continuations allow only `<message>`, `<invoke`, and yield. Once a message or invocation appears, all subsequent continuations use this phase. `<reason>` is simply not a valid option.

This means the ordering constraint is a structural guarantee of the grammar, not a runtime check. Under constrained decoding, the model physically cannot produce a reasoning block after a message.

## Invoke Nesting

Inside an invoke block, the valid continuations are different from the top level. After a parameter or filter close tag, the next element can be another `<parameter`, a `<filter>`, or `</invoke>`. The greedy body rules chain directly to `invoke-next` which handles these continuations.

## Scaling

Grammar size is primarily driven by the number of top-level tag types. Each tag type contributes either:
- One body DFA (eager confirmation): size proportional to tag name length, ~L+8 rules
- Two rules (greedy last-match): the BUC pattern + the body rule

Tool names are **not enumerated** in the grammar — they appear as free-form quoted string values in the `tool` attribute. Adding or removing tools has zero impact on grammar size. Tool-specific parameter validation is handled by the runtime, not the grammar.

Yield tags contribute one alternative each. The `maxLenses` protocol option (bounded lens counting) adds one body DFA per lens slot, which is the only source of significant grammar growth.

## Whitespace Strategy

The grammar uses three whitespace strategies depending on context:

- **Unbounded whitespace** (`[ 	
]*`) between block elements where the model naturally produces newlines. This is safe because it is always followed by a required specific token (a tag open or yield), so the grammar cannot get stuck.
- **Bounded horizontal whitespace** (a fixed number of optional `[ 	]?` slots) for the eager confirmation window. Bounded to prevent the model from getting stuck producing spaces.
- **Implicit whitespace** in the greedy body rules — whitespace after a close tag is naturally handled by the `buc` pattern in the next `*` iteration, or by the `ws` rule in the continuation.

## Protocol Configuration

The grammar entry point varies based on protocol options:

- **`minLenses: 0`** — the chain starts in the lens phase; the model may immediately produce a message, invocation, or yield.
- **`minLenses: 1`** — the first element is forced to be a reasoning block. After it, the lens phase continues normally.
- **`requiredMessageTo`** — a forced-message phase is inserted between the lens phase and the post-lens phase. The model must produce a specific message before any free messages or invocations.
- **`maxLenses`** — generates multiple lens-phase variants, each allowing one fewer reasoning block, counting down to zero. This is the only protocol option that significantly increases grammar size.

## Limitations

### Eager Confirmation (reason, message)

**False confirmation.** If the model writes a close tag in prose and a newline or `<` happens to follow within the confirmation window, the grammar treats it as a real close. Under constrained decoding the grammar steers the model away from false closes, but without constrained decoding it can happen.

**Bounded window cutoff.** If whitespace after a close tag exceeds the bound, the grammar gives up and treats the close tag as content. The bound is a tradeoff between lookahead reach and grammar complexity.

### Greedy Last-Match (parameter, filter)

**No limitations on content.** The greedy approach has no false-commit scenarios — any content is valid, including content that contains close tags followed by structural-looking continuations. The last close tag is always structural.

The only theoretical limitation is performance: each embedded close tag adds one iteration to the `*` repetition, which adds one level to the GBNF engine's internal state. In practice, content rarely contains more than a few embedded close tags, and the cost is negligible.
