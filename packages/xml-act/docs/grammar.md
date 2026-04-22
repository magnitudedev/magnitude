# Grammar Strategy

The grammar constrains LLM output during inference so that every generated turn conforms to the XML response format. It targets GBNF (llama.cpp and compatible providers). This document describes the design at a strategic level — the concepts and tradeoffs behind the grammar, not the implementation details of specific rules.

## Response Format

A turn consists of zero or more reasoning blocks, followed by zero or more messages or tool invocations, followed by exactly one yield token. Reasoning blocks must come before messages and invocations — once a message or invocation appears, no further reasoning blocks are allowed.

```
<reason about="...">...</reason>
<message to="...">...</message>
<invoke tool="...">
  <parameter name="...">value</parameter>
  <filter>$.query</filter>
</invoke>
<yield_user/>
```

Reasoning blocks carry an `about` attribute identifying the lens. Messages carry a `to` attribute. Invocations carry a `tool` attribute and contain parameter children (each with a `name` attribute and free-text value) and an optional filter child. Yield tokens are self-closing tags that terminate the turn.

## The Close-Tag Problem

LLMs occasionally mention their own tags inside body content — for example, a message explaining how to use the response format might contain a literal `</message>` as part of its prose. The grammar must distinguish real close tags from incidental mentions.

This is hard in a context-free grammar because there is no lookahead. Once the grammar starts matching a close tag sequence, it cannot peek at what comes after and change its mind.

## Bounded Lookahead

The solution is a confirmation window. When the body content matches what looks like a close tag, the grammar does not immediately commit. Instead, it enters a short bounded window where it examines what follows:

- If a **newline** appears, the close tag is confirmed. Newlines are the natural boundary between block-level elements.
- If a **`<`** appears (the start of a known next tag), the close tag is confirmed. The grammar consumes the `<` and hands off to continuation rules that match the rest of the tag without its leading `<`.
- If **any other character** appears — including letters, punctuation, or whitespace beyond the bound — the close tag is rejected. The entire sequence is treated as ordinary body content, and the grammar returns to its content-matching state.

The window allows a bounded number of horizontal whitespace characters (spaces and tabs) before requiring a confirmation signal. This bound is deliberate: an unbounded whitespace rule would create a trap where the grammar could consume spaces indefinitely without ever being able to reject and return to content.

**At every point in this window, the grammar can escape back to body content.** The model is never constrained during the lookahead — it can always produce any character. If the confirmation fails, the grammar backs off gracefully.

If the model produces more whitespace than the bound allows, the grammar escapes back to content. The close tag, the excess whitespace, and whatever follows all become part of the body. A properly confirmed close tag would need to appear later. Under constrained decoding this is unlikely since the grammar guides the model to stay within the bound.

## Chain Architecture

A naive grammar structure would use a loop: match reasoning blocks, then loop over messages and invocations, then match a yield. This fails because of how close-tag confirmation interacts with what comes next.

When a close tag is confirmed by seeing `<`, the grammar has already consumed that `<`. An outer loop would then try to match the next element starting with `<`, but that character is gone.

The solution is to structure the grammar as a **recursive chain** rather than a loop. Each body rule's confirmation window, upon seeing `<`, hands off directly to shared **continuation rules** that know what valid next tags look like — matching the tag name without the leading `<` (since it was already consumed). When confirmation is via newline, the continuation rules expect the full tag including `<`.

This means each body rule terminates by flowing into the grammar for the next element, rather than returning to an outer loop. The turn is a chain: reason body → continuation → message body → continuation → invoke body → continuation → yield.

## Phase Enforcement

The ordering constraint — reasons before messages and invocations — is enforced structurally through two sets of continuation rules:

- **Lens phase** continuations allow `<reason>`, `<message>`, `<invoke>`, and yield as valid next elements. Reason bodies chain back to the lens phase, so multiple consecutive reasons are allowed.
- **Post-lens phase** continuations allow only `<message>`, `<invoke>`, and yield. Once a message or invocation appears, all subsequent continuations use this phase. `<reason>` is simply not a valid option.

This means the ordering constraint is a structural guarantee of the grammar, not a runtime check. Under constrained decoding, the model physically cannot produce a reasoning block after a message.

## Invoke Nesting

Inside an invoke block, the valid continuations are different from the top level. After a `</parameter>`, the next element can be another `<parameter>`, a `<filter>`, or `</invoke>`. After `</filter>`, only `</invoke>` is valid. The same chain mechanism applies but with a smaller, context-specific set of continuation rules.

## Scaling

Grammar size is primarily driven by the number of top-level tag types. Each tag type contributes one body DFA whose size is proportional to the tag name length. Adding a new top-level tag type costs roughly `L + 8` rules where L is the tag name length.

Tool names are **not enumerated** in the grammar — they appear as free-form quoted string values in the `tool` attribute. This means adding or removing tools has zero impact on grammar size. Tool-specific parameter validation is handled by the runtime, not the grammar.

Yield tags contribute one alternative each. The `maxLenses` protocol option (bounded lens counting) adds one body DFA per lens slot, which is the only source of significant grammar growth.

## Whitespace Strategy

The grammar uses three whitespace strategies depending on context:

- **Unbounded whitespace** (`[ \t\n]*`) between block elements where the model naturally produces newlines. This is safe because it is always followed by a required specific token (a tag open or yield), so the grammar cannot get stuck.
- **Bounded horizontal whitespace** (a fixed number of optional `[ \t]?` slots) for inline spacing like indentation. Bounded to prevent the model from getting stuck producing spaces.
- **Trailing-whitespace confirmation** (the bounded lookahead window described above) for close-tag confirmation. Bounded for the same reason, with escape-to-content at every state.

The common theme: unbounded whitespace is only safe when followed by a required token. In any context where the grammar might need to reject and try a different path, whitespace must be bounded.

## Protocol Configuration

The grammar entry point varies based on protocol options:

- **`minLenses: 0`** — the chain starts in the lens phase; the model may immediately produce a message, invocation, or yield.
- **`minLenses: 1`** — the first element is forced to be a reasoning block. After it, the lens phase continues normally.
- **`requiredMessageTo`** — a forced-message phase is inserted between the lens phase and the post-lens phase. The model must produce a specific message before any free messages or invocations.
- **`maxLenses`** — generates multiple lens-phase variants, each allowing one fewer reasoning block, counting down to zero. This is the only protocol option that significantly increases grammar size.

## Limitations

Two inherent tradeoffs of the bounded lookahead approach:

**False confirmation.** If the model writes a close tag in prose and a newline or `<` happens to follow within the confirmation window, the grammar treats it as a real close. The `<` path confirms on the bare `<` character alone — the grammar then hands off to continuation rules that constrain what tag name follows. This means any `<` after a close tag confirms it, not just `<` followed by a known tag. Under constrained decoding the grammar steers the model away from false closes, but without constrained decoding it can happen.

**Bounded window cutoff.** If whitespace after a close tag exceeds the bound, the grammar gives up and treats the close tag as content. Any valid opening tag that followed after the excess whitespace is also swallowed as content. The body continues until a properly confirmed close tag appears later. The bound is a tradeoff between lookahead reach and grammar complexity — a larger bound catches more cases but adds states.
