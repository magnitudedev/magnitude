# Tag Lenience Protocols

This document describes how the xml-act parser and grammar handle non-canonical tag forms that models may produce.

## Canonical Tag Format

| Tag type | Canonical form | Example |
|---|---|---|
| Open | `<\|name>` or `<\|name:variant>` | `<\|think:strategy>`, `<\|invoke:shell>` |
| Close | `<name\|>` | `<think\|>`, `<invoke\|>` |
| Self-close | `<\|name:variant\|>` | `<\|yield:user\|>` |
| Parameter open | `<\|parameter:name>` | `<\|parameter:cmd>` |
| Parameter close | `<parameter\|>` | `<parameter\|>` |

## Closing Tag Lenience

Models sometimes produce non-canonical close tags. All accepted forms are normalized to canonical `<name|>` in token output and history.

### Failure modes

| Mode | Form | Example | Description |
|---|---|---|---|
| Canonical | `<name\|>` | `<think\|>` | Correct form |
| Mode 1 | `</name\|>` | `</think\|>` | Slash prefix added, pipe retained |
| Mode 2 | `</name>` | `</think>` | Slash prefix added, pipe omitted |
| Mode 3 | `<name>` | `<think>` | No slash, no pipe |

### Where each mode is handled

| Mode | Tokenizer (parser) | Grammar |
|---|---|---|
| Canonical | ✅ | ✅ |
| Mode 1 | ✅ Lenient — `/` after `<` starts close tag, `/` skipped | ✅ DFA accepts `</name\|>` |
| Mode 2 | ✅ Lenient — `/` skipped, `>` without pipe accepted | ✅ DFA accepts `</name>` |
| Mode 3 | ✅ Lenient — `>` without pipe accepted | ✅ DFA accepts `<name>` |

### Tokenizer implementation

The tokenizer handles `/` after `<` in two code paths:

1. **Inline** — When `<` and `/` appear in the same chunk, the `/` is detected in lookahead, `startCloseTag()` is called, and `/` is skipped. The raw buffer includes `</` so `failAsContent()` dumps correctly if the name is invalid.
2. **Chunk boundary** — When `<` is the last char of a chunk (`pendingLt` state), `/` as the first char of the next chunk triggers the same close-tag path.

After the `/` is skipped, normal `close_name` phase processing handles the tag name. If the name is invalid, `failAsContent()` emits `</` as literal content.

### Grammar implementation

The DFA in `generateBodyRules` tracks the close sequence with branching:

```
s0 (base) → on '<' → s1
s1        → on '/' → slash (then expects tagname)
           → on first tagname char → s2 (shared)
slash     → on first tagname char → s2 (shared)
s2..sN    → track remaining tagname chars
sN+1      → on '|' → pipe state → on '>' → terminal
           → on '>' → terminal (Mode 2/3)
```

All four close variants converge to the same terminal state.

## Opening Tag Lenience

### Invoke without keyword

Models may produce `<|toolname>` instead of `<|invoke:toolname>`.

| Form | Example | Description |
|---|---|---|
| Canonical | `<\|invoke:shell>` | Correct form with invoke keyword |
| Lenient | `<\|shell>` | Tool name without invoke keyword |

**Tokenizer**: When `variant` is undefined and `name` is in `knownToolTags`, the token is rewritten to `Open { name: 'invoke', variant: name }`. Controlled by the `toolKeyword` option (default `'invoke'`).

**Grammar**: No lenience needed — the grammar constrains output to the canonical form, preventing this failure mode when grammar-guided generation is active.

### Rationale for parser-only lenience

When grammar-guided generation is available, the grammar prevents malformed opening tags entirely. The tokenizer lenience exists as a fallback for models running without grammar constraints.

## Newline Enforcement

Top-level tags (`think`, `message`, `invoke`, `yield`) require newlines before and after them.

### Grammar

Newlines are embedded directly in tag literals:

- Open tags: `"\n<|think:name>\n"`, `"\n<|message:recipient>\n"`, `"\n<|invoke:tool>\n"`
- Close tags: `"\n<think|>\n"`, `"\n<message|>\n"`, `"\n<invoke|>\n"`
- Self-close: `"\n<|yield:target|>\n"`
- Parameter close: `"<parameter|>\n"` (no preceding newline required)

This forces the model to output newlines around top-level tags.

### Tokenizer

The `strictNewlines` option (default `false`) enables enforcement in the tokenizer:

- Checks `savedAfterNewline` before emitting open, close, or self-close tokens for top-level tags
- If the tag is not preceded by a newline, `failAsContent()` is called instead of emitting the token
- Parameter tags are exempt — no newline requirement

The flag defaults to `false` to allow incremental adoption.

## Normalization

All lenient forms are normalized to canonical form in token output. This means:

1. Downstream consumers (parser, history) always see canonical tokens
2. The grammar always accepts canonical close tags as the "true" form
3. Lenient acceptance never leaks non-canonical forms into the token stream

## Summary Matrix

| Concern | Tokenizer | Grammar |
|---|---|---|
| Close tag Mode 1 (`</name\|>`) | ✅ Lenient | ✅ Lenient |
| Close tag Mode 2 (`</name>`) | ✅ Lenient | ✅ Lenient |
| Close tag Mode 3 (`<name>`) | ✅ Lenient | ✅ Lenient |
| Open tag without invoke keyword | ✅ Lenient | ❌ Not needed (grammar prevents) |
| Newline before top-level tags | ✅ Optional (`strictNewlines`) | ✅ Enforced |
| Newline after top-level tags | — | ✅ Enforced |
| Newline after parameter close | — | ✅ Enforced |
