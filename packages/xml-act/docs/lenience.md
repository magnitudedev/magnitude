# Close-Tag Confirmation

The tokenizer and grammar use a bounded lookahead mechanism to distinguish real close tags from incidental mentions in body content.

## The Problem

LLMs sometimes mention their own tags inside prose — for example, explaining how `</message>` works. The parser must not treat these as structural close tags.

## Mechanism

When the tokenizer encounters `</tagname>`, it does not immediately emit a Close token. Instead, it enters a **pending close** state and examines subsequent characters:

- **`\n`** (newline) → **confirm**. The close tag is real. Emit Close token.
- **`<`** (start of next tag) → **confirm**. The close tag is real. Emit Close token, then process `<` as the start of the next tag.
- **Horizontal whitespace** (space, tab) → **buffer**, up to a bounded maximum (4 characters). Continue waiting for a confirming signal.
- **Any other character** → **reject**. The close tag was incidental. The entire sequence (`</tagname>` + buffered whitespace) is emitted as content text.
- **Excess whitespace** (more than 4 characters) → **reject**. Same as above.

## Key Properties

1. **The model is never trapped.** At every point in the lookahead window, any character is valid — it either confirms, continues buffering, or rejects back to content.

2. **Grammar and tokenizer are in lockstep.** The grammar's trailing-whitespace (tw) states implement the same logic: bounded horizontal whitespace slots, confirm on `\n` or `<`, escape to content body on anything else.

3. **Applies to all close tags equally.** `</reason>`, `</message>`, `</invoke>`, `</parameter>`, `</filter>` — all go through the same confirmation mechanism. There is no special handling per tag.

## Practical Implications

- `</reason>` inside backticks (`` `</reason>` ``) → the backtick after `>` rejects → content. ✅
- `</reason>` followed by a newline → confirmed close. ✅
- `</reason>` followed by `<message` → confirmed close, then `<message` starts. ✅
- Bare `</reason>` in prose followed by a letter → rejected → content. ✅
- `</reason>` followed by 5+ spaces → rejected (exceeds bound) → content.

## Bound Tradeoff

The 4-character bound is a tradeoff. A larger bound catches more edge cases but adds grammar states. A smaller bound is simpler but rejects legitimate close tags with trailing indentation. Under constrained decoding, the grammar guides the model to stay within the bound.
