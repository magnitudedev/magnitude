# Close-Tag Confirmation

The grammar and parser work together to distinguish real close tags from incidental mentions in body content.

## The Problem

LLMs sometimes mention their own tags inside prose — for example, explaining how `</message>` works, or writing documentation that references `</parameter>`. The parser must not treat these as structural close tags.

## Mechanism

Close-tag confirmation is handled in two cooperating layers.

### Layer 1: Grammar (constrained decoding)

All body types — think, message, invoke, parameter, filter — use a **greedy last-match** rule:

```
body = buc (close buc)* close continuation
```

Where `buc` is "body until close" (any content not starting the close tag), `close` is the structural close tag, and `continuation` is the next valid token after this frame closes (e.g. another tag, or end of turn).

This means the grammar allows the model to write `</tagname>` as content any number of times. Only the *last* occurrence — the one followed by a valid continuation — is structural. Under constrained decoding, the model is guided to produce exactly this pattern naturally.

The model is never trapped: at every point, the grammar allows content continuation. The structural close only "locks in" when the continuation is unambiguous.

For the last parameter in an invoke block, the grammar uses **deep confirmation**: `</parameter>` is confirmed through `</invoke>` all the way to the next top-level tag as a single grammar unit. This eliminates false commits at invoke boundaries.

### Layer 2: Parser (tentative close)

When constrained decoding is not active (cloud providers, etc.), the parser implements the same logic via a **`pendingCloseStack`** tentative close mechanism.

When a `Close` token arrives matching the current frame, the parser does not apply it immediately. Instead it enters a tentative state, buffering the close and waiting for the next token:

| Next token | Action |
|---|---|
| Whitespace Content | Buffer and stay tentative |
| Non-whitespace Content | **Reject** — close tag was incidental; emit as content |
| Valid structural Open / SelfClose | **Confirm** — pop the frame |
| Close matching current frame | **Replace** — greedy last-match; discard earlier tentative close |
| Close matching parent frame | **Cascade** — push to stack alongside current entry; both confirm or reject together |
| EOF | **Confirm** |

"Valid structural" means the Open tag is a known tag type that can legally follow the current frame's close. For example, after `</parameter>` inside an invoke, another `<parameter>` or `<filter>` or `</invoke>` is valid; a random word is not.

## Key Properties

1. **The model is never trapped.** The grammar always allows content continuation. The parser always allows rejection back to content if the next token is not a valid continuation.

2. **Grammar and parser work in concert.** Under constrained decoding, the grammar guides the model to produce the correct pattern. Without it, the parser's tentative close handles confirmation independently.

3. **Cascade support.** Stacked close tags like `</parameter></invoke>` are handled together. The `</parameter>` goes tentative; when `</invoke>` arrives, it is pushed onto the stack alongside the parameter entry. Both entries confirm or reject as a unit when the final continuation arrives.

4. **Schema-aware.** Param name validation uses the tool schema. `<parameter name="unknown">` after `</parameter>` is not a valid continuation and will reject the tentative close back to content.

## Practical Implications

- `</think>` inside backticks (`` `</think>` ``) → the backtick rejects → content. ✅
- `</think>` at end of turn, followed by `<message` → confirmed, then `<message` starts. ✅
- `</parameter>` followed by `</invoke>` followed by `<yield_user/>` → cascade confirms both. ✅
- `</parameter>` followed by `<parameter name="next">` (valid next param) → confirms, opens next param. ✅
- `</parameter>` followed by ` and more prose` → non-whitespace Content → rejected → content. ✅
- `</parameter>` followed by another `</parameter>` → greedy last-match; earlier one absorbed as content. ✅

## Without Constrained Decoding

The parser's tentative close mechanism is fully self-contained — it does not require the grammar to be active. The same confirmation logic applies: hold the close tentatively, confirm on a valid structural continuation, reject on anything else. This makes the parser robust across all providers.
