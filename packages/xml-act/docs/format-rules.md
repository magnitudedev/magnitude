# Format Rules

Comprehensive reference for the xml-act response format — what is accepted, what is rejected, and why.

---

## 1. Core Principles

### 1.1 The `magnitude:` Prefix Rule

All tags prefixed with `magnitude:` are **always interpreted as structural**. They are never treated as literal content. If a `magnitude:`-prefixed tag appears in a context where it is not structurally valid, it is a parse error — not content.

This rule applies to both opening and closing tags.

### 1.2 Grammar–Parser Agreement

The grammar (GBNF constrained decoding) and the parser (streaming token interpreter) enforce the same rules. They should never diverge on whether a given input is valid or invalid.

The grammar constrains LLM generation to prevent invalid output. The parser interprets the output and emits structured events. Both layers reject the same invalid constructs.

**Exception:** In a small number of edge cases (e.g., tool alias opens with extra attributes), the grammar may reject an input that the parser can still recover from. These cases are documented where they occur.

### 1.3 Fail Fast, Preserve Content

When the parser encounters invalid `magnitude:` markup, it emits a structural error immediately. In body contexts (message, think, parameter, filter), the raw text of the invalid tag is preserved as content so downstream consumers can still see what the model intended. In invoke context, invalid subtrees are silently discarded to avoid corrupting tool call structure.

---

## 2. Turn Structure

A turn follows a strict phase ordering:

1. **Optional leading whitespace**
2. **Zero or more think blocks** (thinking/reasoning)
3. **Zero or more message and invoke blocks** (interleaved in any order)
4. **Exactly one yield** (turn termination)

Once a message or invoke appears, no further think blocks are allowed. This is enforced by the grammar.

### 2.1 Think

```
<magnitude:think name="...">body</magnitude:think>
```

- The `name` attribute labels the reasoning lens.
- Body may contain arbitrary text.
- Body may NOT contain other `magnitude:` opens.
- Leading and trailing whitespace is stripped.

### 2.2 Message

```
<magnitude:message to="recipient">body</magnitude:message>
```

- The `to` attribute specifies the message recipient.
- Body may contain arbitrary text.
- Body may NOT contain other `magnitude:` opens.
- Leading whitespace is stripped while body is empty.
- Trailing newline runs are deferred until followed by more content.

### 2.3 Invoke (Tool Call)

```
<magnitude:invoke tool="toolName">
  <magnitude:parameter name="paramName">value</magnitude:parameter>
  <magnitude:filter>query</magnitude:filter>
</magnitude:invoke>
```

- The `tool` attribute is required. Missing it emits `MissingToolName`.
- If the tool is not registered, emits `UnknownTool` and the invoke is dead.
- Direct content between parameters (non-whitespace) emits `UnexpectedContent`.

### 2.4 Yield

```
<magnitude:yield_user/>
<magnitude:yield_invoke/>
<magnitude:yield_worker/>
<magnitude:yield_parent/>
```

- Yields are self-closing tags that terminate the turn.
- They are only valid at the prose (top) level.
- Available yield variants are configured per role.
- After yield, the parser enters observing mode. Any non-whitespace content after yield results in `runaway` termination classification.

---

## 3. Tag Forms

### 3.1 Canonical Tags

The canonical structural vocabulary:

| Tag | Context | Purpose |
|---|---|---|
| `magnitude:think` | Prose | Thinking/reasoning block |
| `magnitude:message` | Prose | Message to a recipient |
| `magnitude:invoke` | Prose | Tool invocation |
| `magnitude:parameter` | Invoke | Tool parameter |
| `magnitude:filter` | Invoke | Output filter query |
| `magnitude:yield_*` | Prose | Turn termination |

### 3.2 Tool Aliases

If a tool named `shell` is registered, the following are equivalent:

```
<magnitude:invoke tool="shell">...</magnitude:invoke>
<magnitude:shell>...</magnitude:shell>
```

Tool aliases:
- Are only valid at the prose level (same as `magnitude:invoke`).
- Close with either the alias close (`</magnitude:shell>`) or the canonical close (`</magnitude:invoke>`).
- Are schema-driven: only registered tool names become valid aliases.
- Self-closing tool aliases are **not** valid.
- Tool aliases with extra attributes are rejected by the grammar (but the parser may recover).

### 3.3 Parameter Aliases

If a tool `shell` has a parameter `command`, the following are equivalent inside that tool's invoke:

```
<magnitude:parameter name="command">value</magnitude:parameter>
<magnitude:command>value</magnitude:command>
```

Parameter aliases:
- Are only valid inside the invoke of the tool that defines them.
- Close with either the alias close (`</magnitude:command>`) or the canonical close (`</magnitude:parameter>`).
- Are schema-driven: only parameter names from the current tool's schema become valid aliases.
- A parameter alias valid in one tool (e.g., `<magnitude:path>` in `edit`) is invalid in another (e.g., `shell`).
- Self-closing parameter aliases are **not** valid.

---

## 4. Body Content Rules

### 4.1 What Bodies May Contain

| Body Type | Plain Text | `magnitude:` Opens | `magnitude:` Closes |
|---|---|---|---|
| Reason | ✅ | ❌ Error | ❌ Error (same-line) |
| Message | ✅ | ❌ Error | ❌ Error (same-line) |
| Parameter | ✅ | ❌ Error | ❌ Error (same-line) |
| Filter | ✅ | ❌ Error | ❌ Error (same-line) |
| Invoke | ❌ (only whitespace) | Only parameter/filter | N/A |

In all body contexts, body scanning stops at any `</magnitude:*>` prefix, not just the canonical close for the current frame. This is what allows alias close tags (such as `</magnitude:shell>` or `</magnitude:command>`) to participate in normal close handling.

### 4.2 Invalid `magnitude:` Opens in Bodies

Any `magnitude:`-prefixed open tag inside a body (think, message, parameter, filter) is an `InvalidMagnitudeOpen` error. The raw tag text is preserved as content in the body.

Examples of invalid opens:
- `<magnitude:invoke>` inside a message body
- `<magnitude:shell>` inside a parameter body
- `<magnitude:parameter>` inside a filter body
- `<magnitude:foo>` (unknown) inside any body

### 4.3 Invalid Opens in Invoke

Inside an invoke frame, only `<magnitude:parameter>` (or valid parameter aliases) and `<magnitude:filter>` are valid children. Everything else is `InvalidMagnitudeOpen`, including:
- `<magnitude:message>`
- `<magnitude:think>`
- `<magnitude:invoke>` (nested)
- Unknown `magnitude:` tags
- Tool aliases or parameter aliases from other tools

When an invalid open occurs inside invoke, the parser enters **invalid subtree mode**: all tokens until the matching close are silently discarded. This prevents invalid nested structure from corrupting the tool call.

### 4.4 Invalid Opens at Prose Level

Unknown `magnitude:`-prefixed opens at the prose level (e.g., `<magnitude:foo>`) are `InvalidMagnitudeOpen`. If the tag is not self-closing, the parser enters invalid subtree mode and discards content until the matching close.

---

## 5. Close Tag Semantics

Matching close tags use **first-close-wins** semantics.

The first close tag that matches the current frame closes it immediately. There is no greedy last-match behavior and no pending close confirmation for matching closes.

Examples:
- Inside `<magnitude:message>`, the first `</magnitude:message>` closes the message.
- Inside `<magnitude:parameter name="command">`, the first valid parameter close — canonical `</magnitude:parameter>` or alias `</magnitude:command>` — closes the parameter.
- Inside `<magnitude:invoke tool="shell">`, the first valid invoke close — canonical `</magnitude:invoke>` or alias `</magnitude:shell>` — closes the invoke.

This means body content that naturally contains close-tag-like sequences (for example, documentation or code samples showing the format itself) will be truncated at the first matching close. There is no escape mechanism to preserve such sequences as literal content.

---

## 6. Close Mismatch Recovery

When a `magnitude:`-prefixed close tag does not match the current frame, the parser applies context-dependent recovery.

### 6.1 Recoverable Mismatch (Silent Close)

A mismatched close may be treated as if it were the correct close for the current frame. When that happens, the current frame closes silently and no structural error is emitted.

A mismatch is confirmed in any of these cases:

- **Newline boundary:** whitespace containing a newline follows the mismatched close, and then a structural token arrives.
- **Cascade close:** the next close tag matches the parent frame.
- **Valid structural continuation:** the next open tag is a valid structural continuation for the parent frame.
- **Magnitude open at prose level:** the next open tag is a `magnitude:` tag and the parent frame is prose.

Examples:

```
<magnitude:message>hello
</magnitude:think>
<magnitude:yield_user/>
```

The mismatched `</magnitude:think>` is treated as closing the message.

```
<magnitude:parameter name="command">echo hi
</magnitude:filter>
</magnitude:invoke>
```

The mismatched `</magnitude:filter>` is treated as closing the parameter, and the following invoke close confirms the cascade.

### 6.2 Same-Line Mismatch (Ambiguous — Error)

If a mismatched close appears on the same line as preceding content:

```
<magnitude:message>hello</magnitude:think> world</magnitude:message>
```

The `</magnitude:think>` is on the same line as `hello`. This is ambiguous — it could be a typo or intentional content. The grammar rejects this input. The parser emits `AmbiguousMagnitudeClose` and preserves the raw close as content.

A mismatch is also rejected when non-whitespace content follows without a newline, or when a following close does not form a valid cascade. In those cases the raw close is dumped as content and `AmbiguousMagnitudeClose` is emitted.

### 6.3 Stray Close at Prose Level

A `magnitude:`-prefixed close at the prose level with no matching open:

```
</magnitude:foo><magnitude:yield_user/>
```

This emits `StrayCloseTag`. The raw close is preserved as prose content.

---

## 7. Parameter Validation

### 7.1 Unknown Parameters

A `<magnitude:parameter name="foo">` where `foo` is not in the tool's schema emits `UnknownParameter`. The parameter frame is dead — its content is parsed but not included in the tool input.

### 7.2 Duplicate Parameters

Opening a parameter with the same name as one already seen in the current invoke emits `DuplicateParameter`. The duplicate frame is dead — the first value is preserved.

### 7.3 Missing Required Fields

When an invoke closes, any required parameters not provided emit `MissingRequiredField`. The tool call does not produce `ToolInputReady`.

### 7.4 Schema Coercion Errors

If a parameter value cannot be coerced to the expected schema type, `SchemaCoercionError` is emitted. The tool call does not produce `ToolInputReady`.

---

## 8. Whitespace Handling

Whitespace handling varies by context:

### 8.1 Prose
- Leading whitespace before first content is stripped.
- Trailing whitespace at end is stripped.
- Internal whitespace is preserved.

### 8.2 Reason
- Leading whitespace is stripped.
- Trailing whitespace is stripped.
- Internal whitespace is preserved.

### 8.3 Message
- Leading whitespace is stripped while body is empty.
- Trailing newline runs are deferred — they are only emitted if followed by more content.
- Internal whitespace is preserved.

### 8.4 Parameter
- All content is preserved as-is (no stripping).
- Content is emitted as `ToolInputFieldChunk` events.

### 8.5 Filter
- All content is preserved as-is.
- No chunk events are emitted (filter content is accumulated internally).

### 8.6 Between Structural Tags
- Whitespace between structural tags (e.g., between `</magnitude:parameter>` and `<magnitude:parameter>`) is allowed and ignored.
- Non-whitespace content between parameters inside invoke emits `UnexpectedContent`.

---

## 9. EOF Behavior

If the stream ends without a yield or with unclosed frames, the parser applies salvage:

| Unclosed Frame | Behavior |
|---|---|
| Parameter | Finalized as if closed (value preserved) |
| Filter | Silently popped |
| Invoke | `IncompleteTool` error emitted |
| Message | `MessageEnd` emitted, frame closed |
| Reason | `UnclosedThink` error emitted, `LensEnd` emitted |
| Prose | `ProseEnd` emitted if any content |

Pending mismatch closes are confirmed at EOF.

---

## 10. Error Reference

### 10.1 Structural Errors

| Error | When |
|---|---|
| `InvalidMagnitudeOpen` | `magnitude:` open tag in invalid context |
| `AmbiguousMagnitudeClose` | Same-line mismatched `magnitude:` close in body |
| `StrayCloseTag` | `magnitude:` close at prose level with no matching open |
| `UnexpectedContent` | Non-whitespace content directly inside invoke frame |
| `MissingToolName` | `<magnitude:invoke>` without `tool` attribute |
| `UnknownTool` | Tool name not in registry |
| `UnclosedThink` | Reason block not closed before EOF |

### 10.2 Tool Errors

| Error | When |
|---|---|
| `UnknownParameter` | Parameter name not in tool schema |
| `DuplicateParameter` | Same parameter name opened twice in one invoke |
| `MissingRequiredField` | Required parameter not provided when invoke closes |
| `SchemaCoercionError` | Parameter value fails schema type coercion |
| `IncompleteTool` | Invoke not closed before EOF |

---

## 11. Non-`magnitude:` Tags

Tags without the `magnitude:` prefix follow standard XML-like behavior:
- Known structural close tags are tokenized as `Close` tokens.
- Unknown close tags become literal content.
- Unknown open tags become literal content.
- Non-magnitude tags inside bodies are always content — they are never structural.

This means `<div>`, `</div>`, `<foo bar="baz">` etc. are all treated as literal text in body content.
