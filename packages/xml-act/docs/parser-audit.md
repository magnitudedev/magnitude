
# XML-Act Parser Audit Findings

Audit of silent abandonment, missing errors, and data loss across the tokenizer, parser, and turn engine.

**Scope:**
- `packages/xml-act/src/tokenizer.ts`
- `packages/xml-act/src/parser/index.ts`
- `packages/xml-act/src/engine/turn-engine.ts`

---

## Classifications

| Label | Meaning |
|---|---|
| `SILENT_ABSORB` | Input is swallowed into content/prose with no error event |
| `SILENT_DROP` | Input is discarded entirely with no error event |
| `MISSING_ERROR` | A structural problem is detected but no error event is emitted |
| `BUG` | Incorrect behavior with no data loss |

---

## Critical

### T10 — `knownToolTags` hardcoded to empty set
**Location:** `engine/turn-engine.ts`  
**Classification:** `MISSING_ERROR`

```ts
const tokenizer = createTokenizer(
  (token) => { parser.pushToken(token) },
  new Set(),  // always empty
  { toolKeyword: 'invoke' },
)
```

`knownToolTags` is always `new Set()`. The invoke-without-keyword leniency (where `<|shell>` is rewritten to `<|invoke:shell>`) never fires in production. The feature was implemented in the tokenizer but is dead code at runtime.

---

### T1 — Malformed known-tool tags become content
**Location:** `tokenizer.ts` — `failAsContent()` call sites  
**Classification:** `SILENT_ABSORB`

When a tag whose name matches `toolKeyword` (e.g. `invoke`) has an invalid character in the name, colon position, variant, or pipe position, `failAsContent()` dumps the raw text into `contentBuffer`. No error event is emitted. The LLM receives no structured feedback.

Affected call sites:
- `open_name`: invalid char in name mid-parse (e.g. `<|invoke!:shell>`)
- `open_colon`: invalid char after `:` (e.g. `<|invoke:!>`)
- `open_variant`: whitespace or invalid char in variant (e.g. `<|invoke:she!ll>`)
- `open_pipe`: char after `|` is not `>` (e.g. `<|invoke:shell|x>`)

The old tokenizer had a `malformed` phase that committed known-tool tags even when malformed, allowing the parser to emit a structured error. This is absent in the new tokenizer.

---

### T2 — Active invoke tag at EOF becomes content
**Location:** `tokenizer.ts` — `end()`  
**Classification:** `SILENT_ABSORB`

```ts
if (activeTag) {
  failAsContent()
}
```

If the stream ends mid-tag (e.g. `<|invoke:shell`), the partial tag is dumped as content. For known-tool tags this means a truncated tool call produces no error event.

---

### T11 / T8 — Filter queries never reach the dispatcher
**Location:** `engine/turn-engine.ts`, `parser/index.ts`  
**Classification:** `SILENT_DROP`

Filter content is accumulated in `FilterFrame` by the parser, but the turn engine always initializes `filterQuery: null`:

```ts
activeInvokes.set(event.toolCallId, { tagName: event.tagName, filterQuery: null })
```

Filter queries never reach `dispatchTool`. The content is dropped at the engine boundary.

---

## Data Loss

### T5 — Content between parameters in invoke frame is dropped
**Location:** `parser/index.ts`  
**Classification:** `SILENT_DROP`

```ts
case 'invoke': {
  // Content between parameters — ignore
  break
}
```

Any text arriving while an invoke frame is on top (between `<parameter|>` and `<|parameter:next>`) is discarded with no event and no error.

---

### T6 — Orphan `<parameter|>` in non-prose frames is dropped
**Location:** `parser/index.ts` — `handleParameterClose`  
**Classification:** `SILENT_DROP`

```ts
if (!paramFrame) {
  const top = machine.peek()
  if (top?.type === 'prose') machine.apply(appendProse(top, '<parameter|>'))
  return
}
```

Only appends to prose if the top frame is prose. If the top frame is `think`, `message`, or `invoke`, the `<parameter|>` is silently dropped. Should use `appendUnknownContent` to handle all frame types.

---

### T7 — `finalizeInvoke` reports only the first errored/missing field
**Location:** `parser/index.ts` — `finalizeInvoke`  
**Classification:** `SILENT_DROP`

```ts
for (const [, fieldState] of invokeFrame.fieldStates) {
  if (fieldState.errored) {
    // emit error for this field, pop, return
    return
  }
}
```

The loop returns after the first errored field. Additional errored or missing required fields are silently dropped. Only one `ParseError` is emitted even when multiple fields fail.

---

### T12 — `ToolInputReady` with no invoke entry is dropped
**Location:** `engine/turn-engine.ts`  
**Classification:** `SILENT_DROP`

```ts
if (!invoke) {
  activeInvokes.delete(event.toolCallId)
  break
}
```

If `ToolInputReady` arrives for a `toolCallId` with no entry in `activeInvokes`, the tool call is silently dropped — no dispatch, no error.

---

## Missing Errors

### T3 — Stray close tags for known structural tags become prose
**Location:** `parser/index.ts` — `appendUnknownContent`  
**Classification:** `MISSING_ERROR`

A `<message|>` arriving outside a message frame, or `</invoke|>` outside an invoke frame, falls to `appendUnknownContent` and is injected as raw text into the current frame's content stream. No error event is emitted.

---

### T4 — `<|invoke>` without tool name becomes prose
**Location:** `parser/index.ts` — `handleOpen('invoke')`  
**Classification:** `SILENT_ABSORB`

```ts
if (!variant) {
  const raw = '<|invoke>'
  if (top?.type === 'prose') machine.apply(appendProse(top, raw))
  return
}
```

An invoke tag with no tool name is absorbed into prose with no error event.

---

## Bugs

### T13 — `ToolInputFieldChunk` coalescing is a no-op
**Location:** `parser/index.ts` — `mergeEvent`  
**Classification:** `BUG` (no data loss)

```ts
function mergeEvent(target, source) {
  if ('text' in target && 'text' in source) {
    target.text += source.text
  }
}
```

`ToolInputFieldChunk` uses `delta`, not `text`. The coalescing merge function checks for `text`, so field chunks are never merged — each chunk is emitted separately. No data is lost, but coalescing is broken for this event type.

---

## Summary Table

| # | Location | Classification | Description |
|---|---|---|---|
| T1 | `tokenizer.ts` — `failAsContent()` | `SILENT_ABSORB` | Malformed known-tool tags become content, no error |
| T2 | `tokenizer.ts` — `end()` | `SILENT_ABSORB` | Active invoke tag at EOF becomes content, no error |
| T3 | `parser/index.ts` — `appendUnknownContent` | `MISSING_ERROR` | Stray close tags for known structural tags → prose, no error |
| T4 | `parser/index.ts` — `handleOpen('invoke')` | `SILENT_ABSORB` | `<\|invoke>` with no tool name → prose, no error |
| T5 | `parser/index.ts` — invoke frame content | `SILENT_DROP` | Text between parameters dropped, no error |
| T6 | `parser/index.ts` — `handleParameterClose` | `SILENT_DROP` | Orphan `<parameter\|>` in non-prose frames dropped |
| T7 | `parser/index.ts` — `finalizeInvoke` | `SILENT_DROP` | Only first errored/missing field reported |
| T8 | `parser/index.ts` — `FilterFrame` | `SILENT_DROP` | Filter query accumulated but never surfaced |
| T9 | `parser/index.ts` — post-yield | `SILENT_DROP` (intentional) | All tokens after yield dropped, content unrecoverable |
| T10 | `engine/turn-engine.ts` — `knownToolTags` | `MISSING_ERROR` | Invoke-without-keyword leniency is dead code in production |
| T11 | `engine/turn-engine.ts` — `filterQuery` | `SILENT_DROP` | Filter queries never reach dispatcher |
| T12 | `engine/turn-engine.ts` — `ToolInputReady` | `SILENT_DROP` | Tool call silently dropped when no invoke entry exists |
| T13 | `parser/index.ts` — `mergeEvent` | `BUG` | `ToolInputFieldChunk` uses `delta` not `text`; merge is no-op |
