# xml-act

Streaming XML tool execution runtime. Parses XML from LLM output, validates inline, dispatches Effect-based tools, and emits a structured event stream. No code generation, no sandboxing — just XML → events.

## Architecture

```
LLM XML Stream → Streaming Parser → Runtime Enrichment → Tool Dispatcher → Event Stream
```

The parser is a character-by-character state machine that handles chunk boundaries transparently. Validation and type coercion happen inline during parsing — errors are detected immediately, not deferred.

## Event Lifecycle

Every tool call follows a strict event sequence. The core guarantee: **every `ToolInputStarted` produces exactly one terminal event** — either `ToolExecutionEnded` (success or error) or `ToolInputParseError` (never dispatched). Never both. Never neither.

### Success Path

```
ToolInputStarted           — tag opened, tool identified
ToolInputFieldValue*       — 0+ per attribute (coerced to correct type)
ToolInputBodyChunk*        — 0+ body text chunks
ToolInputChildStarted*     — 0+ per child element
ToolInputChildComplete*    — 0+ per child element
ToolInputReady             — input fully built
ToolExecutionStarted       — interceptor approved, execution begins
ToolProgress*              — 0+ progress updates during execution
ToolExecutionEnded         — result: Success | Error
```

### Error Path

Parse errors kill the tool call immediately. Further streaming for that `toolCallId` is suppressed, but the parser continues structurally so it knows where the tag ends.

```
ToolInputStarted
ToolInputParseError        — tool call dead, never dispatched
```

### Prose Events

Bare text outside tool tags emits `ProseChunk` (streaming) and `ProseEnd` (complete).

### Terminal Event

`ExecutionEnd` fires once when the entire XML stream is consumed.

## Parse Errors

All pre-execution errors emit `ToolInputParseError` with a discriminated `error` field:

| `_tag` | When | Example |
|---|---|---|
| `UnknownAttribute` | Attribute not in binding | `<read verbose="true"/>` when `verbose` not bound |
| `InvalidAttributeValue` | Type coercion failure | `<add count="abc"/>` when `count` expects number |
| `UnexpectedBody` | Body on bodyless tool | `<read path="x">text</read>` when no body binding |
| `UnclosedChildTag` | Parent closes with open child | `<tool><child>...</tool>` |
| `IncompleteToolTag` | Stream ended mid-tag | EOF inside `<read path="...` |
| `MissingRequiredFields` | Required field not provided | `<read/>` when `path` is required |

## Binding System

Bindings map XML structure to tool input schemas. Each binding type serves a different structural purpose.

### Attributes — scalar fields as XML attributes

```typescript
binding: { attributes: ['path', 'limit'] }
```
```xml
<read path="src/index.ts" limit="100"/>
```
Coerces to `string`, `number`, or `boolean` based on schema.

### Body — string field as inner text

```typescript
binding: { attributes: ['path'], body: 'content' }
```
```xml
<write path="f.ts">const x = 5;</write>
```

### childTags — nested scalar fields as child elements

```typescript
binding: {
  attributes: ['id'],
  childTags: [
    { field: 'options.type', tag: 'type' },
    { field: 'options.title', tag: 'title' },
  ],
}
```
```xml
<create id="a1">
  <type>builder</type>
  <title>Build the feature</title>
</create>
```
Produces `{ id: 'a1', options: { type: 'builder', title: 'Build the feature' } }`.

`field` is a dotted path into the input schema. `tag` is the XML element name.

### children — array fields as repeated child elements

```typescript
binding: {
  attributes: ['path'],
  children: [{ field: 'edits', tag: 'change', attributes: ['old'], body: 'new' }],
}
```
```xml
<edit path="f.ts">
  <change old="foo">bar</change>
  <change old="baz">qux</change>
</edit>
```
Produces `{ path: 'f.ts', edits: [{ old: 'foo', new: 'bar' }, { old: 'baz', new: 'qux' }] }`.

### childRecord — Record<string, string> as key-value child elements

```typescript
binding: { childRecord: { field: 'vars', tag: 'var', keyAttr: 'name' } }
```
```xml
<env>
  <var name="PATH">/usr/bin</var>
  <var name="HOME">/home/user</var>
</env>
```
Produces `{ vars: { PATH: '/usr/bin', HOME: '/home/user' } }`.

## Usage

### Define a tool

```typescript
import { createTool } from '@magnitudedev/tools'
import { Schema } from '@effect/schema'
import { Effect } from 'effect'

const readTool = createTool({
  name: 'read',
  description: 'Read a file',
  inputSchema: Schema.Struct({
    path: Schema.String.annotations({ description: 'File path' }),
    limit: Schema.optional(Schema.Number),
  }),
  outputSchema: Schema.String,
  bindings: () => ({
    xml: { type: 'tag', attributes: ['path', 'limit'] },
  }),
  execute: ({ path }) => Effect.succeed(`contents of ${path}`),
})
```

### Create and run the runtime

```typescript
import { createXmlRuntime } from '@magnitudedev/xml-act'
import { Effect, Stream } from 'effect'

const runtime = createXmlRuntime({
  tools: new Map([
    ['read', { tool: readTool, tagName: 'read', groupName: 'fs', binding: { attributes: ['path', 'limit'] } }],
  ]),
  prosePolicy: 'message',
})

const xmlStream = Stream.make('<read path="src/index.ts"/>')
const eventStream = runtime.streamWith(xmlStream)

// Collect all events
const events = await Effect.runPromise(
  Stream.runCollect(eventStream).pipe(Effect.map(c => Array.from(c)))
)
```

### Event stream shape

```typescript
// events:
// [
//   { _tag: 'ToolInputStarted', toolCallId: 'tc_1', toolName: 'read', group: 'fs' },
//   { _tag: 'ToolInputFieldValue', toolCallId: 'tc_1', field: 'path', value: 'src/index.ts' },
//   { _tag: 'ToolInputReady', toolCallId: 'tc_1', input: { path: 'src/index.ts' } },
//   { _tag: 'ToolExecutionStarted', toolCallId: 'tc_1', ... },
//   { _tag: 'ToolExecutionEnded', toolCallId: 'tc_1', result: { _tag: 'Success', output: '...', outputTree: { tag: 'read', tree: ... }, observe: '.' } },
//   { _tag: 'ToolObservation', toolCallId: 'tc_1', tagName: 'read', observe: '.', content: '...' },
//   { _tag: 'ExecutionEnd', result: { _tag: 'Success' } },
// ]
```

## Service Tags (Effect DI)

### ToolInterceptorTag — permission gates

Called before (and optionally after) tool execution. Can modify input or reject the call.

```typescript
interface ToolInterceptor {
  beforeExecute: (ctx: InterceptorContext) => Effect.Effect<InterceptorDecision>
  afterExecute?: (ctx: InterceptorContext & { result: unknown }) => Effect.Effect<InterceptorDecision>
}

type InterceptorDecision =
  | { _tag: 'Proceed'; modifiedInput?: unknown }
  | { _tag: 'Reject'; rejection: unknown }
```

Rejection produces `ToolExecutionEnded { result: Rejected }` and halts the stream with `ExecutionEnd { result: GateRejected }`.

### ToolObserverTag — event stream monitoring

Receives every runtime event in real-time. Should be fast (push to queue, update state).

```typescript
interface ToolObserver {
  onEvent: (event: XmlRuntimeEvent) => Effect.Effect<void>
}
```

### ToolProgressTag — progress emission from tools

Provided to tools during execution. Emits `ToolProgress` events.

```typescript
interface ToolProgressService {
  emit: (update: unknown) => Effect.Effect<void>
}
```

## Observation with `observe`

Every tool call uses an `observe` attribute to control what part of that tool's own output is surfaced after execution.

Use `observe="."` for the full output:

```xml
<read path="src/index.ts" observe="."/>
```

Use an XPath/XQuery expression to select a subset of the tool's output tree:

```xml
<search pattern="TODO" observe="//item[1]/@file"/>
```

Observation is execution-driven:
- `observe` is framework metadata, not tool input.
- The query runs against the current tool call's output tree.
- Invalid or empty queries fall back to the full output.

## Binding Validation

Bindings are validated against the schema AST at `createXmlRuntime()` time — before any XML is parsed. Invalid bindings throw immediately:

- Attribute fields must be scalar (`string | number | boolean`)
- Body field must be `string`
- Child binding fields must be arrays of structs
- childRecord key attribute must be string
- All declared fields must exist in the schema
- childTag field paths must resolve through the schema

This catches developer bugs at registration, not at runtime.

## Robustness

The parser is designed to handle messy LLM output gracefully. Rather than failing on imperfect XML, it applies a set of leniency rules that keep parsing going while still catching real errors.

### Attribute leniency

- **Unquoted attribute values** — `<read limit=100/>` works the same as `<read limit="100"/>`. The parser handles both quoted and unquoted forms.
- **Empty attribute values** — `key=` with no value before a terminator stores an empty string.
- **Boolean coercion** — accepts `true`, `True`, `TRUE`, `1`, `yes`, `Yes`, `YES` (and matching falsy variants). Rejects anything else.
- **`id` and `observe` attributes are always valid** — these framework attrs are never rejected as unknown, regardless of the tool's binding.
- **Unknown child attributes pass through** — unrecognized attributes on child elements are silently kept as strings (buildInput drops them if they're not in the binding).

### Unknown/malformed tag recovery

- **Unknown tags emitted as prose immediately** — as soon as the parser sees `>` on `<foo>` and `foo` is not a known tool, the opening tag is reconstructed as prose and the parser returns to prose state. The body flows through as normal prose characters — no buffering. The close tag `</foo>` is also emitted as prose when it arrives.
- **Unknown self-closing tags** — same treatment: `<unknown attr="x"/>` becomes prose immediately.
- **Invalid child tags flushed to parent body** — if a child element name doesn't match the parent tool's valid children, the `<tagname...` text is flushed back into the parent's body text, not treated as an error.
- **Mismatched close tags** — if a close tag doesn't match the current open tag, the entire element is reconstructed as prose.

### Stream resilience

- **Chunk boundary transparency** — the state machine maintains full state across arbitrary chunk splits. A tag can be split across any number of chunks with no buffering or special handling needed.
- **Dead tool call suppression** — once a `ToolInputParseError` fires for a tool call, all further streaming events for that `toolCallId` are silently suppressed. The parser continues structurally (to know where the tag ends) but emits nothing for it.
- **Incomplete tags on flush** — if the stream ends mid-tag, known tool tags get an `IncompleteToolTag` error. The partial content is reconstructed as prose.
- **CDATA with prefix mismatch fallback** — `<![CDATA[...]]>` is fully supported in prose, parent body, and child body. If the prefix doesn't fully match (e.g. `<!D` instead of `<![CDATA[`), the accumulated buffer is flushed as normal body/prose text.

### Structural tag auto-close

- **Omitted closing tags on structural blocks** — if the model omits a closing tag for a structural block (`lenses`, `comms`, `actions`) and opens a later one in the sequence (`lenses` → `comms` → `actions` → `next`/`yield`), the earlier block is auto-closed. For example, `<lenses>...<comms>` auto-closes `lenses` before opening `comms`, and `<actions>...<yield/>` auto-closes `actions` before emitting turn control. No parse errors are emitted — this is intentional recovery.

### Prose cleanup

- **Code fence stripping** — lines matching `` ```xml `` or `` ``` `` are stripped from prose output. LLMs frequently wrap XML in markdown code fences; the parser silently removes them.
- **Prose trimming** — leading/trailing whitespace in prose buffers is trimmed before emission.
