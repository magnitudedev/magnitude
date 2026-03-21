# Tool System Architecture Spec

## Overview

The tool system is decomposed into five independent contracts connected by a typed generic chain. Each contract can be developed, tested, and swapped independently. Type safety flows through the entire chain at compile time; types are erased at runtime behind opaque interfaces.

---

## The Five Contracts

```
                    ┌─────────────────────┐
                    │   Tool Function     │
                    │   (pure typed IO)   │
                    └──────┬──────────────┘
                           │ TInput, TOutput, TEmission
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼──────┐  ┌──────▼──────┐  ┌──────▼───────┐
    │ Parser     │  │ Display     │  │ Other        │
    │ Binding    │  │ Binding     │  │ Bindings     │
    │ (XML, JSON)│  │ (model +   │  │ (OpenAI,     │
    │            │  │  renderer)  │  │  test, etc.) │
    └────────────┘  └─────────────┘  └──────────────┘
```

### 1. Tool Function

A pure typed function: `TInput → TOutput + TEmission[]`. No knowledge of parsing, display, or lifecycle.

```typescript
const editTool = defineTool({
  name: 'fileEdit',
  inputSchema: S.Struct({
    path: S.String,
    replaceAll: S.optional(S.Boolean),
    oldString: S.String,
    newString: S.String,
  }),
  outputSchema: EditOutputSchema,
  emissionSchema: S.Struct({ diffs: S.Array(DiffSchema), path: S.String }),

  execute: (input, ctx) => Effect.gen(function*() {
    const result = yield* applyEdit(input.path, input.oldString, input.newString);
    yield* ctx.emit({ diffs: result.diffs, path: input.path });
    return result.output;
  }),

  label: (input) => `Editing ${input.path ?? '...'}`,
});
```

**Depends on:** `tool-contract` types only.
**Does not know about:** XML, display, lifecycle, adapters.
**Testable via:** Direct `execute(input, mockCtx)` calls.

### 2. Parser Binding (XML)

Maps XML structure ↔ tool IO. Separate from the tool. Type-safe against the tool's input/output types via `keyof TInput` constraints.

```typescript
const editXmlBinding = defineXmlBinding(editTool, {
  group: 'fs',
  input: {
    attributes: [
      { attr: 'path',       field: 'path' },
      { attr: 'replaceAll', field: 'replaceAll' },
    ],
    childTags: [
      { tag: 'old', field: 'oldString' },
      { tag: 'new', field: 'newString' },
    ],
  },
  output: {},
} as const);
```

`field` values are constrained to `keyof TInput`. TypeScript rejects typos. The `as const` preserves literal types for streaming shape derivation.

**Depends on:** `tool-contract` types, references tool function's type parameters.
**Does not know about:** display, state models, adapters.
**Testable via:** `buildInput(mockParsedElement, binding.input)` → verify output matches TInput shape.

### 3. Display Binding

Composed of two sub-contracts: **state model** and **display renderer**, paired via `createBinding`.

#### State Model

Pure synchronous reducer. Consumes normalized events + accumulated streaming input. Produces abstract display state. Reusable across tools with compatible input structures.

```typescript
const diffModel = defineStateModel<
  DiffState,                            // TState
  { path?: string },                    // TAccFields (minimum fields needed)
  { old?: ChildAcc[]; new?: ChildAcc[] }, // TAccChildren (minimum children needed)
  { diffs: Diff[] }                     // TEmission (emission shape needed)
>({
  initial: { phase: 'streaming', diffs: [] },
  reduce: (state, event, acc) => {
    switch (event.type) {
      case 'executionStarted': return { ...state, phase: 'executing' };
      case 'emission':         return { ...state, diffs: event.value.diffs };
      case 'completed':        return { ...state, phase: 'completed' };
      case 'error':            return { ...state, phase: 'error' };
      case 'interrupted':      return { ...state, phase: 'interrupted' };
      default:                 return state;
    }
  }
});
```

**Key property:** Does NOT track streaming input accumulation. The framework provides `acc` with typed fields/children. The model only tracks semantic state (phase, diffs).

**Depends on:** `tool-contract` types only.
**Does not know about:** XML, parsing, rendering, specific tools.
**Testable via:** Feed events + mock acc directly, assert state transitions.

#### Display Renderer

Coupled to a specific state model's TState. Renders abstract state + accumulated input. Per UI system (TUI, web, etc.).

```typescript
const diffTuiDisplay = defineDisplay(diffModel, {
  render: ({ state, acc, label, result, isExpanded, onToggle, onFileClick }) => {
    if (state.phase === 'streaming') {
      const oldText = acc.children?.old?.[0]?.body ?? '';
      const newText = acc.children?.new?.[0]?.body ?? '';
      return <Box flexDirection="column">
        <ToolHeader label={label} phase="streaming" onToggle={onToggle} />
        <LiveDiffHunk old={oldText} new={newText} />
      </Box>;
    }
    if (state.phase === 'completed' && state.diffs.length > 0) {
      return <Box flexDirection="column">
        <ToolHeader label={label} phase="completed" onToggle={onToggle} />
        {state.diffs.map((d, i) => <DiffHunk key={i} diff={d} onFileClick={onFileClick} />)}
      </Box>;
    }
    return <ToolHeader label={label} phase={state.phase} onToggle={onToggle} />;
  },
  liveText: ({ state, acc }) => {
    const path = acc.fields?.path ?? '...';
    if (state.phase === 'completed') return `Edited ${path} (${state.diffs.length} hunks)`;
    return `Editing ${path}`;
  },
});
```

**What the display reads:**
- `state` — semantic state from the model (phase, diffs)
- `acc` — streaming input from the adapter (fields, body, children)
- `label` — from the tool's label function
- `result` — raw tool result from the adapter
- `isExpanded`, `onToggle`, `onFileClick` — UI framework props

**Depends on:** `tool-contract` types, state model types, UI framework (React).
**Does not know about:** XML, parsing, tool implementation, events.
**Testable via:** Render with mock state + acc, assert output.

### 4. Tool Event Adapter

Framework infrastructure. Converts raw temporal events from multiple sources into a normalized `ToolStateEvent` stream. Maintains `StreamingInput` per call via schema-driven accumulation. Per-parser, not per-tool.

**Event sources consumed:**
- xml-act `ToolCallEvent` (parse events)
- Background process events (`background_process_output`, `background_process_exited`, etc.)
- Approval events (`awaitingApproval`, `approvalGranted`, `approvalRejected`)
- Interruption signals

**Produces:**
- Normalized `ToolStateEvent` stream per call
- `StreamingInput` accumulated snapshot per call

**Maintains:**
- `SchemaAccumulator` per call (schema-driven, generic for all tools)
- `pid → callId` mappings (for background process correlation)
- Per-call state model instances (via opaque bindings)

```typescript
class ToolEventAdapter {
  handleToolEvent(toolKey: string, callId: string, event: ToolCallEvent): void {
    const call = this.getOrCreateCall(callId, toolKey);
    call.accumulator.ingest(event);                              // schema-driven accumulation
    const normalized = this.normalize(event);                     // ToolCallEvent → ToolStateEvent
    if (normalized) {
      call.binding.reduce(call.state, normalized, call.accumulator.current);
    }
  }

  handleProcessOutput(pid: number, text: string): void {
    const callId = this.pidToCallId.get(pid);
    if (!callId) return;
    const call = this.calls.get(callId)!;
    call.binding.reduce(call.state, { type: 'processOutput', text, stream: 'stdout' }, call.accumulator.current);
  }
}
```

**Depends on:** `tool-contract` types, xml-act event types, tool schemas.
**Does not know about:** specific tools, specific state models, rendering.

### 5. Wiring Registry (Composition Root)

The only place that knows which tool uses which parser binding, which state model, and which display. Type safety enforced at binding creation.

```typescript
// Pair model + display (type-checked: TState must match)
const editDisplayBinding = createBinding(diffModel, diffTuiDisplay);

// Register XML bindings
xmlRuntime.register(editTool, editXmlBinding);
xmlRuntime.register(shellTool, shellXmlBinding);

// Register display bindings
displayRegistry.register('fileEdit',  createBinding(diffModel, diffTuiDisplay));
displayRegistry.register('fileWrite', createBinding(contentModel, contentTuiDisplay));
displayRegistry.register('shell',     createBinding(shellModel, shellTuiDisplay));
// All other tools fall through to default binding
```

---

## Normalized Event Contract

The abstraction boundary between the adapter and state models. ~16 variants covering the full tool lifecycle.

```typescript
type ToolStateEvent<TEmission = never> =
  // Lifecycle
  | { type: 'started' }
  | { type: 'inputUpdated'; changed: 'field' | 'body' | 'child'; name?: string }
  | { type: 'inputReady'; input: unknown }
  | { type: 'parseError'; error: string }

  // Approval
  | { type: 'awaitingApproval'; preview?: unknown }
  | { type: 'approvalGranted' }
  | { type: 'approvalRejected' }

  // Execution
  | { type: 'executionStarted' }
  | { type: 'emission'; value: TEmission }
  | { type: 'completed'; output: unknown }
  | { type: 'error'; error: Error }
  | { type: 'rejected' }
  | { type: 'interrupted' }

  // Post-execution async (background processes)
  | { type: 'processOutput'; text: string; stream: 'stdout' | 'stderr' }
  | { type: 'processExited'; exitCode: number }
  | { type: 'processPromoted' }
  | { type: 'processDemoted'; logPath: string }
```

`TEmission` is the only typed parameter — flows from tool definition through the chain.

---

## Streaming Input Contract

The accumulated input snapshot maintained by the adapter. Typed per-binding via derived streaming shape.

```typescript
type StreamingInput<
  TFields = Record<string, string>,
  TChildren = Record<string, any>
> = {
  fields: Partial<TFields>;
  body: string;
  children: {
    [K in keyof TChildren]?: Array<{
      body: string;
      complete: boolean;
      attrs: Record<string, string>;
    }>;
  };
}
```

Built mechanically by `SchemaAccumulator` from parse events. No per-tool logic. The type parameters are derived from the XML binding's mapping.

---

## Generic Type Chain

Types flow through the entire system with compile-time verification at each boundary.

### Type Derivation Flow

```
Tool: TInput = { path: string; oldString: string; newString: string }
                                    │
                                    ▼ keyof TInput constrains field mappings
XML Binding: TMapping = {
  attributes: [{ attr:'path', field:'path' }, { attr:'replaceAll', field:'replaceAll' }],
  childTags:  [{ tag:'old', field:'oldString' }, { tag:'new', field:'newString' }],
}
                                    │
                                    ▼ DeriveStreamingShape<TMapping>
Streaming Shape = {
  fields:   { path?: string; replaceAll?: string }
  body:     ''
  children: { old?: ChildAcc[]; new?: ChildAcc[] }
}
                                    │
                                    ▼ structural subtyping check
State Model requires: {
  fields:   { path?: string }                       ⊆ streaming fields ✓
  children: { old?: ChildAcc[]; new?: ChildAcc[] }  ⊆ streaming children ✓
  emission: { diffs: Diff[] }                       ⊆ TEmission ✓
}
                                    │
                                    ▼ same TState
Display accepts: DiffState ✓
```

### Key Generic Mechanisms

**`const` type parameters (TS 5.0+):** Preserves literal string types in binding definitions. Without this, `'old'` widens to `string` and the derivation chain collapses.

```typescript
function defineXmlBinding<
  TInput, TOutput, TEmission,
  const TMapping extends { ... }   // ← const preserves literals
>(tool, config): XmlBinding<TInput, TOutput, TEmission, TMapping>
```

**Streaming shape derivation:** Mapped types + conditional types extract field/child names from the binding.

```typescript
type AttrNames<T> = T extends readonly { attr: infer A extends string }[] ? A : never;
type ChildTagNames<T> = T extends readonly { tag: infer T extends string }[] ? T : never;

type DeriveStreamingShape<TMapping> = {
  fields:   { [K in AttrNames<TMapping['attributes']>]?: string };
  body:     TMapping extends { body: string } ? string : '';
  children: { [K in ChildTagNames<TMapping['childTags']>]?: ChildAcc[] };
}
```

**Existential type erasure:** After type verification, chains are stored as opaque bindings in the registry. The adapter operates on the opaque interface without knowing the internal types.

```typescript
function createBinding<TState, TAccFields, TAccChildren, TEmission>(
  stateModel: StateModel<TState, TAccFields, TAccChildren, TEmission>,
  display: Display<TState, TAccFields, TAccChildren>,
): ToolDisplayBinding {
  // Types verified here, erased after
  return {
    createCallState: () => ({ state: stateModel.initial }),
    reduce: (callState, event, acc) => {
      callState.state = stateModel.reduce(callState.state, event, acc);
    },
    render: (callState, props) => display.render({ state: callState.state, ...props }),
  };
}
```

**Builder pattern for chain verification:** Breaks type inference into incremental steps rather than solving 12+ parameters at once.

```typescript
const chain = forTool(editTool)
  .withXmlBinding(editXmlBinding)     // locks TMapping, derives StreamingShape
  .withStateModel(diffModel)          // checks StreamingShape ⊇ model requirements
  .withDisplay(diffTuiDisplay)        // checks TState matches
  .build();
```

### Namespace Mapping

The binding creates a mapping between two namespaces:
- **XML namespace:** attribute names (`path`, `replaceAll`), tag names (`old`, `new`)
- **Tool namespace:** field names on TInput (`path`, `replaceAll`, `oldString`, `newString`)

The streaming shape uses XML namespace (because the parser works in that namespace). The tool function uses tool namespace. The binding is the bridge. State models and displays work in XML namespace via `acc`. They never need to know the tool namespace mapping.

---

## State Model × Tool Matrix

| State Model | What It Tracks | Tools |
|---|---|---|
| `defaultModel` | Phase only | read, tree, search, view, web-search, web-fetch, browser tools, agent tools |
| `diffModel` | Phase + diffs from emissions | edit (+ future diff tools) |
| `contentModel` | Phase + streaming line count | write (+ future content tools) |
| `shellModel` | Phase + output + detached/bg lifecycle | shell, shell-bg |
| `progressModel` | Phase + progress % from emissions | (future: long-running tasks) |

~5 models for ~15+ tools. Most tools use `defaultModel` with zero custom code.

---

## Display Props Contract

What every display renderer receives:

```typescript
type DisplayProps<TState, TAccFields, TAccChildren> = {
  state: TState;                                        // from state model
  acc: StreamingInput<TAccFields, TAccChildren>;        // from adapter accumulator
  label: string;                                        // from tool.label()
  result?: ToolResult;                                  // raw tool result
  isExpanded: boolean;                                  // UI state
  onToggle: () => void;                                 // UI callback
  onFileClick?: (path: string, section?: string) => void; // UI callback
}
```

---

## Emission Channel

Replaces `ToolEmitTag` (single-value Ref side-channel) with a typed emission stream.

```typescript
type ToolContext<TEmission> = {
  emit: (value: TEmission) => Effect<void>;
}
```

- Typed per-tool via `TEmission` parameter
- Supports multiple emissions per execution (queue, not last-write-wins)
- Each emission becomes a `{ type: 'emission', value: TEmission }` event in the normalized stream
- Current tools (edit, write) emit once at end — behavior unchanged
- Future tools can emit progress updates, intermediate results, etc.

---

## End-to-End Flow: Edit Tool

```
1. Model writes: <edit path="foo.ts"><old>x = 1</old><new>x = 2</new></edit>

2. xml-act parser emits ToolCallEvents (character by character)

3. Adapter receives events:
   ToolInputStarted → creates SchemaAccumulator(editSchema), looks up binding
   ToolInputFieldValue {field:'path'} → acc.fields.path = 'foo.ts', emit {type:'inputUpdated'}
   ToolInputBodyChunk {path:['old',0]} → acc.children.old[0].body += text, emit {type:'inputUpdated'}
     → display re-renders: reads acc.children.old[0].body → shows <LiveDiffHunk>
   ToolInputReady → emit {type:'inputReady', input}
   ToolExecutionStarted → emit {type:'executionStarted'}
     → diffModel.reduce → phase = 'executing'
   ToolEmission {value:{diffs}} → emit {type:'emission', value:{diffs}}
     → diffModel.reduce → diffs = [...]
   ToolExecutionEnded → emit {type:'completed'}
     → diffModel.reduce → phase = 'completed'
     → display re-renders: shows final <DiffHunk> from state.diffs

4. Display reads:
   - state.phase, state.diffs (from state model — semantic interpretation)
   - acc.fields.path, acc.children.old[0].body (from accumulator — raw streaming data)
   - label (from tool.label(acc))
```

---

## What Gets Eliminated

| Current | Replacement |
|---------|-------------|
| `packages/agent/src/visuals/` (reducers) | State models in shared package |
| `ToolDisplay` union in `events.ts` | Per-tool typed emissions |
| `ToolEmitTag` (single-value Ref) | Typed emission channel (queue) |
| `visualState: unknown` on ToolStep | Typed through existential binding |
| Per-reducer input accumulation boilerplate | Framework-provided `SchemaAccumulator` |
| Bindings bundled inside Tool object | Separate `defineXmlBinding` |
| Registry wiring connecting agent reducers to CLI renderers | Co-located `createBinding(model, display)` |

---

## Dependency Graph

```
tool-contract (all contract types and helpers)
    ↑           ↑           ↑           ↑
tool funcs   xml bindings  state models  displays
    ↑           ↑                          ↑
    └─ xml-act runtime ──┐                 │
              ↑           │                 │
         execution mgr ───┤                 │
              ↑           │                 │
         adapter ─────────┘─────────────────┘
              ↑
         app event bus

wiring registry (composition root — imports everything)
```

No circular dependencies. Each piece depends only downward on `tool-contract`.

---

## What's Static, Generic, Custom

| Layer | Static (all tools) | Generic (parameterized) | Custom (per tool/category) |
|-------|-------------------|------------------------|---------------------------|
| Tool function | `ToolContext` API, `defineTool` | Schema-parameterized IO | Per-tool execute + label |
| XML binding | `defineXmlBinding` API | Mapping constrained by TInput | Per-tool attr/child/body mapping |
| Adapter | Event normalization, phase lifecycle | Schema-driven accumulation | Per-parser event source |
| State model | `ToolStateEvent` union, reduce signature | `defaultModel` (phase only) | `diffModel`, `shellModel`, etc. |
| Display | `DisplayProps` contract | `defaultDisplay` | `diffDisplay`, `shellDisplay`, etc. |
| Wiring | Registry mechanics, `createBinding` | Default binding fallback | Per-tool registration |

---

## Migration Path

**Phase 1:** Define `tool-contract` types (`ToolStateEvent`, `StreamingInput`, `defineTool`, `defineXmlBinding`, `defineStateModel`, `defineDisplay`, `createBinding`). Port existing tools to separated definitions. Run alongside existing system.

**Phase 2:** Port existing reducers to state models. Port existing renderers to displays. Co-locate each pair via `createBinding`. Verify parity.

**Phase 3:** Replace `ToolEmitTag` with typed emission channel. Remove `ToolDisplay` union from `events.ts`. Update execution manager.

**Phase 4:** Implement `ToolEventAdapter`. Normalize background process events through the same contract. Replace DisplayProjection's tool handling.

**Phase 5:** Separate XML bindings from tool definitions. Update xml-act registration to use `defineXmlBinding`. Remove bindings from `createTool`.

Each phase is incremental. Old and new systems can coexist during migration.
