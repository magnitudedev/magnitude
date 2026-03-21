# Tool System Implementation Spec

## Overview

This spec defines a clean replacement of the current tool system. No incremental migration — the old system is removed and replaced wholesale. The new system decomposes tools into 5 independent contracts connected by a typed generic chain, with clean package ownership boundaries.

---

## Package Ownership

### `@magnitudedev/tools` — Abstract Contracts

All system-level interfaces and types. Zero internal dependencies.

**Owns:**
- `defineTool` — tool function factory
- `ToolDefinition<TInput, TOutput, TEmission>` — tool type
- `ToolContext<TEmission>` — execution context (emission only)
- `defineStateModel` — state model factory
- `StateModel<TState, TAccFields, TAccChildren, TEmission>` — reducer interface
- `ToolStateEvent<TEmission>` — normalized event union (~16 variants)
- `defineDisplay` — display renderer factory  
- `Display<TState, TAccFields, TAccChildren, TRender>` — renderer interface
- `DisplayProps<TState, TAccFields, TAccChildren>` — what displays receive
- `createBinding` — pairs model + display with type verification, erases to opaque `ToolDisplayBinding`
- `ToolDisplayBinding` — opaque paired binding
- `StreamingInput<TFields, TChildren>` — accumulated input snapshot type
- `ToolEventAdapter` — interface for event normalization + state management
- `DisplayBindingRegistry` — interface for looking up display bindings by tool key

### `@magnitudedev/xml-act` — XML-Specific Contracts & Runtime

Depends on `tools` for abstract types.

**Owns:**
- `defineXmlBinding` — XML binding factory with `const TMapping` generic
- `XmlBinding<TInput, TOutput, TEmission, TMapping>` — typed XML mapping
- `DeriveStreamingShape<TMapping>` — type-level streaming shape derivation
- `AttrNames`, `ChildTagNames` — type utilities for shape derivation
- `SchemaAccumulator` — ingests XML parse events, produces `StreamingInput` snapshots
- XML parser, runtime, reactor, dispatcher (existing, adapted)
- `XmlRuntimeEvent`, `ToolCallEvent` types (existing)

### `@magnitudedev/agent` — Implementations & Instances

Depends on `tools`, `xml-act`.

**Owns:**
- All tool function instances (`defineTool` calls)
- All XML binding instances (`defineXmlBinding` calls), co-located with tools
- All state model instances (`defineStateModel` calls)
- `ToolEventAdapter` implementation
- Tool registration (`buildRegisteredTools`)
- Execution manager, permission gate, background process registry

### `cli/` — Display Implementations & Composition Root

Depends on `agent`, `tools`.

**Owns:**
- All display renderer instances (`defineDisplay` calls)
- Wiring registry — composition root that calls `createBinding(model, display)` for each pair

---

## Contract Definitions

### 1. `defineTool`

```typescript
// packages/tools/src/tool.ts

interface ToolConfig<TInput, TOutput, TEmission, R> {
  name: string;
  description?: string;
  group?: string;
  inputSchema: S.Schema<TInput>;
  outputSchema: S.Schema<TOutput>;
  emissionSchema?: S.Schema<TEmission>;
  execute: (input: TInput, ctx: ToolContext<TEmission>) => Effect.Effect<TOutput, ToolError, R>;
  label: (input: Partial<TInput>) => string;
}

interface ToolContext<TEmission> {
  emit: (value: TEmission) => Effect.Effect<void>;
}

interface ToolDefinition<TInput, TOutput, TEmission = never> {
  readonly name: string;
  readonly description?: string;
  readonly group?: string;
  readonly inputSchema: S.Schema<TInput>;
  readonly outputSchema: S.Schema<TOutput>;
  readonly emissionSchema?: S.Schema<TEmission>;
  readonly execute: (input: TInput, ctx: ToolContext<TEmission>) => Effect.Effect<TOutput, ToolError, any>;
  readonly label: (input: Partial<TInput>) => string;
}

function defineTool<TInput, TOutput, TEmission = never, R = never>(
  config: ToolConfig<TInput, TOutput, TEmission, R>
): ToolDefinition<TInput, TOutput, TEmission>;
```

**Key differences from current `createTool`:**
- No `bindings` on the tool object — XML binding is separate
- No `errorSchema` — errors use a standard `ToolError` type (just `{ message: string }`)
- `execute` receives `ToolContext<TEmission>` as second argument instead of using `ToolEmitTag` service
- `label` function on the tool itself, not in visual reducers
- `R` (Effect requirements) is preserved in the execute signature but erased from `ToolDefinition` — the execution manager provides layers at runtime

**What happens to current Effect service dependencies:**
- `ToolEmitTag` → replaced by `ctx.emit()` 
- `ToolReminderTag` → stays as Effect service tag (orthogonal to tool contract)
- `WorkingDirectoryTag`, `ForkContext`, `BackgroundProcessRegistryTag`, etc. → stay as Effect service tags, provided via `layerProvider` at registration time (unchanged from current)

### 2. `defineXmlBinding`

```typescript
// packages/xml-act/src/xml-binding.ts

interface XmlMappingConfig<TInput> {
  group?: string;
  tag?: string;
  input: {
    attributes?: readonly { attr: string; field: keyof TInput & string }[];
    body?: keyof TInput & string;
    childTags?: readonly { tag: string; field: keyof TInput & string }[];
    children?: readonly {
      field: keyof TInput & string;
      tag?: string;
      attributes?: readonly { attr: string; field: string }[];
      body?: string;
    }[];
    childRecord?: {
      field: keyof TInput & string;
      tag: string;
      keyAttr: string;
    };
  };
  output: XmlOutputConfig<TOutput>;  // output binding stays here
}

function defineXmlBinding<
  TInput, TOutput, TEmission,
  const TMapping extends XmlMappingConfig<TInput>
>(
  tool: ToolDefinition<TInput, TOutput, TEmission>,
  config: TMapping
): XmlBinding<TInput, TOutput, TEmission, TMapping>;
```

**`field` values are constrained to `keyof TInput`** — TypeScript rejects typos at compile time.

**`const TMapping`** preserves literal string types so that `DeriveStreamingShape` can extract field/child names.

**Output binding lives here** (not on the tool) because it's XML-specific serialization. The tool itself doesn't know how its output gets serialized.

#### Streaming Shape Derivation

```typescript
// packages/xml-act/src/type-chain.ts

type AttrNames<T> = T extends readonly { attr: infer A extends string }[] ? A : never;
type ChildTagNames<T> = T extends readonly { tag: infer T extends string }[] ? T : never;
type BodyFieldNames<T> = T extends readonly { field: infer F extends string }[] ? F : never;

type DeriveStreamingShape<TMapping> = {
  fields: { [K in AttrNames<TMapping['input']['attributes']>]?: string };
  body: TMapping['input'] extends { body: string } ? string : '';
  children: {
    [K in (
      ChildTagNames<TMapping['input']['childTags']> |
      ChildTagNames<TMapping['input']['children']>
    )]?: ChildAcc[];
  };
};

type ChildAcc = {
  body: string;
  complete: boolean;
  attrs: Record<string, string>;
};
```

#### SchemaAccumulator

```typescript
// packages/xml-act/src/schema-accumulator.ts

class SchemaAccumulator<TMapping> {
  constructor(mapping: TMapping);
  
  /** Ingest a raw ToolCallEvent from the xml-act parser */
  ingest(event: ToolCallEvent): void;
  
  /** Current accumulated snapshot */
  get current(): StreamingInput<DeriveFields<TMapping>, DeriveChildren<TMapping>>;
  
  /** Reset for reuse */
  reset(): void;
}
```

**Replaces** all manual accumulation in current reducers. The accumulator:
- On `ToolInputFieldValue`: sets `fields[attr] = value`
- On `ToolInputBodyChunk`: appends to `body`
- On `ToolInputChildStarted`: creates new `ChildAcc` entry in `children[tag]`
- On `ToolInputBodyChunk` (child path): appends to child's `body`
- On `ToolInputChildComplete`: marks child `complete = true`

This is generic — works for all tools based on the XML binding mapping. No per-tool code.

### 3. `defineStateModel`

```typescript
// packages/tools/src/state-model.ts

interface StateModelConfig<TState, TAccFields, TAccChildren, TEmission> {
  initial: TState;
  reduce: (
    state: TState,
    event: ToolStateEvent<TEmission>,
    acc: StreamingInput<TAccFields, TAccChildren>
  ) => TState;
}

function defineStateModel<TState, TAccFields = {}, TAccChildren = {}, TEmission = never>(
  config: StateModelConfig<TState, TAccFields, TAccChildren, TEmission>
): StateModel<TState, TAccFields, TAccChildren, TEmission>;
```

**Key differences from current reducers:**
- Receives `ToolStateEvent` (normalized) instead of raw `ToolCallEvent`
- Receives `acc` (framework-provided `StreamingInput`) instead of manually accumulating
- No per-tool accumulation logic — just semantic state transitions
- `TAccFields` and `TAccChildren` declare minimum required streaming input fields (structural subtyping)

#### Normalized Event Contract

```typescript
// packages/tools/src/tool-state-event.ts

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
  | { type: 'error'; error: string }
  | { type: 'rejected'; reason?: string }
  | { type: 'interrupted' }

  // Post-execution (background processes)
  | { type: 'processOutput'; text: string; stream: 'stdout' | 'stderr' }
  | { type: 'processExited'; exitCode: number }
  | { type: 'processPromoted' }
  | { type: 'processDemoted'; logPath: string };
```

#### State Model Instances

**`defaultModel`** — Phase-only tracking. Used by most tools.

```typescript
// packages/agent/src/models/default.ts

type Phase = 'streaming' | 'executing' | 'completed' | 'error' | 'rejected' | 'interrupted';

interface DefaultState {
  phase: Phase;
}

const defaultModel = defineStateModel<DefaultState>({
  initial: { phase: 'streaming' },
  reduce: (state, event) => {
    switch (event.type) {
      case 'started':          return { phase: 'streaming' };
      case 'executionStarted': return { phase: 'executing' };
      case 'completed':        return { phase: 'completed' };
      case 'error':            return { phase: 'error' };
      case 'rejected':         return { phase: 'rejected' };
      case 'interrupted':      return { phase: 'interrupted' };
      default:                 return state;
    }
  },
});
```

**Tools using `defaultModel`:** `fs.read`, `fs.tree`, `fs.search`, `fs.view`, `web-search`, `web-fetch`, all browser tools, `agent-create`, `agent-kill`, `think`, `skill`, `agent-message`, `parent-message`

**`diffModel`** — Phase + diffs from emissions.

```typescript
// packages/agent/src/models/diff.ts

interface DiffState {
  phase: Phase;
  diffs: Diff[];
  path?: string;
}

const diffModel = defineStateModel<
  DiffState,
  { path?: string },                                    // TAccFields
  { old?: ChildAcc[]; new?: ChildAcc[] },               // TAccChildren
  { type: 'edit_diff'; path: string; diffs: Diff[] }    // TEmission
>({
  initial: { phase: 'streaming', diffs: [] },
  reduce: (state, event, acc) => {
    switch (event.type) {
      case 'started':          return { ...state, phase: 'streaming' };
      case 'inputUpdated':     return { ...state, path: acc.fields?.path };
      case 'executionStarted': return { ...state, phase: 'executing' };
      case 'emission':         return { ...state, diffs: event.value.diffs, path: event.value.path };
      case 'completed':        return { ...state, phase: 'completed' };
      case 'error':            return { ...state, phase: 'error' };
      case 'rejected':         return { ...state, phase: 'rejected' };
      case 'interrupted':      return { ...state, phase: 'interrupted' };
      default:                 return state;
    }
  },
});
```

**Tools using `diffModel`:** `edit`

**`contentModel`** — Phase + streaming content stats.

```typescript
// packages/agent/src/models/content.ts

interface ContentState {
  phase: Phase;
  path?: string;
  charCount: number;
  lineCount: number;
}

const contentModel = defineStateModel<
  ContentState,
  { path?: string },    // TAccFields
  {},                    // TAccChildren
  { type: 'write_stats'; path: string; linesWritten: number }  // TEmission
>({
  initial: { phase: 'streaming', charCount: 0, lineCount: 0 },
  reduce: (state, event, acc) => {
    switch (event.type) {
      case 'inputUpdated': {
        // Derive counts from accumulated body
        const body = acc.body ?? '';
        return {
          ...state,
          path: acc.fields?.path,
          charCount: body.length,
          lineCount: body.split('\n').length,
        };
      }
      case 'executionStarted': return { ...state, phase: 'executing' };
      case 'emission':         return { ...state, lineCount: event.value.linesWritten, path: event.value.path };
      case 'completed':        return { ...state, phase: 'completed' };
      case 'error':            return { ...state, phase: 'error' };
      case 'interrupted':      return { ...state, phase: 'interrupted' };
      default:                 return state;
    }
  },
});
```

**Tools using `contentModel`:** `fs.write`

**`shellModel`** — Phase + command + output + background lifecycle.

```typescript
// packages/agent/src/models/shell.ts

type ShellDone =
  | { kind: 'success'; stdout: string; stderr: string; exitCode: number }
  | { kind: 'detached'; pid: number; stdout: string; stderr: string }
  | { kind: 'error'; message: string }
  | { kind: 'rejected'; systemReason: string | null }
  | { kind: 'interrupted' };

interface ShellState {
  phase: Phase;
  command: string;
  done: ShellDone | null;
  // Background process state (post-execution)
  bgOutput: string;
  bgExited: boolean;
  bgExitCode?: number;
}

const shellModel = defineStateModel<ShellState, {}, {}, never>({
  initial: { phase: 'streaming', command: '', done: null, bgOutput: '', bgExited: false },
  reduce: (state, event, acc) => {
    switch (event.type) {
      case 'inputUpdated':
        return { ...state, command: acc.body ?? '' };
      case 'inputReady':
        return { ...state, command: (event.input as any)?.command ?? state.command };
      case 'executionStarted':
        return { ...state, phase: 'executing' };
      case 'completed':
        return { ...state, phase: 'completed', done: resolveShellResult(event.output) };
      case 'error':
        return { ...state, phase: 'error', done: { kind: 'error', message: event.error } };
      case 'rejected':
        return { ...state, phase: 'rejected', done: { kind: 'rejected', systemReason: event.reason ?? null } };
      case 'interrupted':
        return { ...state, phase: 'interrupted', done: { kind: 'interrupted' } };
      // Background process events
      case 'processOutput':
        return { ...state, bgOutput: state.bgOutput + event.text };
      case 'processExited':
        return { ...state, bgExited: true, bgExitCode: event.exitCode };
      case 'processPromoted':
        return state; // no visual change
      default:
        return state;
    }
  },
});
```

**Tools using `shellModel`:** `shell`, `shell-bg`

### 4. `defineDisplay`

```typescript
// packages/tools/src/display.ts

interface DisplayConfig<TState, TAccFields, TAccChildren, TRender> {
  render: (props: DisplayProps<TState, TAccFields, TAccChildren>) => TRender;
  liveText: (props: { state: TState; acc: StreamingInput<TAccFields, TAccChildren> }) => string;
}

interface DisplayProps<TState, TAccFields, TAccChildren> {
  state: TState;
  acc: StreamingInput<TAccFields, TAccChildren>;
  label: string;
  result?: ToolResult;
  isExpanded: boolean;
  onToggle: () => void;
  onFileClick?: (path: string, section?: string) => void;
}

function defineDisplay<TState, TAccFields, TAccChildren, TRender>(
  model: StateModel<TState, TAccFields, TAccChildren, any>,
  config: DisplayConfig<TState, TAccFields, TAccChildren, TRender>
): Display<TState, TAccFields, TAccChildren, TRender>;
```

**`TRender` is generic** — not React-specific. CLI provides `ReactNode`, web could provide something else.

#### Display Instances

Defined in `cli/src/visuals/`. Each display receives typed state + acc instead of casting `unknown`.

**`defaultDisplay`** — Simple phase-based rendering.
- Streaming: shimmer + label
- Executing: shimmer + label  
- Completed: label + success indicator
- Error: label + error indicator
- Used by all tools with `defaultModel`

**`diffDisplay`** — Streaming diff + final hunks.
- Streaming: reads `acc.children.old[0].body` and `acc.children.new[0].body` for live diff preview
- Completed: renders `state.diffs` as `DiffHunk` components
- Reads `acc.fields.path` for file path display

**`contentDisplay`** — Streaming write preview.
- Streaming: shows `acc.body` content preview with char/line counters from `state`
- Completed: shows path + final line count

**`shellDisplay`** — Command + output + lifecycle.
- Streaming: shows `state.command` as it streams
- Executing: shimmer + command
- Completed: expandable output from `state.done`
- Detached: shows PID, expandable initial output
- Background events: shows accumulated `state.bgOutput`

### 5. `createBinding`

```typescript
// packages/tools/src/binding.ts

function createBinding<TState, TAccFields, TAccChildren, TEmission, TRender>(
  model: StateModel<TState, TAccFields, TAccChildren, TEmission>,
  display: Display<TState, TAccFields, TAccChildren, TRender>,
): ToolDisplayBinding<TRender> {
  // Type safety verified here — TState, TAccFields, TAccChildren must match
  // Types erased after this point
  return {
    createCallState: () => ({ state: model.initial }),
    reduce: (callState, event, acc) => {
      callState.state = model.reduce(callState.state, event, acc);
    },
    render: (callState, props) => display.render({ state: callState.state, ...props }),
    liveText: (callState, acc) => display.liveText({ state: callState.state, acc }),
  };
}
```

### 6. `ToolEventAdapter` Interface

```typescript
// packages/tools/src/adapter.ts

interface ToolEventAdapterInterface {
  /** Handle a raw xml-act tool call event */
  handleToolEvent(toolKey: string, callId: string, event: ToolCallEvent): void;
  
  /** Handle a background process event, correlated by pid */
  handleProcessEvent(pid: number, event: ProcessEvent): void;
  
  /** Handle an approval lifecycle event */
  handleApprovalEvent(callId: string, event: ApprovalEvent): void;
  
  /** Get current state for a tool call (for rendering) */
  getCallState(callId: string): ToolCallState | undefined;
  
  /** Get all active call states */
  getActiveCalls(): Map<string, ToolCallState>;
}

interface ToolCallState {
  toolKey: string;
  binding: ToolDisplayBinding;
  modelState: unknown;        // opaque — only the binding can read it
  acc: StreamingInput;        // current accumulated input
  label: string;
  result?: ToolResult;
}

type ProcessEvent =
  | { type: 'output'; text: string; stream: 'stdout' | 'stderr' }
  | { type: 'exited'; exitCode: number }
  | { type: 'promoted' }
  | { type: 'demoted'; logPath: string };

type ApprovalEvent =
  | { type: 'requested'; preview?: unknown }
  | { type: 'granted' }
  | { type: 'rejected' };
```

### 7. `DisplayBindingRegistry`

```typescript
// packages/tools/src/registry.ts

interface DisplayBindingRegistry<TRender = unknown> {
  get(toolKey: string): ToolDisplayBinding<TRender> | undefined;
  getDefault(): ToolDisplayBinding<TRender>;
}
```

---

## ToolEventAdapter Implementation

```typescript
// packages/agent/src/adapter/tool-event-adapter.ts

class ToolEventAdapter implements ToolEventAdapterInterface {
  private calls = new Map<string, CallEntry>();
  private pidToCallId = new Map<number, string>();
  
  constructor(
    private displayRegistry: DisplayBindingRegistry,
    private xmlBindings: Map<string, XmlBinding>,  // for SchemaAccumulator creation
    private toolDefs: Map<string, ToolDefinition>,  // for label function
  ) {}
  
  handleToolEvent(toolKey: string, callId: string, event: ToolCallEvent): void {
    switch (event._tag) {
      case 'ToolInputStarted': {
        const binding = this.displayRegistry.get(toolKey) ?? this.displayRegistry.getDefault();
        const xmlBinding = this.xmlBindings.get(toolKey);
        const accumulator = xmlBinding ? new SchemaAccumulator(xmlBinding.config) : new SchemaAccumulator({});
        const call: CallEntry = {
          toolKey,
          binding,
          callState: binding.createCallState(),
          accumulator,
          label: this.toolDefs.get(toolKey)?.label({}) ?? `${toolKey}()`,
          result: undefined,
        };
        this.calls.set(callId, call);
        accumulator.ingest(event);
        binding.reduce(call.callState, { type: 'started' }, accumulator.current);
        break;
      }
      
      case 'ToolInputFieldValue':
      case 'ToolInputBodyChunk':
      case 'ToolInputChildStarted':
      case 'ToolInputChildComplete': {
        const call = this.calls.get(callId);
        if (!call) return;
        call.accumulator.ingest(event);
        // Update label from accumulated fields
        const toolDef = this.toolDefs.get(call.toolKey);
        if (toolDef) call.label = toolDef.label(call.accumulator.current.fields as any);
        const normalized: ToolStateEvent = {
          type: 'inputUpdated',
          changed: event._tag === 'ToolInputFieldValue' ? 'field' 
                 : event._tag === 'ToolInputBodyChunk' ? 'body' : 'child',
          name: event._tag === 'ToolInputFieldValue' ? event.field : undefined,
        };
        call.binding.reduce(call.callState, normalized, call.accumulator.current);
        break;
      }
      
      case 'ToolInputReady': {
        const call = this.calls.get(callId);
        if (!call) return;
        call.accumulator.ingest(event);
        const toolDef = this.toolDefs.get(call.toolKey);
        if (toolDef) call.label = toolDef.label(event.input as any);
        call.binding.reduce(call.callState, { type: 'inputReady', input: event.input }, call.accumulator.current);
        break;
      }
      
      case 'ToolInputParseError': {
        const call = this.calls.get(callId);
        if (!call) return;
        call.binding.reduce(call.callState, { type: 'parseError', error: event.detail }, call.accumulator.current);
        break;
      }
      
      case 'ToolExecutionStarted': {
        const call = this.calls.get(callId);
        if (!call) return;
        call.binding.reduce(call.callState, { type: 'executionStarted' }, call.accumulator.current);
        break;
      }
      
      case 'ToolExecutionEnded': {
        const call = this.calls.get(callId);
        if (!call) return;
        const normalized = this.mapExecutionResult(event.result, event.display);
        call.result = mapToolResult(event.result, event.display);
        call.binding.reduce(call.callState, normalized, call.accumulator.current);
        // Track pid for background process correlation
        if (event.result._tag === 'Success' && event.result.output?.pid) {
          this.pidToCallId.set(event.result.output.pid, callId);
        }
        break;
      }
      
      case 'ToolObservation':
        // Ignored — not relevant to display state
        break;
    }
  }
  
  handleProcessEvent(pid: number, event: ProcessEvent): void {
    const callId = this.pidToCallId.get(pid);
    if (!callId) return;
    const call = this.calls.get(callId);
    if (!call) return;
    
    const normalized: ToolStateEvent = (() => {
      switch (event.type) {
        case 'output':   return { type: 'processOutput', text: event.text, stream: event.stream };
        case 'exited':   return { type: 'processExited', exitCode: event.exitCode };
        case 'promoted': return { type: 'processPromoted' };
        case 'demoted':  return { type: 'processDemoted', logPath: event.logPath };
      }
    })();
    
    call.binding.reduce(call.callState, normalized, call.accumulator.current);
  }
  
  handleApprovalEvent(callId: string, event: ApprovalEvent): void {
    const call = this.calls.get(callId);
    if (!call) return;
    
    const normalized: ToolStateEvent = (() => {
      switch (event.type) {
        case 'requested': return { type: 'awaitingApproval', preview: event.preview };
        case 'granted':   return { type: 'approvalGranted' };
        case 'rejected':  return { type: 'approvalRejected' };
      }
    })();
    
    call.binding.reduce(call.callState, normalized, call.accumulator.current);
  }
  
  getCallState(callId: string): ToolCallState | undefined {
    const call = this.calls.get(callId);
    if (!call) return undefined;
    return {
      toolKey: call.toolKey,
      binding: call.binding,
      modelState: call.callState.state,
      acc: call.accumulator.current,
      label: call.label,
      result: call.result,
    };
  }
  
  private mapExecutionResult(result: XmlToolResult, display?: ToolDisplay): ToolStateEvent {
    switch (result._tag) {
      case 'Success':
        // If there's a display emission, emit that first, then completed
        // (In practice, the adapter would emit both events)
        return { type: 'completed', output: result.output };
      case 'Error':
        return { type: 'error', error: result.error };
      case 'Rejected':
        return { type: 'rejected', reason: extractRejectionReason(result.rejection) };
      case 'Interrupted':
        return { type: 'interrupted' };
    }
  }
}
```

---

## Emission Channel

### Current: `ToolEmitTag` (Ref side-channel)

Tools call `yield* (yield* ToolEmitTag).emit(value)`. Execution manager backs this with a `Ref<ToolDisplay | undefined>`, reads it on `ToolExecutionEnded`, and attaches to the forwarded event.

### New: `ToolContext.emit()`

```typescript
// In execution manager, when providing tool context:

const emissionQueue: TEmission[] = [];

const ctx: ToolContext<TEmission> = {
  emit: (value) => Effect.sync(() => { emissionQueue.push(value); }),
};

// After tool execution completes:
// Each emission becomes a ToolStateEvent { type: 'emission', value }
// forwarded through the adapter before the 'completed' event
```

The execution manager:
1. Creates `ToolContext` with emission queue before each tool execution
2. Runs `tool.execute(input, ctx)`
3. After execution, drains emission queue
4. For each emission, emits a `ToolCallEvent` variant (new: `ToolEmission`) that flows through the normal event pipeline
5. The adapter normalizes `ToolEmission` → `ToolStateEvent { type: 'emission', value }`

**New `ToolCallEvent` variant needed in xml-act:**

```typescript
// Added to ToolCallEvent union in xml-act types
| { _tag: 'ToolEmission'; toolCallId: string; value: unknown }
```

This is emitted by the execution manager (not the xml-act runtime itself) between `ToolExecutionStarted` and `ToolExecutionEnded`, so it flows through the same event pipeline.

---

## Tool Porting Guide

### Current → New for each tool

#### `fs.read`

**Current:**
```typescript
const readTool = createTool({
  name: 'read', group: 'fs',
  inputSchema: S.Struct({ path: S.String, offset: S.optional(S.Number), limit: S.optional(S.Number) }),
  outputSchema: S.String,
  bindings: { xmlInput: { type: 'tag', attributes: [...], }, xmlOutput: { type: 'tag' } },
  execute: (input) => Effect.gen(function*() { ... }),
});
```

**New:**
```typescript
// packages/agent/src/tools/fs.ts
const readTool = defineTool({
  name: 'read', group: 'fs',
  inputSchema: S.Struct({ path: S.String, offset: S.optional(S.Number), limit: S.optional(S.Number) }),
  outputSchema: S.String,
  execute: (input, ctx) => Effect.gen(function*() {
    // Same logic, no ToolEmitTag needed (read doesn't emit)
    const { cwd, workspacePath } = yield* WorkingDirectoryTag;
    // ...
  }),
  label: (input) => input.path ? `Reading ${input.path}` : 'Reading file…',
});

// packages/agent/src/tools/fs.ts (co-located)
const readXmlBinding = defineXmlBinding(readTool, {
  input: {
    attributes: [
      { attr: 'path', field: 'path' },
      { attr: 'offset', field: 'offset' },
      { attr: 'limit', field: 'limit' },
    ],
  },
  output: {},
} as const);
```

#### `edit`

**Current:** Uses `ToolEmitTag` to emit `{ type: 'edit_diff', path, diffs }`.

**New:**
```typescript
const editTool = defineTool({
  name: 'edit',
  inputSchema: S.Struct({
    path: S.String,
    oldString: S.String,
    newString: S.String,
    replaceAll: S.optional(S.Boolean),
  }),
  outputSchema: S.String,
  emissionSchema: S.Struct({ type: S.Literal('edit_diff'), path: S.String, diffs: S.Array(DiffSchema) }),
  execute: (input, ctx) => Effect.gen(function*() {
    const { cwd, workspacePath } = yield* WorkingDirectoryTag;
    // ... same logic ...
    yield* ctx.emit({ type: 'edit_diff', path: input.path, diffs });
    return summary;
  }),
  label: (input) => input.path ? `Editing ${input.path}` : 'Editing file…',
});

const editXmlBinding = defineXmlBinding(editTool, {
  input: {
    attributes: [
      { attr: 'path', field: 'path' },
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

#### `shell`

**Current:** Uses `ToolReminderTag` for detach guidance. No `ToolEmitTag`.

**New:**
```typescript
const shellTool = defineTool({
  name: 'shell',
  inputSchema: S.Struct({
    command: S.String,
    timeout: S.optional(S.Number),
    background: S.optional(S.Boolean),
  }),
  outputSchema: ShellOutputSchema,
  // No emissionSchema — shell doesn't emit display data
  execute: (input, ctx) => Effect.gen(function*() {
    // Same logic — ToolReminderTag stays as Effect service
    // ...
  }),
  label: (input) => input.command ? `$ ${shortenCommand(input.command)}` : 'Running command…',
});

const shellXmlBinding = defineXmlBinding(shellTool, {
  input: {
    attributes: [
      { attr: 'timeout', field: 'timeout' },
      { attr: 'background', field: 'background' },
    ],
    body: 'command',
  },
  output: {
    childTags: ['mode', 'reason', 'pid', 'stdout', 'stderr', 'exitCode'],
  },
} as const);
```

---

## DisplayProjection Changes

The `tool_event` handler in `DisplayProjection` is replaced by delegation to `ToolEventAdapter`.

**Current:** DisplayProjection directly handles every `ToolCallEvent` variant, manages visual state via reducer registry singleton, constructs `ToolStep` entries.

**New:** DisplayProjection:
1. On `tool_event`, delegates to `this.adapter.handleToolEvent(toolKey, callId, event)`
2. On `background_process_*`, delegates to `this.adapter.handleProcessEvent(pid, event)`
3. Reads `this.adapter.getCallState(callId)` to construct/update `ToolStep` entries
4. The adapter owns all state model reduction and input accumulation
5. DisplayProjection only handles step placement in think blocks, visibility policy, and message ordering

**What stays in DisplayProjection:**
- Think block lifecycle (create, close, interrupt finalization)
- Message ordering (queued user messages, insertion order)
- Tool visibility policy (`isToolHidden` based on agent role)
- Tool activity signals (`forkToolStep` emission)
- Non-tool display concerns (messages, thinking, subagent activity, etc.)

**What moves to adapter:**
- All `ToolCallEvent` → visual state reduction
- Input accumulation
- Label generation
- Result mapping
- Background process → tool call correlation

---

## Registration & Wiring

### Agent-side: Tool + Binding Registration

```typescript
// packages/agent/src/tools/index.ts

function buildRegisteredTools(
  agentDef: AgentDefinition,
  layers: Layer,
  xmlBindings: Map<string, XmlBinding>,  // new: separate binding registry
): Map<string, RegisteredTool> {
  const tools = new Map<string, RegisteredTool>();
  
  for (const [defKey, tool] of Object.entries(agentDef.tools)) {
    const tagName = defaultXmlTagName(tool);
    const xmlBinding = xmlBindings.get(tagName);
    if (!xmlBinding) continue;
    
    tools.set(tagName, {
      tool,
      tagName,
      groupName: tool.group,
      binding: xmlBinding.toXmlTagBinding(),  // extract runtime binding for xml-act
      meta: { defKey },
      layerProvider: () => Effect.succeed(layers),
    });
  }
  
  return tools;
}
```

### CLI-side: Composition Root

```typescript
// cli/src/visuals/registry.ts

import { defaultModel, diffModel, contentModel, shellModel } from '@magnitudedev/agent/models';
import { createBinding } from '@magnitudedev/tools';
import { defaultDisplay, diffDisplay, contentDisplay, shellDisplay } from './displays';

// Type-safe pairing — compile error if TState doesn't match
const bindings = {
  default: createBinding(defaultModel, defaultDisplay),
  diff:    createBinding(diffModel, diffDisplay),
  content: createBinding(contentModel, contentDisplay),
  shell:   createBinding(shellModel, shellDisplay),
};

// Tool key → binding mapping
export const displayBindingRegistry: DisplayBindingRegistry = {
  get(toolKey: string) {
    switch (toolKey) {
      case 'fileEdit':  return bindings.diff;
      case 'fileWrite': return bindings.content;
      case 'shell':     return bindings.shell;
      default:          return bindings.default;
    }
  },
  getDefault() { return bindings.default; },
};
```

---

## What Gets Deleted

| File/Module | Reason |
|---|---|
| `packages/agent/src/visuals/` (entire directory) | Replaced by state models in `agent/src/models/` |
| `packages/agent/src/visuals/registry.ts` | Replaced by `DisplayBindingRegistry` |
| `packages/agent/src/execution/tool-emit.ts` (`ToolEmitTag`) | Replaced by `ToolContext.emit()` |
| `ToolDisplay` union in `events.ts` | Replaced by per-tool emission schemas |
| `setVisualRegistry` / `getVisualRegistry` singleton | Replaced by explicit `DisplayBindingRegistry` passed to adapter |
| `cli/src/visuals/registry.ts` (old reducer+render registries) | Replaced by composition root with `createBinding` |
| Per-reducer accumulation logic in all reducers | Replaced by `SchemaAccumulator` |
| `mapXmlToolResultForDisplay` / `mapXmlToolResult` duplication | Single `mapToolResult` in adapter |
| `visualState: unknown` on `ToolStep` | Replaced by typed state through opaque binding |

---

## What Gets Added

| New | Package | Purpose |
|---|---|---|
| `defineTool` | `tools` | Tool function factory (replaces `createTool`) |
| `defineStateModel` | `tools` | State model factory |
| `defineDisplay` | `tools` | Display renderer factory |
| `createBinding` | `tools` | Type-safe model+display pairing |
| `ToolStateEvent` | `tools` | Normalized event union |
| `StreamingInput` | `tools` | Accumulated input snapshot type |
| `ToolEventAdapter` interface | `tools` | Adapter contract |
| `DisplayBindingRegistry` interface | `tools` | Registry contract |
| `DisplayProps` | `tools` | Display prop contract |
| `ToolContext` | `tools` | Execution context with `emit()` |
| `defineXmlBinding` | `xml-act` | XML binding factory |
| `SchemaAccumulator` | `xml-act` | Generic input accumulator |
| `DeriveStreamingShape` | `xml-act` | Type-level shape derivation |
| `ToolEmission` event variant | `xml-act` | New `ToolCallEvent` variant |
| `ToolEventAdapter` implementation | `agent` | Concrete adapter |
| `defaultModel`, `diffModel`, `contentModel`, `shellModel` | `agent` | State model instances |
| `defaultDisplay`, `diffDisplay`, `contentDisplay`, `shellDisplay` | `cli` | Display renderer instances |
| Composition root | `cli` | `createBinding` wiring |

---

## File Structure

### `packages/tools/src/`
```
tool.ts                 # defineTool, ToolDefinition, ToolContext, ToolConfig
state-model.ts          # defineStateModel, StateModel
tool-state-event.ts     # ToolStateEvent union
streaming-input.ts      # StreamingInput type
display.ts              # defineDisplay, Display, DisplayProps, createBinding, ToolDisplayBinding
adapter.ts              # ToolEventAdapter interface, ToolCallState, ProcessEvent, ApprovalEvent
registry.ts             # DisplayBindingRegistry interface
index.ts                # barrel exports
```

### `packages/xml-act/src/`
```
xml-binding.ts          # defineXmlBinding, XmlBinding, XmlMappingConfig (NEW)
schema-accumulator.ts   # SchemaAccumulator (NEW)
type-chain.ts           # DeriveStreamingShape, AttrNames, ChildTagNames (NEW)
types.ts                # ToolCallEvent gains ToolEmission variant (MODIFIED)
execution/
  tool-dispatcher.ts    # Updated to handle ToolContext/emission queue (MODIFIED)
  xml-runtime.ts        # Unchanged structurally
  input-builder.ts      # Unchanged
```

### `packages/agent/src/`
```
tools/
  fs.ts                 # defineTool + defineXmlBinding calls (REWRITTEN)
  shell.ts              # defineTool + defineXmlBinding calls (REWRITTEN)
  web-search-tool.ts    # defineTool + defineXmlBinding calls (REWRITTEN)
  agent-tools.ts        # defineTool + defineXmlBinding calls (REWRITTEN)
  globals.ts            # defineTool + defineXmlBinding calls (REWRITTEN)
  skill.ts              # defineTool + defineXmlBinding calls (REWRITTEN)
  index.ts              # buildRegisteredTools updated (MODIFIED)

models/                 # NEW directory
  default.ts            # defaultModel
  diff.ts               # diffModel
  content.ts            # contentModel
  shell.ts              # shellModel
  index.ts              # barrel

adapter/                # NEW directory
  tool-event-adapter.ts # ToolEventAdapter implementation
  event-mapping.ts      # ToolCallEvent → ToolStateEvent helpers

visuals/                # DELETED entirely

execution/
  tool-emit.ts          # DELETED (ToolEmitTag)
  execution-manager.ts  # Updated: ToolContext wiring, emission queue (MODIFIED)

projections/
  display.ts            # Updated: delegates tool events to adapter (MODIFIED)
```

### `cli/src/visuals/`
```
displays/               # NEW directory
  default-display.tsx   # defineDisplay(defaultModel, ...)
  diff-display.tsx      # defineDisplay(diffModel, ...)
  content-display.tsx   # defineDisplay(contentModel, ...)
  shell-display.tsx     # defineDisplay(shellModel, ...)
  index.ts

registry.ts             # Composition root: createBinding calls + DisplayBindingRegistry (REWRITTEN)
define.ts               # DELETED (old contracts)
fs.tsx                  # DELETED (absorbed into displays/)
shell.tsx               # DELETED (absorbed into displays/)
tools.tsx               # DELETED (absorbed into displays/)
```
