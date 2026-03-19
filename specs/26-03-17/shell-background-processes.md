# Implementation plan: shell auto-detach background process support

## Recommended approach

Implement a **session-scoped background process registry** inside `packages/agent`, wire it into the existing event bus, and expose process activity to agents via a **projection-backed observable** — the same pattern used by `agentsStatusObservable`.

Key design principles:

1. `shell` runs via `spawn(shell, ['-c', command])`, not `exec`
2. it waits **~5s**
3. if still running, it **registers** the process in a new registry and returns a **detached shell result** that clearly communicates the PID and what's happening
4. the registry publishes **background process events** (`background_process_output`, `background_process_exited`) as output arrives and when the process exits
5. a **new projection** (`BackgroundProcessesProjection`) consumes these events and maintains per-fork process state
6. a **new observable** (`backgroundProcessesObservable`) reads the projection at each natural turn start, surfacing organized per-process state in the agent's context
7. **no wake events for output** — the agent sees process updates at natural turn boundaries only
8. **process exit** triggers a new turn by having `WorkingState` directly handle `background_process_exited` (setting `willContinue = true` for the owning fork, like `user_message` does)
9. process control stays shell-native (`kill <pid>`, etc.); Magnitude only tracks/observes known PIDs

Data flow:
```
process registry → background_process_* events → BackgroundProcessesProjection → backgroundProcessesObservable → observations_captured → memory → LLM context
```

---

## Major design decisions and trade-offs

### 1) Tool result shape: extend `shell` output instead of inventing a new tool
**Chosen**
- Keep tool name `shell`
- Broaden `packages/agent/src/tools/shell.ts` output schema from fixed `{ stdout, stderr, exitCode }` to a discriminated union:
  - `completed`
  - `detached`

Suggested shape:
```ts
type ShellOutput =
  | {
      mode: 'completed'
      stdout: string
      stderr: string
      exitCode: number
    }
  | {
      mode: 'detached'
      pid: number
      stdout: string
      stderr: string
    }
```

The detached result is pure data — PID and initial output captured during the 5s window. No explanatory notes; the agent knows how background processes work from the shell tool docs.

Why:
- preserves one-call ergonomics
- agent never has to predict fast vs slow
- keeps success/error semantics unchanged: detached is still a successful tool result

Trade-off:
- requires updating XML output bindings and shell visual code to handle union output.

### 2) Process events should be **AppEvents**, not overloaded `tool_event`
**Chosen**
Add dedicated app events in `packages/agent/src/events.ts` for lifecycle/output:
- `background_process_registered`
- `background_process_output`
- `background_process_exited`

Why:
- these occur **after** the tool call/turn ends
- they are not xml-act `ToolCallEvent`s
- UI/projections can subscribe without pretending they belong to the original tool runtime

Trade-off:
- slightly wider event surface, but much cleaner than abusing `tool_event`.

### 3) Observable-based context delivery, no wake for output
**Chosen**
- Background process state is exposed via a **projection-backed observable** (same pattern as `agentsStatusObservable`)
- Process output updates do **not** trigger turns — the agent sees accumulated output at natural turn starts only
- Process **exit** triggers a turn by having `WorkingState` directly handle `background_process_exited` (setting `willContinue = true` for the owning fork)

Why:
- follows the established observable pattern exactly
- avoids noisy wake-triggered turn spam from chatty processes
- process exit is the one event that genuinely requires agent attention
- clean separation: observable = how agent sees data, WorkingState = when agent gets a turn

Trade-off:
- if a process produces important output but doesn't exit, the agent won't see it until its next natural turn. This is acceptable because the agent is always doing other work that triggers turns anyway.

### 4) Track processes by PID, but registry owns the richer record
**Chosen**
Internally track:
- PID
- owning `forkId`
- spawning `turnId`
- original command
- `ChildProcess` handle
- ring buffers for stdout/stderr
- sequence counters and truncation metadata
- exit state

Why:
- agent uses only PID externally
- registry can detect exit/kills and emit clean lifecycle events

Trade-off:
- PID reuse exists in theory; mitigate by only acting on live `ChildProcess` handles and by storing start time.

### 5) No special classifier behavior for `kill <pid>` initially
**Chosen**
Leave `kill` classified as current `normal` behavior unless we discover a regression.

Why:
- desired design explicitly says PID management should use ordinary shell commands
- permission model already permits `normal` shell depending on agent policy

Potential follow-up:
- optionally harden classifier around daemonization syntaxes (`nohup`, `disown`, shell `&`) later, but this redesign makes raw shell backgrounding unnecessary.

---

## Output format design

This section defines exactly what the agent sees in two key moments: (1) when a shell command is detached, and (2) on each subsequent turn via the observable.

### Shell tool result on detach

When a command exceeds 5s and is detached, the tool result XML is pure data:

```xml
<shell>
  <mode>detached</mode>
  <pid>48291</pid>
  <stdout>
... initial 5s of stdout output ...
  </stdout>
  <stderr>
... initial 5s of stderr output ...
  </stderr>
</shell>
```

Key principles:
- The PID is prominent and clearly labeled
- The initial captured output is included so the agent has context
- No explanatory notes — the agent knows how detached processes work from the tool docs

### Observable output format (system inbox each turn)

The observable produces text that appears in the agent's `<system>` block. It must be:
- **Organized by process** when multiple are running
- **Explicit about new vs no-new output**
- **Clear about process status**

#### Running processes with new output:
```
<background_processes>
<process pid="48291" status="running" command="npm run dev">
<new_stdout>
Server started on port 3000
Compiled successfully in 2.3s
</new_stdout>
<new_stderr>
(no new output)
</new_stderr>
</process>
</background_processes>
```

#### Running process with NO new output:
```
<background_processes>
<process pid="48291" status="running" command="npm run dev">
(no new output since last turn)
</process>
</background_processes>
```

#### Process that just exited:
```
<background_processes>
<process pid="48291" status="exited" command="npm run dev" exitCode="0">
<final_stdout>
... last output before exit ...
</final_stdout>
<final_stderr>
(no output)
</final_stderr>
</process>
</background_processes>
```

#### Process that was killed:
```
<background_processes>
<process pid="48291" status="killed" command="npm run dev" signal="SIGTERM">
<final_stdout>
... last output before kill ...
</final_stdout>
</process>
```

#### Multiple processes:
```
<background_processes>
<process pid="48291" status="running" command="npm run dev">
<new_stdout>
Compiled successfully
</new_stdout>
</process>
<process pid="48305" status="running" command="npm test --watch">
(no new output since last turn)
</process>
</background_processes>
```

#### No background processes:
When there are no active or recently-exited background processes, the observable returns `[]` (no observation parts) — nothing appears in the system inbox.

### Output rate demotion: inline vs file mode

Processes start in **inline mode** where output is shown directly in the observable. If a process produces too much output in a single inter-turn window (threshold: ~8KB), it gets **demoted to file mode** permanently for that process.

**Inline mode** (default) — quiet dev server, ~5 lines since last turn:
```
<process pid="48291" status="running" command="npm run dev">
<new_stdout>
[12:03:41] Compiled successfully in 1.2s
[12:03:45] GET /api/users 200 12ms
[12:03:46] GET /api/posts 200 8ms
</new_stdout>
<new_stderr>
(no new output)
</new_stderr>
</process>
```

**File mode** (demoted) — noisy test suite, 4000+ lines since last turn:
```
<process pid="48305" status="running" command="npm test">
<stdout mode="file" newLines="4291" totalLines="12847" file="~/.magnitude/tmp/48305-stdout.log">
  Tests:       187 passed, 7 failed, 194 total
</stdout>
<stderr mode="file" newLines="23" file="~/.magnitude/tmp/48305-stderr.log">
FAIL src/api/auth.test.ts
  ● login > should reject invalid credentials
</stderr>
</process>
```

**Both modes coexisting** — two processes, one quiet, one noisy:
```
<background_processes>
<process pid="48291" status="running" command="npm run dev">
<new_stdout>
[12:03:53] Compiled successfully in 0.9s
</new_stdout>
</process>
<process pid="48305" status="running" command="npm test">
<stdout mode="file" newLines="4291" totalLines="12847" file="~/.magnitude/tmp/48305-stdout.log">
  Tests:       187 passed, 7 failed, 194 total
</stdout>
</process>
</background_processes>
```

**Demotion rules:**
- Threshold: ~8KB of output accumulated between turns triggers demotion
- Once demoted, stays demoted — no bouncing back
- File mode output is written to `~/.magnitude/tmp/{pid}-stdout.log` and `~/.magnitude/tmp/{pid}-stderr.log`
- File mode observable shows: `mode="file"` attribute, `newLines`/`totalLines` counts, file path, and tail (~last few lines)
- Agent can `fs-read` the file if it needs full output
- Files are cleaned up when the process exits and the agent has seen the exit notification

### Design principles for the format:
1. **Always show PID and command** — the agent needs to know which process is which. Truncate command to ~80 chars with ellipsis if longer.
2. **Explicit "no new output"** — silence should be stated, not implied by absence
3. **Separate stdout and stderr** — agent needs to distinguish between them
4. **Show exit info clearly** — exitCode for normal exits, signal for kills
5. **Recently exited processes stay visible for one turn** — so the agent can react to the exit, then they're cleared
6. **Rate-based demotion** — noisy processes get demoted to file output to protect context size

---

## Tool reminder system

A new `ToolReminderTag` Effect service allows tools to emit contextual reminders that appear in the agent's system inbox wrapped in `<reminder>` tags. This is used by the shell tool to explain detached processes on first encounter.

### New service: `ToolReminderTag`
Same Ref-based pattern as `ToolEmitTag`:
- `packages/agent/src/execution/tool-reminder.ts` (new)
- Provides `{ add(text: string): void }`
- Backed by a `Ref<string[]>` that accumulates reminders during tool execution
- Execution manager resets it before each tool call, reads it after

### Execution manager wiring
In `execution-manager.ts`:
- Create a `toolReminderRef` alongside `toolEmitRef`
- Provide `ToolReminderTag` in fork layers via `makeForkLayers`
- After tool execution ends, read collected reminders
- On `turn_completed`, push each collected reminder as `{ kind: 'reminder', text }` into system entries

### Formatting change
In `packages/agent/src/prompts/agents.ts`, change the reminder rendering in `formatSystemInbox`:
```ts
// Before:
} else if (entry.kind === 'reminder') {
  push(`${entry.text}\n`)
}

// After:
} else if (entry.kind === 'reminder') {
  push(`<reminder>${entry.text}</reminder>\n`)
}
```

### Shell tool usage
In `shell.ts`, when returning a detached result:
```ts
const reminder = yield* ToolReminderTag
reminder.add(`Background process detached (PID ${pid}). You will see its stdout/stderr output in your system context each turn. Use \`kill ${pid}\` to stop it.`)
```

### What the agent sees
```
<system>
<results>
<shell observe=".">
  <mode>detached</mode>
  <pid>48291</pid>
  <stdout>...</stdout>
  <stderr>...</stderr>
</shell>
</results>
<reminder>Background process detached (PID 48291). You will see its stdout/stderr output in your system context each turn. Use `kill 48291` to stop it.</reminder>
</system>
```

---

## Concrete implementation plan by layer

## A. Shell tool layer
**Primary file:** `packages/agent/src/tools/shell.ts` (current impl at lines 1-89)

### A1. Replace `exec`/`promisify` with `spawn`
- Remove `exec`/`promisify` imports around lines 10-14
- Import `spawn` from `child_process`
- Run via platform shell explicitly, not `exec`:
  - Unix: `spawn(process.env.SHELL ?? '/bin/sh', ['-c', command], ...)`
  - Windows fallback if needed later, but current session/platform is macOS
- Preserve:
  - `cwd` from `WorkingDirectoryTag`
  - inherited env + `NO_COLOR=1`

### A2. Introduce bounded stream collectors
Inside `shell.ts`, add a small helper for:
- collecting stdout/stderr incrementally from `child.stdout` / `child.stderr`
- storing:
  - initial output for the first 5s response
  - rolling recent buffer for registry handoff
  - truncation flags/counters

Recommended helper shape:
```ts
type StreamWindow = {
  text: string
  bytes: number
  truncated: boolean
}
```

Use:
- a small "initial response" cap (ex. 32-64 KB per stream)
- a larger rolling registry cap (ex. 128-256 KB per stream)

### A3. Add 5s initial wait + auto-detach logic
In `execute`, after spawn:
- await whichever happens first:
  - process exits
  - 5s timer fires

If process exits before timer:
- return `{ mode: 'completed', stdout, stderr, exitCode }`

If timer fires first and process is still alive:
- register process with new registry service
- return:
```ts
{
  mode: 'detached',
  pid,
  stdout: initialStdout,
  stderr: initialStderr,
}
```

### A4. Add registry dependency to shell tool
Inject a new Effect service, e.g.:
- `BackgroundProcessRegistryTag`
in `shell.ts`

On detach:
- call `registry.register({...})`

### A5. Keep non-zero shell exit codes as data
For completed foreground runs:
- preserve current convention from lines 57-82: tool transport success, shell exit in payload
- but now derive exit from `child.on('exit')` rather than `exec` exception handling

### A6. Add kill detection notes for detached results
If the process exits just after detach registration and before response serialization:
- still return `detached`
- registry will quickly emit `background_process_exited`

### A7. Tests
Add/update tests near shell tool coverage (new file likely needed under `packages/agent/src/tools/__tests__/shell-background.test.ts` or existing harness tests):
- completes under 5s => `mode: completed`
- exceeds 5s => `mode: detached` with PID
- captures initial output before detach
- truncation flag set when output exceeds response cap
- non-zero foreground exit preserved

---

## B. New process registry component
**New files recommended**
- `packages/agent/src/processes/background-process-registry.ts`
- optional `packages/agent/src/processes/types.ts`

### B1. Create a session-scoped registry service
Implement a service similar in style to other session services:
```ts
interface BackgroundProcessRegistryService {
  register(processInfo): Effect.Effect<RegisteredProcess>
  listByFork(forkId): Effect.Effect<...>
  getByPid(pid): Effect.Effect<...>
  noteSignalAttempt(pid, signal?): Effect.Effect<void>
  cleanupFork(forkId): Effect.Effect<void>
  shutdownAll(): Effect.Effect<void>
}
```

Store:
- `Map<number, BackgroundProcessRecord>`

Record fields:
```ts
{
  pid: number
  forkId: string | null
  turnId: string
  command: string
  startedAt: number
  child: ChildProcess
  status: 'running' | 'exited'
  exitCode: number | null
  signal: string | null

  // Output buffering (inline mode)
  stdoutRing: RingBuffer
  stderrRing: RingBuffer
  stdoutSeq: number
  stderrSeq: number

  // File demotion
  outputMode: 'inline' | 'file'
  stdoutFilePath: string | null   // e.g. ~/.magnitude/tmp/{pid}-stdout.log
  stderrFilePath: string | null
  stdoutLineCount: number
  stderrLineCount: number

  truncated: { stdout: boolean; stderr: boolean }
  lastPublishedAt: number
}
```

When `outputMode` is `'inline'`, output accumulates in ring buffers and is published in `background_process_output` events.

When output between turns exceeds ~8KB, `outputMode` flips to `'file'`:
- Ring buffers are flushed to the file and then dropped
- All subsequent output appends directly to `~/.magnitude/tmp/{pid}-stdout.log` / `stderr.log`
- Events still published (for projection/UI) but with file metadata instead of inline chunks
- Demotion is permanent per process — no bouncing back

### B2. Publish background-process events via WorkerBus
Registry should receive `WorkerBusTag<AppEvent>` in its layer and publish:
- registration event immediately on detach
- output chunk events on throttled intervals
- exit event on close/exit

### B3. Output batching/throttling
Do **not** emit one event per stream chunk.
Instead:
- accumulate stdout/stderr chunks in memory
- publish at a fixed cadence, e.g. every 250-1000ms, or when bytes exceed a threshold

Event shape is defined in section C1 (`BackgroundProcessOutput`), with `outputMode` discriminator for inline vs file mode.

This limits event spam and gives backpressure control.

### B4. Rolling buffer policy
Registry should keep:
- a rolling recent output window per stream (for UI + agent context)
- cumulative truncation booleans/counters

Do **not** persist full process output forever in memory.

Suggested defaults:
- ring buffer: 128-256 KB per stream
- publish chunk cap: 8-16 KB per event
- "dropped bytes" counters for observability

### B5. Exit / killed detection
Listen on child:
- `exit`
- `close`
- `error`

When exited:
- publish `background_process_exited` with:
```ts
{
  type: 'background_process_exited',
  forkId,
  pid,
  exitCode,
  signal,
  finalStdout: maybeSmallTail,
  finalStderr: maybeSmallTail,
  status: signal ? 'killed' : 'exited'
}
```

Do not create a separate `killed` event unless strongly desired; `signal !== null` is enough.
If you want explicit lifecycle types, define:
- `status: 'exited' | 'killed'`

### B6. Detect agent-issued `kill <pid>` cleanly
We do **not** need to intercept `kill` shell commands.
Instead:
- the OS signal causes the tracked child handle to emit exit/close
- registry sees it and removes the PID from active records
- publish `background_process_exited` with `signal='SIGTERM' | 'SIGKILL'`

### B7. Cleanup APIs
Registry should support:
- `cleanupFork(forkId)` for agent dismissal if desired
- `shutdownAll()` on client dispose/session shutdown

Open question:
- whether session shutdown should actively kill all tracked children or merely stop observing them.
- Recommended: **kill tracked children on session dispose** to avoid orphan observables, unless product explicitly wants persistence outside Magnitude.

---

## C. Event system
**Primary file:** `packages/agent/src/events.ts` (tool event definitions around lines 249-276; union ends around lines 470-505)

### C1. Add new event interfaces
Add:
```ts
export interface BackgroundProcessRegistered { ... }
export interface BackgroundProcessOutput { ... }
export interface BackgroundProcessExited { ... }
```

Suggested shapes:

```ts
export interface BackgroundProcessRegistered {
  readonly type: 'background_process_registered'
  readonly forkId: string | null
  readonly pid: number
  readonly command: string
  readonly sourceToolCallId: string
  readonly sourceTurnId: string
  readonly startedAt: number
  readonly initialStdout: string
  readonly initialStderr: string
  readonly truncated: boolean
}

export interface BackgroundProcessOutput {
  readonly type: 'background_process_output'
  readonly forkId: string | null
  readonly pid: number
  readonly outputMode: 'inline' | 'file'
  // Inline mode
  readonly stdoutChunk: string
  readonly stderrChunk: string
  // File mode
  readonly stdoutFilePath?: string
  readonly stderrFilePath?: string
  readonly stdoutNewLines?: number
  readonly stderrNewLines?: number
  readonly stdoutTotalLines?: number
  readonly stderrTotalLines?: number
  readonly stdoutTail?: string
  readonly stderrTail?: string
  // Common
  readonly stdoutSeq: number
  readonly stderrSeq: number
}

export interface BackgroundProcessExited {
  readonly type: 'background_process_exited'
  readonly forkId: string | null
  readonly pid: number
  readonly exitCode: number | null
  readonly signal: string | null
  readonly status: 'exited' | 'killed'
  readonly stdoutTail: string
  readonly stderrTail: string
}
```

### C2. Extend `AppEvent` union
Add these to the union near the existing streaming/control events.

### C3. Serialization/tests
Update persistence serialization tests:
- `packages/agent/src/persistence/__tests__/serialization.test.ts`
to cover round-tripping the new event types.

---

## D. Execution manager integration
**Primary file:** `packages/agent/src/execution/execution-manager.ts`

Relevant anchors:
- fork layer assembly in `makeForkLayers(...)` around lines 152-227
- service creation around lines 234-807
- `initFork` around lines 724-780
- `disposeFork` around lines 783-788

### D1. Inject registry service into fork layers
In `makeForkLayers(...)`, merge a `BackgroundProcessRegistryTag` service so `shellTool` can call it.

Since processes are **session-scoped**, build the registry once in `makeExecutionManager` and capture it like:
```ts
const processRegistry = yield* makeBackgroundProcessRegistry(...)
```
Then provide it in every fork's layer.

### D2. Give shell tool enough metadata to register source linkage
The registry should know:
- owner `forkId`
- active `turnId`
- source `toolCallId`
- original command

Today tools do not directly know `turnId`/`toolCallId`.
There are two options:

#### Preferred
Add a tiny execution-context tag, e.g. `ToolExecutionContextTag`, set by the runtime/execution layer before tool execution if xml-act allows it.

#### Simpler if runtime metadata is unavailable
Register without `sourceToolCallId` in the registry, and only link to `forkId`/`turnId`/command.
This is acceptable for v1.

Because the request asked for concrete design, I recommend adding:
- `packages/agent/src/execution/tool-execution-context.ts` (new)
and populating it where `ToolExecutionStarted` is handled or where registered tools are wrapped.

If that is too invasive after implementation review, drop `sourceToolCallId` from event shape.

### D3. No changes required to xml-act `TurnEvent` flow for post-turn updates
`packages/agent/src/execution/types.ts` should likely remain unchanged.
Reason:
- background-process updates are no longer part of the active xml-act turn stream
- they go directly through `WorkerBus<AppEvent>`

This is an important simplification.

### D4. Fork disposal cleanup
In `disposeFork` (around lines 783-788):
- also tell registry to `cleanupFork(forkId)` or at least stop waking that fork
- if dismissal semantics should kill owned processes, do it here
- if not, retain only for root session lifetime

Recommendation:
- on `disposeFork(forkId)`, kill/cleanup child-fork-owned processes
- on root/session dispose, kill all tracked processes

---

## E. Turn loop integration
**Primary files**
- `packages/agent/src/projections/working-state.ts`
- `packages/agent/src/coding-agent.ts`

### E1. WorkingState handles `background_process_exited` directly
Add a handler in `WorkingStateProjection` for `background_process_exited`:
- Set `willContinue = true` for `event.forkId`
- This follows the same pattern as `user_message` — it's an event that the agent needs to react to
- If the fork is currently working, the exit info accumulates in the projection and the agent sees it on its next natural turn
- If the fork is idle, this triggers a new turn so the agent can react to the process completing

No wake events. No background-process worker. Just direct event handling in WorkingState.

### E2. No worker needed
Unlike the original plan, there is **no `background-process-worker.ts`**. The architecture is:
- Registry publishes events → Projection consumes them → Observable reads projection at turn start
- WorkingState handles exit events for scheduling
- That's it. No intermediary worker.

### E3. Session shutdown cleanup
On client dispose in `packages/agent/src/coding-agent.ts`:
- call registry `shutdownAll()` before `originalDispose()` completes
- ensure best-effort kill of tracked processes

---

## F. Agent awareness / observables / context
This is the most important behavior question: how does the next turn know what's running and what changed?

### F1. Add a projection for background process state
**New file recommended**
- `packages/agent/src/projections/background-processes.ts`

State per fork:
```ts
interface BackgroundProcessSummary {
  pid: number
  command: string
  status: 'running' | 'exited' | 'killed'
  startedAt: number
  lastUpdatedAt: number
  exitCode: number | null
  signal: string | null

  // Output mode
  outputMode: 'inline' | 'file'

  // Inline mode fields
  recentStdout: string
  recentStderr: string
  unreadStdout: string
  unreadStderr: string

  // File mode fields
  stdoutFilePath: string | null
  stderrFilePath: string | null
  stdoutNewLines: number
  stderrNewLines: number
  stdoutTotalLines: number
  stderrTotalLines: number
  stdoutTail: string   // last ~20 lines for file mode display
  stderrTail: string

  unreadEventCount: number
}
```

Projection handlers:
- `background_process_registered` => add process
- `background_process_output` => append to `recent*` and `unread*`
- `background_process_exited` => finalize status and tails
- `turn_started` or `turn_completed` => clear `unread*` once the agent has had a chance to see them

### F2. Expose process updates through an observable
Current `ExecutionManager.initFork` binds `agentDef.observables` around lines 767-779.
Recommended new observable:
- session/fork-specific "background processes" observable
- implemented similarly to existing projection-based observables

This should render into agent context something like:
```xml
<background_processes>
  <process pid="1234" status="running">
    <command>npm test --watch</command>
    <stdout>...</stdout>
    <stderr>...</stderr>
  </process>
</background_processes>
```

or a compact textual summary, depending on observable framework expectations.

The key is:
- include only **running processes** plus **unread deltas since last turn**
- include recent exit info for just-exited processes until acknowledged

### F3. Decide when unread updates are consumed
Recommended:
- unread process deltas remain visible until the next `turn_started` for that fork
- then projection marks them as seen/consumed

Why:
- ensures a wake-triggered turn definitely sees the update
- avoids repeatedly surfacing the same chunk forever

### F4. Root/orchestrator visibility
For `forkId === null`, surface root-owned processes directly.
For child agents, keep process visibility fork-local, same as other local state.

Open question:
- whether orchestrator should also see child-fork processes.
Recommendation for v1: **no**. Parent already has child-agent activity summaries; keep process observability local to the fork that launched it.

---

## G. Cleanup and lifecycle semantics

### G1. Session end
On client dispose:
- kill all tracked running processes with `SIGTERM`
- after short grace period, `SIGKILL` if still alive
- clean up temp files in `~/.magnitude/tmp/` for all tracked PIDs
- then clear registry

Files:
- `packages/agent/src/coding-agent.ts`
- `packages/agent/src/execution/execution-manager.ts`
- new registry file

### G2. Agent dismissal
On `agent_dismissed` currently `AgentOrchestrator` calls `execManager.disposeFork(event.forkId)` in `packages/agent/src/workers/agent-orchestrator.ts` around lines 46-52.
Extend that cleanup to:
- kill any running processes owned by that fork
- emit final exit/killed events if appropriate
- clean up any temp files in `~/.magnitude/tmp/` for processes owned by that fork

### G3. Hydration / crash recovery
Current hydration recovery in `coding-agent.ts` handles unstable turns, but OS child processes cannot be reattached after Magnitude itself dies.

Recommendation:
- do **not** attempt reattachment in v1
- persist only historical process events already emitted
- any live children that survive a crash become unmanaged; document this limitation

Optional improvement:
- on startup, emit a synthetic warning event if persistence indicates running processes were previously detached but have no live registry entry after hydration.

### G4. Orphan cleanup policy
Since registry only tracks children spawned in this process, "orphan cleanup" means:
- kill tracked children on clean shutdown
- accept that unclean crash may leak them
- explicitly document this as a limitation

---

## H. Shell classifier / permission model
**Primary files**
- `packages/agent/src/execution/permission-gate.ts`
- `packages/shell-classifier/src/parser.ts`
- `packages/shell-classifier/src/classifier.ts`

### H1. No required permission-gate changes for core design
`permission-gate.ts` lines 42-66 already classify shell commands before execution.
That can remain unchanged for auto-detach.

### H2. Optional classifier hardening for explicit shell backgrounding
Parser currently treats bare `&` as a separator in `packages/shell-classifier/src/parser.ts` lines 246-262.
That means explicit `cmd &` is not specially surfaced, just parsed as separated commands.

Recommendation:
- **do not block this redesign on classifier changes**
- but add a follow-up task to classify daemonization-oriented commands as `normal` or annotate them:
  - `nohup`
  - `disown`
  - `setsid`
  - shell `&`

Reason:
- with auto-detach, the agent no longer needs explicit shell backgrounding, but forbidding it outright would be behavior-changing and unrelated to the core implementation.

### H3. `kill <pid>` should remain allowed as current normal shell
No special approval path needed beyond existing agent/tool policy.

### H4. Update shell tool description to discourage explicit backgrounding
Add guidance to the shell tool's description (in the agent definition / tool docs) explaining:
- Long-running commands are automatically detached after 5s with full output tracking
- The agent should **not** use `&`, `nohup`, `disown`, or other explicit backgrounding — these bypass the tracking system and create orphaned processes
- Just run the command normally and the system handles the rest

---

## I. Display / UI
**Primary files**
- `packages/agent/src/projections/display.ts`
- `packages/agent/src/visuals/shell.ts`
- `cli/src/visuals/shell.tsx`

### I1. Extend shell visual reducer for detached result
`packages/agent/src/visuals/shell.ts` currently assumes `Success` output is `{ stdout, stderr, exitCode }` (lines 26-39, 61-66).
Change reducer/result mapping to understand:
- `mode: 'completed'`
- `mode: 'detached'`

Suggested state additions:
```ts
type DoneVariant =
  | { kind: 'success'; ... }
  | { kind: 'detached'; pid: number; stdout: string; stderr: string; note: string }
  | ...
```

### I2. Add background process messages/steps to display projection
In `packages/agent/src/projections/display.ts`, add a new display message type, e.g.:
```ts
interface BackgroundProcessMessage {
  id: string
  type: 'background_process'
  pid: number
  command: string
  status: 'running' | 'exited' | 'killed'
  startedAt: number
  updatedAt: number
  stdout: string
  stderr: string
  exitCode: number | null
  signal: string | null
}
```

Handler behavior:
- `background_process_registered`: insert/update a message in the owning fork timeline
- `background_process_output`: append chunks to that message
- `background_process_exited`: mark complete and keep final state visible

Recommendation:
- render as a standalone timeline message, **not** a think-block tool step, because it outlives the originating turn

### I3. New CLI renderer
Add CLI support in `cli/src/components/message-view.tsx` plus a renderer component if needed.
Show:
- running indicator
- PID
- command
- rolling stdout/stderr preview
- exit status when complete

### I4. Keep the original shell tool card concise
For the tool invocation itself:
- when detached, shell step should say something like `Detached background process PID 1234`
- ongoing output should appear in the new background-process timeline entry

This avoids mutating a stale tool step long after the turn ended.

### I5. Server/session manager compatibility
If web/server path consumes display state from `serve/session-manager.ts`, extend the default display typing only if new `DisplayMessage` variant requires frontend handling.

---

## J. Turn/memory/debug projections

### J1. New projection registration
Register `BackgroundProcessesProjection` in `packages/agent/src/coding-agent.ts` near other projections.

### J2. Consider memory transcript behavior
Current memory/conversation systems key primarily off `turn_completed`, `message_*`, and tool results.
Recommendation for v1:
- do **not** add raw background chunks to long-term memory transcript automatically
- let the next agent turn summarize/respond to them naturally

Why:
- raw tailing output can be huge/noisy
- memory should reflect what the agent acted on, not every emitted byte

### J3. Debug panel / event inspection
Because new app events flow through the normal event bus, debug tooling should start seeing them automatically.
Still update any switch statements in CLI debug UI that enumerate event types.

Files likely touched:
- `cli/src/components/debug-panel.tsx`
- `packages/agent/src/memory/transcript.ts` only if exhaustive matching complains

---

## K. Exact file change list

### Core implementation
1. `packages/agent/src/tools/shell.ts`
2. `packages/agent/src/events.ts`
3. `packages/agent/src/execution/execution-manager.ts`
4. `packages/agent/src/coding-agent.ts`
5. `packages/agent/src/projections/display.ts`
6. `packages/agent/src/visuals/shell.ts`
7. `cli/src/visuals/shell.tsx`
8. `packages/agent/src/workers/agent-orchestrator.ts` (fork cleanup semantics)
9. `packages/agent/src/execution/permission-gate.ts` (likely type-only/exhaustiveness, maybe no logic change)

### Modified files
10. `packages/agent/src/projections/working-state.ts` (handle `background_process_exited`)
11. `packages/agent/src/observables/projection-reader.ts` (extend with background process state getter)
12. `packages/agent/src/prompts/agents.ts` (wrap reminders in `<reminder>` tags)

### New files
13. `packages/agent/src/processes/background-process-registry.ts`
14. `packages/agent/src/processes/types.ts` (optional)
15. `packages/agent/src/projections/background-processes.ts`
16. `packages/agent/src/observables/background-processes-observable.ts`
17. `packages/agent/src/execution/tool-reminder.ts` (`ToolReminderTag` service)
18. `packages/agent/src/execution/tool-execution-context.ts` (optional, if source toolCallId/turnId injection is implemented)

### Tests / supporting updates
16. `packages/agent/src/persistence/__tests__/serialization.test.ts`
17. `packages/agent/src/test-harness/...` tests for event flow
18. `packages/shell-classifier/src/__tests__/...` only if daemonization rules are changed
19. `cli/src/components/debug-panel.tsx` if exhaustive event rendering needs updating

---

## Suggested implementation order

### Phase 1: Make shell detach and observe
1. Add registry service + tests
2. Convert `shell.ts` to `spawn`
3. Implement 5s wait / detach result
4. Publish `background_process_registered/output/exited` events

### Phase 2: Make updates visible and actionable
5. Add `BackgroundProcessesProjection`
6. Add `backgroundProcessesObservable`
7. Extend `ProjectionReader` and wire observable into agent definitions
8. Add `background_process_exited` handler in `WorkingState`
9. Register projection in `coding-agent.ts`

### Phase 3: UI + cleanup polish
9. Extend shell visual reducer for detached result
10. Add display message + CLI rendering for running processes
11. Add shutdown/dispose cleanup
12. Add dismissal cleanup

### Phase 4: Optional hardening
13. Add tool execution context linkage (`sourceToolCallId`)
14. Add classifier daemonization follow-up
15. Add crash/hydration warning behavior if needed

---

## Event flow after implementation

### Fast command
1. agent emits `<shell>npm test -- --runInBand</shell>`
2. shell tool spawns process
3. exits before 5s
4. ordinary `tool_event` + `turn_completed` path unchanged
5. result is `mode: completed`

### Slow command
1. agent emits `<shell>npm run dev</shell>`
2. shell tool spawns process
3. after 5s, tool registers child with registry
4. tool returns `mode: detached` with PID, initial stdout/stderr, and clear note
5. turn ends normally
6. registry publishes `background_process_output` events on throttled intervals
7. `BackgroundProcessesProjection` accumulates unread output
8. on next natural turn start, observable reads projection and surfaces accumulated output
9. agent can respond, inspect, or run `kill <pid>`
10. if process exits, registry publishes `background_process_exited`
11. `WorkingState` handles `background_process_exited`, sets `willContinue = true`
12. if fork was idle, a new turn starts; observable shows exit info + final output

---

## Open questions / uncertainties

### 1. Does xml-act expose per-tool execution metadata to tools easily?
- If yes, include `sourceToolCallId` and `turnId` in registry events
- If not, omit those from v1 event shape

### 2. Should process output events be persisted?
Recommended:
- yes, because they are normal `AppEvent`s and useful for UI/debug history
- but keep chunk sizes bounded aggressively

### 3. Should session shutdown kill processes?
Recommended:
- yes, kill tracked processes on clean shutdown
- otherwise observables become misleading and leak host resources

### 4. Should detached process output enter memory automatically?
Recommended:
- no direct raw-memory ingestion
- observable surfaces it at turn start; the agent's natural turn response captures what matters

### 5. Should child-fork processes be visible to the parent orchestrator?
Recommended:
- no for v1; keep process state local to the owning fork

---

## Notes on specific existing anchors

- `packages/agent/src/tools/shell.ts:10-14, 57-82`
  - current `exec` implementation and timeout handling to replace
- `packages/agent/src/events.ts:249-276, 470-505`
  - where `ToolEvent`, `ToolResult`, and `AppEvent` union live
- `packages/agent/src/execution/execution-manager.ts:152-227`
  - fork layer assembly; inject registry here
- `packages/agent/src/execution/execution-manager.ts:724-780`
  - `initFork`, where bound observables are created
- `packages/agent/src/execution/execution-manager.ts:783-788`
  - `disposeFork`, add process cleanup hook
- `packages/agent/src/workers/turn-event-drain.ts:21-50`
  - confirms turn-stream mapping is separate; background process events should bypass this
- `packages/agent/src/projections/display.ts:773-865`
  - current `tool_event` handling, useful for detached shell card update
- `packages/agent/src/projections/display.ts:898-931`
  - `turn_completed` handling; background-process UI must not depend on active turn
- `packages/agent/src/projections/working-state.ts:109-123`
  - `wake` behavior used for between-turn process updates
- `packages/agent/src/projections/working-state.ts:157-194`
  - end-of-turn scheduling already coalesces follow-up work
- `packages/agent/src/visuals/shell.ts:15-39, 53-71`
  - shell visual result assumptions to broaden
- `cli/src/visuals/shell.tsx`
  - renderer needs detached/running presentation
- `packages/agent/src/execution/permission-gate.ts:42-66`
  - existing shell permission path; likely no logic change required
- `packages/shell-classifier/src/parser.ts:246-262`
  - current `&` tokenization behavior
- `packages/shell-classifier/src/classifier.ts:24-36, 77-116`
  - current classification pipeline