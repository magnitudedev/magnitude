
# Shell Tool & Background Process System

This document describes the complete design of Magnitude's shell tool and background process management system.

## Overview

The shell tool executes commands in a subprocess. Commands that finish quickly return their result inline. Commands that exceed a declared timeout are automatically **detached** — they continue running as tracked background processes with output surfaced to the agent on subsequent turns.

The system enforces intentional process management: the agent must declare whether a command is expected to terminate (and how long it should take) or whether it's a persistent background process. Unexpected hangs are caught and escalated through a two-stage timeout lifecycle.

---

## Shell Tool

### Interface

```xml
<shell timeout="30">npm test</shell>
<shell background="true">npm run dev</shell>
<shell>ls -la</shell>
```

**Parameters:**
- `command` (required, body) — the command to execute
- `timeout` (optional, number) — expected duration in seconds (default: 10). If the command hasn't finished by this time, it is detached as a background process — not killed.
- `background` (optional, boolean) — for commands that won't terminate on their own (servers, watchers). Detaches immediately.

**Description:**
```
Execute a shell command. Long-running commands are automatically detached after the timeout
with output tracking. Do NOT use &, nohup, disown, or other explicit backgrounding; these
bypass tracking. Use kill <pid> to stop background processes. Do not use this for operations
covered by built-in tools like fs-read, fs-search, fs-tree, fs-write, edit, and web-fetch.
```

### Output

Discriminated union on `mode`:

**Completed** (command finished within timeout):
```xml
<shell>
  <mode>completed</mode>
  <stdout>...</stdout>
  <stderr>...</stderr>
  <exitCode>0</exitCode>
</shell>
```

**Detached** (command exceeded timeout or was explicitly backgrounded):
```xml
<shell>
  <mode>detached</mode>
  <reason>timeout_exceeded</reason>
  <pid>12345</pid>
  <stdout>...</stdout>
  <stderr>...</stderr>
</shell>
```

The `reason` field distinguishes intentional backgrounding from unexpected timeout:
- `background` — agent set `background="true"`
- `timeout_exceeded` — command didn't finish within the declared timeout

### Execution

1. Command is spawned via `$SHELL -c <command>` (fallback `/bin/sh`) with `NO_COLOR=1`
2. Working directory is the session/fork cwd
3. stdout and stderr are accumulated
4. **If `background=true`**: detach after `BACKGROUND_DETACH_MS` (1s) — enough to capture initial output
5. **Otherwise**: detach after `timeout` seconds (default: `DEFAULT_TIMEOUT_S` = 10s)
6. If the command finishes before the detach timer: return `completed` result
7. If the detach timer fires first: register process with the background process registry, return `detached` result
8. If the agent's turn is interrupted while the command is running: send `SIGTERM` to the child

---

## Two-Stage Timeout Lifecycle

When a non-background command exceeds its timeout, it enters a two-stage lifecycle that forces resolution:

### Stage 1: Detach (at 1× timeout)

- Process is detached (not killed) and registered with the background process registry
- Agent receives the `detached` result with `reason: 'timeout_exceeded'`
- An auto-kill deadline is set at `2× timeout + 5s` from command start
- Agent sees a reminder:
  > "Command exceeded expected timeout and was detached (PID X). This process will be automatically terminated in Ns. If the process is hanging, kill it now with `kill X`. If it is still working correctly and you want it to keep running, declare it as a background process with `<shell-bg pid="X" />`."

### Stage 2: Auto-kill (at 2× timeout + 5s)

- If the agent has not killed or promoted the process by this deadline, the system automatically kills it
- Agent is woken via `willContinue: true` and sees:
  > "Process PID X was automatically terminated after exceeding 2× its expected duration with no resolution."
- Surfaced via `background_process_auto_killed` event

### Agent escape hatches (before Stage 2)

1. **`kill <pid>`** — agent decides it's hanging, kills it via shell
2. **`<shell-bg pid="X" />`** — agent decides it's fine, promotes to background process

### Background promotion

The `shell-bg` tool promotes a `timeout_exceeded` process to a full background process:

```xml
<shell-bg pid="12345" />
```

- Flips the process's `reason` from `timeout_exceeded` to `background`
- Cancels the auto-kill timer
- From this point the process is treated as a normal background process (output tracked, no termination deadline)

---

## Background Process Registry

Session-scoped, in-memory registry that tracks all detached processes.

### Interface

```ts
interface BackgroundProcessRegistry {
  register(input: RegisterProcessInput): Effect<void>
  promote(pid: number): Effect<void>
  flush(forkId: string): Effect<void>
  listByFork(forkId: string): Effect<BackgroundProcessRecord[]>
  getByPid(pid: number): Effect<BackgroundProcessRecord | undefined>
  cleanupFork(forkId: string): Effect<void>
  shutdownAll(): Effect<void>
}
```

### Process Record

```ts
interface BackgroundProcessRecord {
  pid: number
  forkId: string | null
  turnId: string
  command: string
  reason: 'background' | 'timeout_exceeded'
  startedAt: number
  child: ChildProcess
  autoKillTimer?: ReturnType<typeof setTimeout>
  timeoutSeconds?: number
  status: 'running' | 'exited'
  exitCode: number | null
  signal: string | null
  stdoutBuffer: string
  stderrBuffer: string
  demoted: boolean
  stdoutFilePath: string | null
  stderrFilePath: string | null
}
```

### Registration flow

1. Create record, store in `Map<pid, record>`
2. Publish `background_process_registered` event
3. Attach stdout/stderr listeners to continue buffering post-detach
4. Attach exit/close/error handlers for finalization
5. If `reason === 'timeout_exceeded'`: start auto-kill timer at `(timeoutSeconds + AUTO_KILL_BUFFER_S) * 1000` ms

### Output flushing

Between agent turns, the execution manager calls `flush(forkId)` to publish accumulated output:

**Inline mode** (default):
- Publishes `background_process_output` with `mode: 'inline'`, containing stdout/stderr chunks

**Demotion** (triggered when buffered output exceeds `DEMOTION_THRESHOLD_CHARS` = 8192):
- Creates log files at `~/.magnitude/tmp/{pid}-stdout.log` and `{pid}-stderr.log`
- Marks record as demoted (permanent)
- Publishes `background_process_demoted` event
- Subsequent flushes publish `background_process_output` with `mode: 'tail'`, containing a line-aligned tail (max `TAIL_MAX_CHARS` = 4096 chars)

### Exit finalization

When a tracked process exits:
- Flush remaining buffered output
- Publish `background_process_exited` with status (`exited` or `killed`), exit code, and signal
- Cancel auto-kill timer if present

### Promotion

`promote(pid)`:
- Validates process exists, is running, has `reason: 'timeout_exceeded'`
- Sets `reason` to `'background'`
- Cancels auto-kill timer
- Publishes `background_process_promoted`

### Cleanup

- `cleanupFork(forkId)`: terminates all processes owned by that fork
- `shutdownAll()`: terminates all processes, cleans up temp files
- Termination policy: `SIGTERM` → wait `SHUTDOWN_GRACE_MS` (2s) → `SIGKILL`

---

## Event System

All background process lifecycle events are first-class `AppEvent`s published through the worker bus:

| Event | Key Fields |
|---|---|
| `background_process_registered` | pid, forkId, command, reason, initialStdout, initialStderr |
| `background_process_output` | pid, forkId, mode (inline/tail), stdout/stderr chunks |
| `background_process_demoted` | pid, forkId, stdoutFilePath, stderrFilePath |
| `background_process_exited` | pid, forkId, exitCode, signal, status (exited/killed) |
| `background_process_auto_killed` | pid, forkId, command |
| `background_process_promoted` | pid, forkId |

---

## Projection: BackgroundProcessesProjection

Consumes background process events and maintains per-fork process state.

### State

```ts
Map<forkId, Map<pid, BackgroundProcessState>>
```

Per-process state includes:
- pid, command, status, reason, startedAt
- exitCode, signal
- demoted, stdoutFilePath, stderrFilePath
- totalStdoutLines, totalStderrLines
- newStdoutLines, newStderrLines
- unreadStdout, unreadStderr

### Event handling

- `background_process_registered` → create entry, seed unread output
- `background_process_output` → accumulate unread output
- `background_process_demoted` → store file paths, mark demoted
- `background_process_exited` → set final status
- `background_process_promoted` → update reason to `'background'`
- `background_process_auto_killed` → set status to `'killed'`
- `observations_captured` → clear unread output for running processes (the "since last turn" mechanism)

---

## Observable: backgroundProcessesObservable

Reads the projection and formats process state as system context injected into the agent's turn.

### Format

```xml
<background_processes>
<process pid="48291" status="running" reason="background" command="npm run dev">
<new_stdout>
Server started on port 3000
</new_stdout>
<new_stderr>
(no new output)
</new_stderr>
</process>
</background_processes>
```

### Formatting rules

- **Running, inline**: show `<new_stdout>` / `<new_stderr>` if unread output exists, otherwise `(no new output since last turn)`
- **Running, demoted**: show `<stdout file="..." newLines="N" totalLines="N">tail...</stdout>`
- **Exited/killed**: show `<final_stdout>` / `<final_stderr>` with exit code or signal
- Command truncated to 80 chars
- `reason` attribute included on `<process>` element
- Empty when no processes exist (observable returns `[]`)

### Included by

All major agent types: orchestrator, builder, debugger, explorer, planner, reviewer.

---

## Working State Integration

- `background_process_exited` → sets `willContinue: true` for the owning fork (wakes idle agent)
- `background_process_auto_killed` → sets `willContinue: true` (wakes agent to see termination)

---

## Tool Reminders

On detach, the shell tool adds a `ToolReminderTag` message that appears in the agent's system context:

**`background`** (intentional):
> "Background process started (PID X). You will see its stdout/stderr output in your system context each turn. Use `kill X` to stop it."

**`timeout_exceeded`** (unexpected):
> "Command exceeded expected timeout and was detached (PID X). This process will be automatically terminated in Ns. If the process is hanging, kill it now with `kill X`. If it is still working correctly and you want it to keep running, declare it as a background process with `<shell-bg pid="X" />`."

---

## Execution Manager Wiring

- Creates one `BackgroundProcessRegistry` instance per session
- Provides it to all fork layers via `BackgroundProcessRegistryTag`
- Per-fork `ProjectionReader` restricts visibility to that fork's processes
- `flushProcesses(forkId)` called between turns to publish accumulated output
- `interruptProcesses(forkId)` for fork cleanup

---

## Lifecycle & Recovery

### Session shutdown
- `shutdownAll()` kills all tracked processes (SIGTERM → grace → SIGKILL)
- Cleans up demoted output files

### Fork dismissal
- `cleanupFork(forkId)` kills processes owned by that fork

### Hydration / crash recovery
- Background processes are NOT resumed across app restarts
- On hydration, any process still marked running is synthetically marked as killed (`signal: 'SIGTERM'`, `status: 'killed'`)

---

## Constants

| Constant | Value | Purpose |
|---|---|---|
| `DEFAULT_TIMEOUT_S` | 10 | Default shell timeout when none specified |
| `AUTO_KILL_BUFFER_S` | 5 | Extra seconds added to 2× timeout for auto-kill deadline |
| `BACKGROUND_DETACH_MS` | 1000 | Delay before detaching `background=true` commands |
| `DEMOTION_THRESHOLD_CHARS` | 8192 | Output size that triggers demotion to file mode |
| `TAIL_MAX_CHARS` | 4096 | Max chars in tail for demoted output |
| `SHUTDOWN_GRACE_MS` | 2000 | Grace period between SIGTERM and SIGKILL |
| `OUTPUT_DIR_NAME` | `.magnitude/tmp` | Directory under `$HOME` for demoted output files |

---

## Key Files

| Area | File |
|---|---|
| Shell tool | `packages/agent/src/tools/shell.ts` |
| Shell-bg tool | `packages/agent/src/tools/shell-bg.ts` |
| Registry | `packages/agent/src/processes/background-process-registry.ts` |
| Constants | `packages/agent/src/processes/constants.ts` |
| Types | `packages/agent/src/processes/types.ts` |
| Events | `packages/agent/src/events.ts` |
| Projection | `packages/agent/src/projections/background-processes.ts` |
| Working state | `packages/agent/src/projections/working-state.ts` |
| Observable | `packages/agent/src/observables/background-processes-observable.ts` |
| Projection reader | `packages/agent/src/observables/projection-reader.ts` |
| Execution manager | `packages/agent/src/execution/execution-manager.ts` |
| App lifecycle | `packages/agent/src/coding-agent.ts` |
| Tool reminder | `packages/agent/src/execution/tool-reminder.ts` |
