# Magnitude Current AGENTS.md Handling

## Executive Summary

Magnitude currently has a **minimal, flat** AGENTS.md mechanism: it checks only for `AGENTS.md` (or `CLAUDE.md`) at the **project root (cwd)** at session initialization. There is **no directory hierarchy scanning**, **no per-file tracking** of which AGENTS.md files have been shown, and **no injection triggered by file read/write/edit operations**.

---

## 1. Collection: Only cwd Root, Only Once

**File:** `packages/agent/src/util/collect-session-context.ts` (lines 155-167)

```typescript
async function readAgentsFile(cwd: string): Promise<{ filename: string; content: string } | null> {
  const filenames = ['AGENTS.md', 'CLAUDE.md']

  for (const filename of filenames) {
    try {
      const content = await readFile(join(cwd, filename), 'utf8')
      return { filename, content: content.trim() }
    } catch {
      continue
    }
  }

  return null
}
```

**Key observations:**
- Only checks `cwd` root — no walking up the directory tree
- Only reads one file (prefers `AGENTS.md`, falls back to `CLAUDE.md`)
- Collected once at session init via `collectSessionContext()` (line 183)
- If session init fails, `agentsFile: null` is the fallback (line 419 of `coding-agent.ts`)

---

## 2. How AGENTS.md Flows Into Agent Prompts

### 2a. SessionContext Type

**File:** `packages/agent/src/events.ts` (line 71)

```typescript
readonly agentsFile: { readonly filename: string; readonly content: string } | null
```

The `agentsFile` is a required field of `SessionContext`, set once at session init and never updated except during compaction refresh.

### 2b. Prompt Rendering — `buildProjectContext()`

**File:** `packages/agent/src/prompts/session-context.ts` (lines 47-48)

```typescript
if (ctx.agentsFile) {
  content += '\n\n<agentfile filename="' + ctx.agentsFile.filename + '">\n' + ctx.agentsFile.content + '\n</agentfile>'
}
```

This is called from `buildSessionContextContent()` which produces the initial `<session_context>` block.

### 2c. Where the Session Context is Injected

1. **Root (Leader) agent**: Injected as a `session_context` entry in the window projection, at position 0. The `windowToPrompt` function renders it as a `UserMessage` via `systemEntryToMessages()`.

2. **Spawned workers**: `buildSpawnContext()` in `packages/agent/src/prompts/fork-context.ts` includes `buildProjectContext(sessionContext)` as part of the `<project-context>` block. This means all spawned workers see the root AGENTS.md.

3. **Cloned workers**: `buildCloneContext()` does **NOT** include project context at all — cloned workers inherit the leader's full conversation context directly.

### 2d. Leader System Prompt

The leader's system prompt (`packages/roles/src/prompts/leader.txt`) makes **no mention** of AGENTS.md. It's purely injected via the runtime session context message.

### 2e. Worker System Prompts

Worker role prompts use `{{WORKER_BASE}}` (from `packages/roles/src/prompts/shared/worker-base.txt`), which also makes no mention of AGENTS.md. Workers only see AGENTS.md if it comes through the `<project-context>` block from `buildSpawnContext()`.

---

## 3. No Existing Tracking Mechanism

**There is no mechanism to track which AGENTS.md files have been shown to the agent.**

- The `agentsFile` in `SessionContext` is a single, static value
- No per-turn or per-file tracking exists
- No system tracks "files already seen in context" for deduplication
- The compaction system (`CompactionProjection`) carries `refreshedContext` (a full `SessionContext | null`) but does not track individual files

---

## 4. Filesystem Tools Have No AGENTS.md Awareness

**File:** `packages/agent/src/tools/fs.ts`

The `readTool`, `writeTool`, `editTool`, `treeTool`, `grepTool`, and `viewTool` are **pure filesystem operations** with zero awareness of AGENTS.md, project context, or any tracking of the files being operated on.

- `/packages/agent/src/tools/toolkits.ts` shows all tools are defined independently from each other
- None of the tools emit events that could trigger an AGENTS.md injection
- The tool execute functions (`Effect.gen`) do not interact with the context system

---

## 5. Flow Diagram

```
Session Init
  └─ collectSessionContext()
       └─ readAgentsFile(cwd)        ← only cwd, no directory walk
            └─ returns AGENTS.md content or null

SessionInitialized event
  └─ SessionContext stored with agentsFile field

Window Projection
  └─ session_context entry (index 0)
       └─ buildSessionContextContent() → includes <agentfile>...</agentfile>

Compaction (optional)
  └─ may refresh SessionContext (refreshedContext)
       └─ new session_context entry replaces old one
```

---

## 6. Gaps vs. Desired Behavior

| Desired Feature | Current State |
|---|---|
| Scan directory hierarchy for AGENTS.md files | ❌ Only checks cwd root |
| Multiple AGENTS.md files in one project | ❌ Only one file supported |
| Auto-inject matching AGENTS.md when agent reads/edits/writes in a directory | ❌ No tool integration exists |
| Track which AGENTS.md files have been shown | ❌ No tracking mechanism |
| Avoid duplicate injections of same AGENTS.md | ❌ Not applicable |
| Refresh AGENTS.md on file change | ❌ Only refreshed via compaction (re-reading SessionContext) |

---

## 7. Key File Locations

| File | Description |
|---|---|
| `packages/agent/src/util/collect-session-context.ts` | Collects session context including agentsFile |
| `packages/agent/src/events.ts` | `SessionContext` type definition (agentsFile field) |
| `packages/agent/src/prompts/session-context.ts` | `buildProjectContext()` and `buildSessionContextContent()` — renders agentsFile into prompt |
| `packages/agent/src/prompts/fork-context.ts` | Spawn/clone context builders — spawned workers get project context |
| `packages/agent/src/tools/fs.ts` | Read/write/edit/tree/grep tools — no AGENTS.md awareness |
| `packages/agent/src/window/projection.ts` | Window projection — session_context entry handling |
| `packages/agent/src/compaction/worker.ts` | Compaction — `refreshedContext: null` never refreshed currently |
| `packages/agent/src/projections/compaction.ts` | Compaction projection state with `refreshedContext` field |
| `packages/agent/src/prompts/system-prompt-builder.ts` | System prompt builder — no AGENTS.md injection |
| `packages/roles/src/prompts/leader.txt` | Leader system prompt — no AGENTS.md awareness |
| `packages/roles/src/prompts/shared/worker-base.txt` | Worker base prompt — no AGENTS.md awareness |
