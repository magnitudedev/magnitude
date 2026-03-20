# Spec: Migrate Artifacts to File-Based System with @-References

## Overview

Replace the artifact system with plain files in `$M/`. Artifacts become files. Artifact tools are removed. The `[[wikilink]]` system is replaced with `@path` references — a universal file reference syntax used by both users and agents. The TUI's artifact panel becomes a generic file viewer.

## Design Principles

1. **Files are the primitive.** No in-memory artifact abstraction. `$M/plan.md` is just a file.
2. **`@path` is the universal file reference.** Users, orchestrators, and subagents all use the same syntax to share file context. Mentioning `@path` injects file content into the recipient's context.
3. **Every file reference is interactive.** Any `@path` in agent output is clickable in the TUI, opening a file viewer panel.
4. **The TUI shows file activity.** Writes and edits to `$M/` paths get rich streaming preview UI.

---

## 1. @-Reference System

### Syntax
`@path` references a file. Optional section: `@path#Section`.

Examples:
- `@plan.md` — workspace file
- `@src/auth.ts` — project file
- `@plan.md#Approach` — workspace file, scroll to "Approach" section
- `@$M/plan.md` — explicit workspace reference (for rare disambiguation)

### Resolution Order
1. `$M/<path>` — workspace first
2. `<cwd>/<path>` — project second
3. Not found — no injection, note as unresolved

### Parsing — Single Parser, Two Consumers

Build a standalone `@path` text scanner. Both the agent runtime and the TUI use this same scanner.

**Scanner behavior:**
- Scans text for `@path` patterns
- Returns array of `{ path, section?, start, end }` refs
- Heuristics to avoid false positives:
  - Must look like a file path: contains `.` extension OR `/` separator
  - Not inside a code block (fenced or inline)
  - Not an email (`user@domain.com` — `@` preceded by word chars)
  - Not an npm scope (`@scope/package` — starts with `@` + word chars + `/` + word chars, no file extension)
  - Not a decorator (`@Injectable()` — followed by `(`)
- Split on first `#` → `{ path, section? }`
- Normalize path (no `..` escaping out of allowed roots)

**Two consumers of the same scanner:**
1. **Agent runtime (context injection):** scan message text → extract file refs → trigger file content injection into recipient's context
2. **TUI rendering (clickable links):** scan message text → use ref positions to render clickable interactive spans

This replaces both `extractArtifactRefs` (runtime) and `remark-wiki-link` (TUI) with one unified scanner.

**Location:** `packages/agent/src/workspace/file-refs.ts` — pure function, no dependencies. CLI already depends on `@magnitudedev/agent` so both consumers import from there.

### Context Injection
When `@path` is mentioned in:
- User messages
- Agent creation context (orchestrator → subagent instructions)
- Outbound agent messages (subagent → parent, agent → agent)

The system:
1. Resolves the file path (workspace first, then project)
2. Reads the file content (if it exists)
3. Injects into the target fork's memory as a system reminder:
   ```
   <file path="plan.md">
   ...file content...
   </file>
   ```
4. Tracks which forks have seen which files (for update notifications)

If the file doesn't exist yet, record a pending ref. When the file is later created (detected via `fs-write`/`edit` tool events), fulfill the pending ref and inject content.

### Update Notifications
When a file that has been injected into a fork's context is modified:
- Queue a system reminder with the change summary (compact diff or full content)
- Coalesce by file path to avoid spam
- Detection: hook into `fs-write` and `edit` tool completion events

### Full Circle: Same Mechanism Everywhere
- User → orchestrator: "fix the bug in @src/auth.ts"
- Orchestrator → subagent: "read @src/auth.ts and @src/middleware.ts, write your plan to @plan.md"
- Subagent → parent: "done, see @analysis.md#Findings"

The `@` reference is the universal context-sharing primitive.

---

## 2. Remove Artifact Tools

### Delete
- `packages/agent/src/tools/artifact-tools.ts` — remove entirely
- Remove artifact tool imports from all agent definitions
- Remove `artifactOrchestratorTools` and `artifactAgentTools` exports

### Remove artifact events
- Remove `artifact_changed` and `artifact_synced` from event types
- Remove `ArtifactProjection`
- Remove `ArtifactAwarenessProjection`
- Remove `ArtifactSyncWorker`

### Remove artifact persistence
- Remove `ChatPersistence.saveArtifact`
- Remove artifact storage from session storage

---

## 3. New: FileAwarenessProjection

Replaces `ArtifactAwarenessProjection`. Simpler model:

### State (per fork)
- `injectedFiles: Map<string, string>` — resolved path → content hash of what was last injected
- `pendingRefs: Map<string, Set<forkId>>` — resolved path → forks waiting for file creation

### Triggers
- On `@path` mention: resolve file, inject if exists, else pend
- On fs-write/edit completion for tracked paths: check if any forks have this file injected → emit update notification. Check pending refs → fulfill.

### Signals
- `fileFirstMentioned(forkId, path, content)` — consumed by MemoryProjection
- `fileUpdateNotification(forkId, path, notificationText)` — consumed by MemoryProjection

---

## 4. TUI Changes

### @-Reference Rendering in Messages
- Any `@path` in agent messages renders as a clickable link
- Styling: `@ path` with file icon or distinct color (like artifact chips today)
- Existence indicator: resolved files show as active links, unresolved as dimmed
- Click opens file viewer panel

### File Viewer Panel (replaces Artifact Reader Panel)
Generalized from artifact panel:
- **Props:** `filePath`, `content`, `scrollToSection?`, `streaming?`
- **Content source:** two modes:
  - **Static:** read from disk on open, re-read when a tool write/edit completes for that path
  - **Streaming:** during active `fs-write`/`edit`, content comes from DisplayProjection visual state (same as artifact streaming today). On tool completion, switch to static mode.
- **Rendering by file type:**
  - `.md` files: full markdown rendering via `StreamingMarkdownContent` (same as artifacts today)
  - Code files: syntax-highlighted via existing `lowlight` pipeline — extract `highlightFile(content, filename) → Span[][]` from current markdown code block logic (`tryHighlight` + `highlightToLines` + `hljsClassToColor`). Infer language from file extension. No new dependencies — reuses `lowlight` + theme already in `cli/src/markdown/`.
  - Unknown extensions: plain monospace text fallback
- **Section scroll:** works for markdown files (heading slug matching)
- **Streaming preview:** live content for active writes/edits (see below)
- **Header:** file path, copy button, close button
- **Navigation:** clicking `@` references within the panel opens that file

### `highlightFile` utility
Extract from existing `cli/src/markdown/blocks.ts` code block highlighting:
- Input: `content: string`, `filename: string`
- Infer lang from extension (`.ts` → typescript, `.py` → python, `.rs` → rust, etc.) using existing alias map
- If `lowlight.registered(lang)`: `lowlight.highlight(lang, content)` → `highlightToLines` → themed `Span[][]`
- If unknown: plain text lines
- Same visual style as fenced code blocks in markdown output — consistent appearance

### Streaming Preview for fs-write / edit

**Key insight:** `fs-write` and `edit` don't modify the file on disk until tool completion. During streaming, disk content is unchanged. This makes the architecture clean.

**`fs-write` streaming:**
- Visual reducer captures `path` from tool input field (streams early)
- Streams `content` body chunks progressively
- Computes char/line counts
- Panel shows live streaming content with cursor
- On tool completion → file now on disk → re-read for final state
- DisplayProjection selector: `getLatestInProgressFileStream(state, resolvedPath)`

**`edit` streaming:**
- Visual reducer captures `path` from tool input field
- Streams `old` and `new` body chunks (same pattern as artifact-update reducers)
- Panel computes optimistic preview:
  - **Base content** = read from disk (safe because file hasn't been modified yet during streaming)
  - Apply streamed old→new replacement to base → show result with highlighted changes
- On tool completion → file now modified on disk → re-read for final state

**Matching streams to the open panel:**
- DisplayProjection tracks visual state per tool step, including `path`
- Panel queries: "is there an active `fs-write` or `edit` stream where resolved `path` matches the file I'm showing?"
- If yes → streaming mode (content from visual state)
- If no → static mode (content from disk)

**Multiple edits to same file in one turn:**
- Each edit is a separate tool call with its own visual state
- Panel shows the latest active one
- When it completes, if another starts, that takes over
- Between edits, re-read from disk (which now reflects prior edits)

**Scope:** streaming preview works for ANY file the agent writes/edits (not just `$M/` paths). If the user has `@src/auth.ts` open in the panel and the agent edits it, they see the live edit.

### Tool Row Rendering
- `fs-write` to `$M/` paths: "Writing @plan.md" with live preview, clickable path
- `edit` on `$M/` paths: "Editing @plan.md" with old/new preview, clickable path
- Click on tool row opens/closes file viewer panel
- Suppress inline preview when same file is open in panel

### Fork Activity
- Replace `artifactsWritten` counter with `filesWritten`
- Track which `$M/` files were written/edited per turn
- Display as `files_written="plan.md, spec.md"` in activity summaries

---

## 5. Prompt Changes

### `subagent-base.txt`
Replace:
```
## Artifacts
You can create artifacts with `artifact-write` using any descriptive ID, and read or update artifacts shared with you via `[[artifact-id]]`.
When messaging parent, always mention artifacts using `[[artifact-id]]` so your parent gains access to it.
```
With:
```
## File Sharing
Write files to your workspace (`$M/`) using `fs-write` and share them by mentioning @filename in messages to your parent.
When you mention a file like @plan.md, the recipient gets the file content injected into their context.
You can also reference project files like @src/utils.ts to share context.
```

### Orchestrator prompts
- Remove artifact tool documentation
- Update any references to `[[artifact-id]]` syntax
- Document `@path` as the way to share files with subagents

### Workspace prompt (`workspace.txt`)
Add note about file sharing:
```
- Share workspace files with other agents by mentioning @filename in messages (e.g., @plan.md)
```

### Tool docs
- Remove artifact tool entries from tool documentation generation

---

## 6. Subagent Activity Tracking

Replace artifact-specific tracking in `SubagentActivityProjection`:
- Detect `$M/` file writes from tool completion events (instead of `artifact_changed` events)
- Emit turn entries with `filesWritten: ['plan.md', 'spec.md']` instead of `artifactsWritten`
- Memory projection formats as `files_written="plan.md, spec.md"` in activity summaries

---

## 7. Migration / Backward Compatibility

### Approach: clean break
Artifacts are session-scoped and ephemeral — no persistent data to migrate. New sessions use files, old sessions' artifacts remain in their event logs.

### Implementation Order
1. Implement `@path` parsing with false-positive heuristics
2. Implement FileAwarenessProjection + MemoryProjection integration
3. Wire `@path` rendering as clickable links in TUI messages
4. Generalize artifact panel → file viewer panel
5. Add fs-write/edit streaming preview reducers + panel wiring
6. Update prompts (remove artifact instructions, add @-reference docs)
7. Remove artifact tools, projections, events, persistence
8. Update tests

---

## 8. Files to Modify/Delete

### Delete
- `packages/agent/src/tools/artifact-tools.ts`
- `packages/agent/src/projections/artifact.ts`
- `packages/agent/src/projections/artifact-awareness.ts`
- `packages/agent/src/workers/artifact-sync-worker.ts`
- `packages/agent/tests/artifact-propagation.test.ts` (rewrite for files)

### New
- `packages/agent/src/projections/file-awareness.ts`
- `packages/agent/src/workspace/file-tracking.ts` (detect $M/ writes from tool events)
- `packages/agent/src/workspace/file-refs.ts` (`scanFileRefs` — replaces artifact-links.ts)

### Modify (agent)
- `packages/agent/src/events.ts` — remove artifact events
- `packages/agent/src/projections/memory.ts` — replace artifact signal handlers with file signal handlers
- `packages/agent/src/projections/subagent-activity.ts` — replace artifact tracking with file tracking
- `packages/agent/src/projections/display.ts` — add file stream selectors
- `packages/agent/src/visuals/tools.ts` — add file write/edit visual reducers for $M/ paths
- `packages/agent/src/agents/*.ts` — remove artifact tool imports from all agent definitions
- `packages/agent/src/agents/prompts/subagent-base.txt` — update instructions
- `packages/agent/src/agents/prompts/orchestrator*.txt` — update references
- `packages/agent/src/agents/prompts/workspace.txt` — add file sharing note
- `packages/agent/src/coding-agent.ts` — remove artifact projection/worker registration
- `packages/agent/src/prompts/agents.ts` — remove artifact attachment handling, update activity formatting

### Modify (CLI)
- `cli/src/components/artifact-reader-panel.tsx` → rename/generalize to file viewer panel
- `cli/src/markdown/block-renderer.tsx` — replace wikilink rendering with @-reference rendering
- `cli/src/markdown/parse.ts` — replace remark-wiki-link with @-reference parser
- `cli/src/markdown/blocks.ts` — handle @-reference AST nodes
- `cli/src/hooks/use-artifacts.tsx` → replace with file state hook
- `cli/src/visuals/tools.tsx` — add file write/edit renderers with streaming preview
- `cli/src/visuals/registry.ts` — register file tool renderers
- `cli/src/components/think-block.tsx` — wire file tool click handlers
- `cli/src/app.tsx` — replace artifact state with file state, update panel wiring
- `cli/src/components/inline-fork-activity.tsx` — update activity labels

### Modify (markdown)
- `cli/src/markdown/` — @-reference detection and rendering throughout the pipeline
