# `@magnitudedev/vcs` — Shadow VCS

A private checkpoint system for AI agents. It creates periodic, lightweight snapshots of the worktree so the agent can safely experiment, inspect changes, undo mistakes, or roll back to any previous point in time — without touching the user's git repository.

## What it is

Most Magnitude agents work directly in the user's project directory. That creates a risk: an agent might make a series of edits, realize the approach was wrong, and need to revert cleanly. The Shadow VCS solves this by maintaining a **separate, agent-owned git repository** that checkpoints the worktree at every turn boundary and tool call boundary.

The user's `git` history is never touched. The shadow repo lives in a separate `.git` directory (typically under a `storagePath`) and tracks the same worktree via a different git database. This lets the agent freely commit, diff, and revert without corrupting the user's working state.

## Architecture overview

The package is split into three layers:

```
┌─────────────────────────────────────────────────────────────┐
│  Agent Layer (in packages/agent)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ vcs-tools.ts │  │ vcs-models.ts│  │ Cortex (turn     │   │
│  │ checkpoint_  │  │ state models │  │ boundaries)      │   │
│  │ rollback /   │  │ for UI       │  │ auto-record()    │   │
│  │ checkpoint_  │  │ rendering    │  │                  │   │
│  │ changes      │  │              │  │                  │   │
│  └──────┬───────┘  └──────┬───────┘  └───────┬──────────┘   │
└─────────┼─────────────────┼────────────────────┼────────────┘
          │                 │                    │
          └─────────────────┴────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────┐
│  Service Layer (service.ts)                                  │
│  ShadowVcs — public Effect-based interface                    │
│  record / restore / undo / redo / diff / resolve / list …     │
└───────────────────────────┬───────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────┐
│  Layer Layer (layer.ts)                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ undo / redo  │  │ resolvePoint │  │ auto-save pre-   │   │
│  │ logic & redo │  │ (all PointIn │  │ restore / pre-   │   │
│  │ stack        │  │ Time kinds)  │  │ undo commits     │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└───────────────────────────┬───────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────┐
│  Backend Layer (backend.ts + backends/just-git.ts)             │
│  Pure-JS git operations via `just-git` (zero shell execs)     │
│  buildCommit / diffTree / extractTree / walkHistory …        │
└─────────────────────────────────────────────────────────────┘
```

- **Backend** — Low-level git primitive operations backed by the `just-git` library. All methods return `Effect.Effect` and operate on git object databases, trees, and refs. There are zero `exec("git …")` calls.
- **Layer** — The service implementation (`buildShadowVcs`). Implements all user-facing semantics: undo/redo (with a per-instance redo stack), point-in-time resolution (operation, checkpoint, snapshot, relative, file, time, message), and scoped restore (full, file, directory, glob, delta-kind). Also handles **auto-save** — before any destructive operation (restore, undo, redo), the current worktree state is committed so it can be recovered.
- **Service** — A clean Effect `Context.Tag` interface (`ShadowVcs`) that the rest of the agent code consumes. All errors are typed (`VcsFailure`).

## Core concepts

### Checkpoints

A **checkpoint** is a commit in the shadow repo. Checkpoints are auto-numbered (`1`, `2`, `3`, …) and stored as `refs/checkpoints/N`. Every time the agent's turn ends (or when the agent explicitly calls `record()`), a new checkpoint is created if the worktree changed.

### Operations and Snapshots

- **OperationId** — a commit hash. Identifies a specific checkpoint (a point in time).
- **SnapshotId** — a tree hash. Identifies the *content* of the worktree at a given moment. Two checkpoints with different OperationIds can share the same SnapshotId if their content is identical.

### Point-in-time resolution

Almost all operations accept a `PointInTime` rather than a raw hash. The service resolves it to a concrete `OperationId`:

| Kind | Resolves to |
|------|-------------|
| `operation` | A specific commit hash |
| `checkpoint` | A named or numbered checkpoint ref |
| `snapshot` | The checkpoint whose tree hash matches |
| `relative` | N checkpoints forward/backward from an anchor |
| `file` | The most recent checkpoint that touched a path |
| `time` | The latest checkpoint at or before a given Date |
| `message` | A checkpoint whose commit message contains a string |

`time` is the most common in practice — the agent passes an `HH:MM:SS` timestamp and the tool resolves it to the checkpoint that existed at that moment.

### Undo / Redo

- **undo** moves HEAD to the parent commit and restores the worktree to that tree. The old HEAD is pushed onto a **redo stack**.
- **redo** pops from the redo stack, moves HEAD back, and restores the worktree.

The redo stack is scoped to a single `ShadowVcs` instance. It is cleared on every `record()` and `restore()` to prevent branching-history confusion.

### Restore (with scope)

`restore({ to: PointInTime, scope? })` brings the worktree back to a historical state. It supports scoped restoration so the agent doesn't accidentally wipe unrelated work:

- `full` — restore every tracked file
- `file` — restore one file
- `directory` — restore a folder
- `files` — restore a list of paths
- `glob` — restore paths matching a glob pattern
- `delta-kind` — restore only added / deleted / modified files

**Auto-save:** `restore` first commits the current worktree as a `pre-restore-*` checkpoint, then performs the restore, then commits a `post-restore-*` checkpoint. This means the rollback itself is fully reversible — you can roll back to the rollback's own timestamp to undo it.

## Agent integration

Two tools are exposed to the agent:

- **`checkpoint_rollback(since, glob)`** — Diff the worktree against the checkpoint at the given `since` time (an `HH:MM:SS` string), then restore only the matching files back to that state.
- **`checkpoint_changes(since, glob?)`** — Show a diff of changes since a checkpoint, scoped to an optional glob pattern.

These tools are wired into the agent harness via:

- `packages/agent/src/tools/vcs-tools.ts` — tool definitions and Effect implementations
- `packages/agent/src/models/vcs-models.ts` — state models for tool execution lifecycle
- `packages/agent/src/tools/toolkits.ts` — `vcsToolkit`, assigned to the `leader` role
- `packages/agent/src/workers/cortex.ts` — auto-records a checkpoint at every turn boundary

The VCS package itself is agnostic to agent concepts (turns, tool call IDs, workers). It stores opaque commit messages. The agent layer assigns meaning to those messages by joining the commit stream with the event stream at query time.

## Key design decisions

### Separate from the user's git

We intentionally do not use the user's `.git` directory. This prevents:
- Polluting the user's commit history with thousands of micro-checkpoints
- Accidentally creating merge conflicts or rebase hazards
- Corrupting the user's working tree state

The shadow repo is invisible to the user and can be safely discarded after a session.

### just-git (pure-JS, zero shell execs)

The backend uses `just-git`, a pure-TypeScript git implementation. This means:
- No `child_process.exec("git …")` calls, which are brittle across platforms and git versions
- Custom `FileSystem` interface allows swapping to an in-memory filesystem for fast, hermetic tests
- Full control over object database location (the shadow `.git` lives wherever we want)

#### Dependency patch: file mode normalization

We patch `just-git@1.7.0` because it mis-normalizes Node `fs.stat().mode` values. Node includes file-type bits, so a `chmod 777` regular file can be written as invalid git mode `100777`, causing `git status` or `git commit` to fail:

```text
invalid tree entry mode: '100777' for 'magnitude.js'
```

The patch masks to permission bits before choosing `100755` vs `100644`. Remove it only after an upstream fix, verified with a shadow-VCS repro that commits a `chmod 777` regular file.

### Per-session isolation

A `ShadowVcs` instance is tied to a specific `worktreePath` + `storagePath` pair. Each session gets its own git database, its own checkpoint sequence, and its own undo/redo stack. Two sessions running against the same project directory cannot interfere with each other's checkpoints.

### FileSystem as a dependency

The backend accepts a `FileSystem` handle at construction time. In production this is the real `node:fs/promises` adapter (`realFs`). In tests we pass a `MemoryFileSystem` so every test gets a fresh, isolated repo without disk I/O.
