// ── Identity ─────────────────────────────────────────────────────────

/** Unique identifier for an operation (checkpoint commit). */
export type OperationId = string & { readonly _tag: "OperationId" }

/** Unique identifier for a snapshot (tree hash). */
export type SnapshotId = string & { readonly _tag: "SnapshotId" }

/** A named checkpoint bookmark. */
export type CheckpointName = string & { readonly _tag: "CheckpointName" }

// ── Point-in-Time ────────────────────────────────────────────────────

/** Resolvable points in the VCS history.
 *
 *  All kinds are resolved purely from git data.
 *
 *  turn / agent kinds are NOT here — they require event-stream
 *  knowledge and belong in the agent package's projection layer. */
/** Convenience: a bare OperationId is treated as kind="operation". */
export type PointInTime =
  | OperationId
  | { readonly kind: "operation"; readonly id: OperationId }
  | { readonly kind: "checkpoint"; readonly name: CheckpointName }
  | { readonly kind: "relative"; readonly offset: number; readonly anchor: PointInTime }
  | { readonly kind: "file"; readonly path: string; readonly position: "last-change" }
  | { readonly kind: "time"; readonly when: Date }
  | { readonly kind: "snapshot"; readonly id: SnapshotId }
  | { readonly kind: "message"; readonly value: string }

// ── Restore Scope ────────────────────────────────────────────────────

export type RestoreScope =
  | { readonly kind: "full" }
  | { readonly kind: "file"; readonly path: string }
  | { readonly kind: "directory"; readonly path: string }
  | { readonly kind: "files"; readonly paths: ReadonlyArray<string> }
  | { readonly kind: "glob"; readonly pattern: string }
  | { readonly kind: "delta-kind"; readonly kinds: ReadonlyArray<"added" | "deleted" | "modified"> }

// ── Delta / FileChange ───────────────────────────────────────────────

export interface Delta {
  readonly additions: number
  readonly deletions: number
  readonly modifications: number
  readonly renames: number
  readonly files: ReadonlyArray<FileChange>
}

export interface FileChange {
  readonly path: string
  readonly status: "added" | "deleted" | "modified" | "renamed"
  readonly oldPath?: string
  readonly diff: string
}

// ── VCS commit record (Layer 2 — no provenance, that lives in Layer 3) ──

/** Minimal checkpoint data owned by the VCS package.
 *  `message` is the commit message — an opaque string from VCS's
 *  perspective.  Agent-domain meaning (tool call ID, etc.) is
 *  joined externally in the agent projection layer. */
export interface VcsCommit {
  readonly name: string
  readonly operationId: OperationId
  readonly commitHash: string
  readonly treeHash: string
  readonly timestamp: Date
  readonly message?: string
  readonly filesChanged?: ReadonlyArray<string>
}

// ── Undo/redo result ─────────────────────────────────────────────────

export interface UndoRedoResult {
  readonly restoredOperationId: OperationId
  readonly undoOperationId: OperationId
}

// ── Restore result ───────────────────────────────────────────────────

export interface RestoreResult {
  readonly targetSnapshotId: SnapshotId
  readonly undoOperationId: OperationId
}

// ── Shadow VCS config ────────────────────────────────────────────────

export interface ShadowVcsConfig {
  readonly worktreePath: string
  readonly storagePath: string
  readonly timezone: string
}
