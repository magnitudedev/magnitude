import { Effect } from "effect"
import type { Delta, RestoreScope } from "./types"
import { VcsBackendError } from "./errors"

/**
 * Internal contract implemented by a git backend.
 * This IS the Effect/Promise boundary — implementations wrap
 * async git operations into Effects here, so the service layer
 * composes Effects directly without `Effect.tryPromise` noise.
 */
export interface VcsBackend {
  /**
   * Build a commit from the current worktree state.
   * Stages worktree changes via the index, creates tree + commit, advances branch ref.
   * Returns the new commit hash.
   *
   * NOTE: `buildCommit` always creates a commit even if there are no changes.
   * Call `getChangedFiles` first to detect no-op.
   */
  readonly buildCommit: (options: {
    branch?: string
    message: string
  }) => Effect.Effect<string, VcsBackendError>

  /**
   * Detect changed / new / deleted paths in the worktree vs HEAD.
   * Uses git status --porcelain (stat-based, fast).
   * Returns structural status only (no line-level diff).
   */
  readonly getChangedFiles: () => Effect.Effect<
    ReadonlyArray<{ path: string; status: "added" | "deleted" | "modified" }>,
    VcsBackendError
  >

  /** Diff two trees (or commits). Returns structural deltas. */
  readonly diffTree: (
    fromTreeHash: string,
    toTreeHash: string,
  ) => Effect.Effect<Delta, VcsBackendError>

  /**
   * Materialize a tree into the working directory, optionally scoped.
   * Writes files from the tree to disk. Does NOT delete untracked files —
   * caller must do that if needed.
   * Returns list of restored paths.
   */
  readonly extractTree: (
    treeHash: string,
    scope?: RestoreScope,
  ) => Effect.Effect<ReadonlyArray<string>, VcsBackendError>

  /** Read raw file bytes from a tree at a given path. */
  readonly readFileAt: (treeHash: string, path: string) => Effect.Effect<Uint8Array | null, VcsBackendError>

  /** Walk a tree recursively, returning all entries. */
  readonly walkTree: (treeHash: string) => Effect.Effect<
    ReadonlyArray<{ path: string; hash: string; mode: string }>,
    VcsBackendError
  >

  /** Resolve a ref name to a commit hash. */
  readonly readRef: (ref: string) => Effect.Effect<string | null, VcsBackendError>

  /** Update (or create) a ref to point at a commit. */
  readonly updateRef: (ref: string, commitHash: string) => Effect.Effect<void, VcsBackendError>

  /** Delete a ref. */
  readonly deleteRef: (ref: string) => Effect.Effect<void, VcsBackendError>

  /** List refs under a prefix. */
  readonly listRefs: (prefix: string) => Effect.Effect<
    ReadonlyArray<{ ref: string; hash: string }>,
    VcsBackendError
  >

  /** Read HEAD (handles symbolic, direct, unborn). */
  readonly readHead: () => Effect.Effect<
    { kind: "symbolic"; target: string } | { kind: "direct"; hash: string } | { kind: "unborn" },
    VcsBackendError
  >

  /** Walk commits backward from a start hash. */
  readonly walkHistory: (options?: {
    start?: string
    limit?: number
    pathFilter?: string
  }) => Effect.Effect<
    ReadonlyArray<{
      hash: string
      message: string
      tree: string
      parents: ReadonlyArray<string>
      author: { name: string; email: string; timestamp: number; timezone: string }
      committer: { name: string; email: string; timestamp: number; timezone: string }
    }>,
    VcsBackendError
  >

  /** Read a file from the worktree (not from git history). */
  readonly readWorktreeFile: (relativePath: string) => Effect.Effect<Uint8Array | null, VcsBackendError>

  /** Clean up resources (no-op for just-git). */
  readonly dispose: () => Effect.Effect<void, never>
}
