import { Context, Effect, Layer } from "effect"
import type { ToolkitEntry } from "@magnitudedev/harness"
import type {
  OperationId,
  SnapshotId,
  CheckpointName,
  PointInTime,
  RestoreScope,
  Delta,
  VcsCommit,
  UndoRedoResult,
  RestoreResult,
} from "./types"
import type { VcsFailure } from "./errors"

export interface VcsToolEntry {
  readonly key: 'checkpointRollback' | 'checkpointChanges'
  readonly tool: ToolkitEntry
}

/**
 * Public-facing ShadowVcs service interface.
 *
 * All operations return Effect.Effect so callers can compose them
 * with the rest of the Effect ecosystem.
 *
 * The VCS package owns only git operations and opaque commit messages.
 * turnId, agentId, toolName, and provenance enrichment
 * are agent-domain concepts — joined externally in the agent package.
 */
export interface ShadowVcs {
  // ── Lifecycle ──
  readonly shutdown: Effect.Effect<void, never>

  // ── Recording ──
  readonly record: (
    options?: { readonly message?: string },
  ) => Effect.Effect<OperationId, VcsFailure>

  // ── Head ──
  readonly head: Effect.Effect<VcsCommit, VcsFailure>

  // ── Resolution ──
  readonly resolve: (point: PointInTime) => Effect.Effect<OperationId, VcsFailure>
  readonly getCheckpoint: (nameOrId: string) => Effect.Effect<VcsCommit, VcsFailure>

  // ── Listing ──
  readonly listCheckpoints: (options?: {
    readonly limit?: number
    readonly from?: PointInTime
    readonly to?: PointInTime
  }) => Effect.Effect<ReadonlyArray<VcsCommit>, VcsFailure>

  // ── Diffs ──
  readonly diff: (options: {
    readonly from: PointInTime
    readonly to: PointInTime
    readonly includeFileDiffs?: boolean
    readonly pathFilter?: string
  }) => Effect.Effect<Delta, VcsFailure>

  readonly diffWorking: (options: {
    readonly against: PointInTime
    readonly includeFileDiffs?: boolean
    readonly pathFilter?: string
  }) => Effect.Effect<Delta, VcsFailure>

  // ── Restore ──
  readonly restore: (options: {
    readonly to: PointInTime
    readonly scope?: RestoreScope
  }) => Effect.Effect<RestoreResult, VcsFailure>

  // ── Undo / Redo ──
  readonly undo: (options?: {
    readonly count?: number
    readonly scope?: RestoreScope
  }) => Effect.Effect<UndoRedoResult, VcsFailure>
  readonly redo: (options?: {
    readonly count?: number
    readonly scope?: RestoreScope
  }) => Effect.Effect<UndoRedoResult, VcsFailure>

  // ── Read at point ──
  readonly readAt: (options: {
    readonly point: PointInTime
    readonly paths: ReadonlyArray<string>
  }) => Effect.Effect<ReadonlyMap<string, Uint8Array>, VcsFailure>

  // ── Named checkpoints ──
  readonly checkpoint: (options: {
    readonly name: CheckpointName
    readonly at?: PointInTime
  }) => Effect.Effect<void, VcsFailure>
  readonly deleteCheckpoint: (name: CheckpointName) => Effect.Effect<void, never>
  readonly listNamedCheckpoints: () => Effect.Effect<
    ReadonlyArray<{ name: string; operationId: OperationId }>,
    VcsFailure
  >

  // ── History ──
  readonly historyForPath: (options: {
    readonly path: string
    readonly limit?: number
  }) => Effect.Effect<ReadonlyArray<VcsCommit>, VcsFailure>

  // ── Introspection ──
  readonly isClean: Effect.Effect<boolean, VcsFailure>
  readonly changedSinceHead: Effect.Effect<ReadonlyArray<string>, VcsFailure>

  // ── Configuration ──
  readonly timezone: string

  // ── Tools ──
  readonly getTools: () => ReadonlyArray<VcsToolEntry>
}

export const ShadowVcs = Context.Tag("@magnitudedev/vcs/ShadowVcs")<ShadowVcs, ShadowVcs>()
export type ShadowVcsService = ShadowVcs
