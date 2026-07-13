import { Effect, Layer } from "effect"
import { ShadowVcs, type VcsToolEntry } from "./service"
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

const SENTINEL_OP_ID = "noop" as OperationId

function sentinelCommit(): VcsCommit {
  return {
    name: "noop",
    operationId: SENTINEL_OP_ID,
    commitHash: "noop",
    treeHash: "noop",
    timestamp: new Date(0),
    message: undefined,
    filesChanged: [],
  }
}

const emptyDelta: Delta = {
  additions: 0,
  deletions: 0,
  modifications: 0,
  renames: 0,
  files: [],
}

const sentinelRestoreResult: RestoreResult = {
  targetSnapshotId: "noop" as SnapshotId,
  undoOperationId: SENTINEL_OP_ID,
}

const sentinelUndoRedoResult: UndoRedoResult = {
  restoredOperationId: SENTINEL_OP_ID,
  undoOperationId: SENTINEL_OP_ID,
}

/**
 * No-op ShadowVcs implementation.
 *
 * All operations silently succeed with empty / sentinel values.
 * record() returns a sentinel operation ID.
 * getTools() returns an empty array.
 */
const noOpShadowVcs: typeof ShadowVcs.Service = {
  timezone: 'UTC',

  getTools: (): ReadonlyArray<VcsToolEntry> => [],

  shutdown: Effect.void,

  record: (_options?: { readonly message?: string }) =>
    Effect.succeed(SENTINEL_OP_ID),

  head: Effect.succeed(sentinelCommit()),

  resolve: (_point: PointInTime) =>
    Effect.succeed(SENTINEL_OP_ID),

  getCheckpoint: (_nameOrId: string) =>
    Effect.succeed(sentinelCommit()),

  listCheckpoints: (_options?: {
    readonly limit?: number
    readonly from?: PointInTime
    readonly to?: PointInTime
  }) => Effect.succeed([]),

  diff: (_options: {
    readonly from: PointInTime
    readonly to: PointInTime
    readonly includeFileDiffs?: boolean
    readonly pathFilter?: string
  }) => Effect.succeed(emptyDelta),

  diffWorking: (_options: {
    readonly against: PointInTime
    readonly includeFileDiffs?: boolean
    readonly pathFilter?: string
  }) => Effect.succeed(emptyDelta),

  restore: (_options: {
    readonly to: PointInTime
    readonly scope?: RestoreScope
  }) => Effect.succeed(sentinelRestoreResult),

  undo: (_options?: {
    readonly count?: number
    readonly scope?: RestoreScope
  }) => Effect.succeed(sentinelUndoRedoResult),

  redo: (_options?: {
    readonly count?: number
    readonly scope?: RestoreScope
  }) => Effect.succeed(sentinelUndoRedoResult),

  readAt: (_options: {
    readonly point: PointInTime
    readonly paths: ReadonlyArray<string>
  }) => Effect.succeed(new Map()),

  checkpoint: (_options: {
    readonly name: CheckpointName
    readonly at?: PointInTime
  }) => Effect.void,

  deleteCheckpoint: (_name: CheckpointName) => Effect.void,

  listNamedCheckpoints: () =>
    Effect.succeed([]),

  historyForPath: (_options: {
    readonly path: string
    readonly limit?: number
  }) => Effect.succeed([]),

  isClean: Effect.succeed(true),

  changedSinceHead: Effect.succeed([]),
}

/**
 * Layer that provides a no-op ShadowVcs implementation.
 *
 * Use this when VCS is disabled. All operations silently succeed;
 * getTools() returns an empty array so no checkpoint tools are exposed.
 */
export function makeNoOpVcsLayer(): Layer.Layer<ShadowVcs> {
  return Layer.succeed(ShadowVcs, noOpShadowVcs)
}
