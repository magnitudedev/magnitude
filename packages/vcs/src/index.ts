// ── Types ──
export type {
  OperationId,
  SnapshotId,
  CheckpointName,
  PointInTime,
  RestoreScope,
  Delta,
  FileChange,
  VcsCommit,
  UndoRedoResult,
  RestoreResult,
  ShadowVcsConfig,
} from "./types"

// ── Errors ──
export {
  VcsError,
  VcsBackendError,
  OperationNotFound,
  CorruptSnapshot,
  RestoreConflict,
  InvalidPointInTime,
  type VcsFailure,
} from "./errors"

// ── Service ──
export { ShadowVcs, type ShadowVcsService, type VcsToolEntry } from "./service"

// ── Layer / Service builders ──
export { makeShadowVcsLayer, buildShadowVcs } from "./layer"
export { makeNoOpVcsLayer } from "./noop"

// ── Filesystem DI ──
export { VcsFs, VcsFsLive } from "./vcs-fs"

// ── Commit message helpers ──
export { formatCommitMessage, parseCommitMessage } from "./commit-message"

// ── Tools ──
export {
  CheckpointChangesStateSchema,
  CheckpointRollbackStateSchema,
  getVcsToolEntries,
  vcsToolkit,
  type CheckpointChangesState,
  type CheckpointRollbackState,
} from "./tools"

// ── Path selector helpers ──
export {
  createPathSelectorPredicate,
  createRestoreScopePredicate,
  filterDeltaBySelector,
  selectorToRestoreScope,
} from "./path-selector"

// ── Ref management helpers ──
export {
  nextCheckpointNumber,
  checkpointRef,
  namedCheckpointRef,
  readCheckpointRef,
  listCheckpointRefs,
  writeCheckpointRef,
  deleteCheckpointRef,
  readHead,
  updateHead,
  walkCheckpointHistory,
} from "./ref-management"
