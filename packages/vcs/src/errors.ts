import { Data } from "effect"

export class VcsError extends Data.TaggedError("VcsError")<{
  readonly operation: string
  readonly message: string
  readonly cause?: unknown
}> {}

export class VcsBackendError extends Data.TaggedError("VcsBackendError")<{
  readonly operation: string
  readonly message: string
  readonly cause?: unknown
}> {}

export class OperationNotFound extends Data.TaggedError("OperationNotFound")<{
  readonly point: string
  readonly message: string
}> {}

export class CorruptSnapshot extends Data.TaggedError("CorruptSnapshot")<{
  readonly snapshotId: string
  readonly message: string
  readonly cause?: unknown
}> {}

export class RestoreConflict extends Data.TaggedError("RestoreConflict")<{
  readonly message: string
  readonly cause?: unknown
}> {}

export class InvalidPointInTime extends Data.TaggedError("InvalidPointInTime")<{
  readonly point: string
  readonly message: string
}> {}

/** Union of all public-facing VCS errors. */
export type VcsFailure =
  | VcsError
  | OperationNotFound
  | CorruptSnapshot
  | RestoreConflict
  | InvalidPointInTime
