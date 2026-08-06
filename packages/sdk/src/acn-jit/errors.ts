import { Schema } from "effect"
import { Rpc, RpcClientError } from "@effect/rpc"
import { StreamDisplayView, WatchFile } from "@magnitudedev/acn-protocol"

export class AcnEnsuranceFailed extends Schema.TaggedError<AcnEnsuranceFailed>()(
  "AcnEnsuranceFailed",
  { reason: Schema.String.pipe(Schema.minLength(1)) },
) {}

export class AcnAdministrationFailed extends Schema.TaggedError<AcnAdministrationFailed>()(
  "AcnAdministrationFailed",
  { reason: Schema.String.pipe(Schema.minLength(1)) },
) {}

export class BinaryNotFound extends Schema.TaggedError<BinaryNotFound>()(
  "BinaryNotFound",
  { path: Schema.String },
) {}

export class BinaryVersionMismatch extends Schema.TaggedError<BinaryVersionMismatch>()(
  "BinaryVersionMismatch",
  {
    path: Schema.String,
    expected: Schema.String,
    actual: Schema.String,
  },
) {}

export class DownloadFailed extends Schema.TaggedError<DownloadFailed>()(
  "DownloadFailed",
  {
    url: Schema.String,
    status: Schema.Number,
    reason: Schema.String,
  },
) {}

export class ChecksumMismatch extends Schema.TaggedError<ChecksumMismatch>()(
  "ChecksumMismatch",
  {
    path: Schema.String,
    expected: Schema.String,
    actual: Schema.String,
  },
) {}

export const AcnEnsuranceError = Schema.Union(
  AcnEnsuranceFailed,
  BinaryNotFound,
  BinaryVersionMismatch,
  DownloadFailed,
  ChecksumMismatch,
)
export type AcnEnsuranceError = typeof AcnEnsuranceError.Type

export type StreamDisplayViewFailure =
  | Rpc.ErrorExit<typeof StreamDisplayView>
  | RpcClientError.RpcClientError
  | AcnEnsuranceError

export type WatchFileFailure =
  | Rpc.ErrorExit<typeof WatchFile>
  | RpcClientError.RpcClientError
  | AcnEnsuranceError
