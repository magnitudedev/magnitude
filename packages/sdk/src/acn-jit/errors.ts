import { Schema } from "effect";
import { Rpc, RpcClientError } from "@effect/rpc";
import { StreamDisplayView, WatchFile } from "@magnitudedev/acn-protocol";

export class DaemonDiscoveryFailed extends Schema.TaggedError<DaemonDiscoveryFailed>()(
  "DaemonDiscoveryFailed",
  {
    reason: Schema.String,
  },
) {}

export class DaemonSpawnFailed extends Schema.TaggedError<DaemonSpawnFailed>()(
  "DaemonSpawnFailed",
  {
    reason: Schema.String,
  }
) {}

export class BinaryNotFound extends Schema.TaggedError<BinaryNotFound>()(
  "BinaryNotFound",
  {
    path: Schema.String,
  }
) {}

export class BinaryVersionMismatch extends Schema.TaggedError<BinaryVersionMismatch>()(
  "BinaryVersionMismatch",
  {
    path: Schema.String,
    expected: Schema.String,
    actual: Schema.String,
  }
) {}

export class DownloadFailed extends Schema.TaggedError<DownloadFailed>()(
  "DownloadFailed",
  {
    url: Schema.String,
    status: Schema.Number,
    reason: Schema.String,
  }
) {}

export class ChecksumMismatch extends Schema.TaggedError<ChecksumMismatch>()(
  "ChecksumMismatch",
  {
    path: Schema.String,
    expected: Schema.String,
    actual: Schema.String,
  }
) {}

export class DaemonCrashed extends Schema.TaggedError<DaemonCrashed>()(
  "DaemonCrashed",
  {
    exitCode: Schema.Number,
    diagnostic: Schema.optionalWith(Schema.String, {
      as: "Option",
      exact: true,
    }),
  }
) {}

/**
 * Everything that can go wrong resolving or spawning a daemon. A schema so
 * consumers get `Schema.is(DaemonError)` as the guard instead of hand-rolled
 * `instanceof` chains.
 */
export const DaemonError = Schema.Union(
  DaemonDiscoveryFailed,
  DaemonSpawnFailed,
  BinaryNotFound,
  BinaryVersionMismatch,
  DownloadFailed,
  ChecksumMismatch,
  DaemonCrashed
);
export type DaemonError = typeof DaemonError.Type;

/**
 * Failure types for streaming RPCs — union of domain errors, RPC client
 * errors, and daemon resolution errors.
 */
export type StreamDisplayViewFailure =
  | Rpc.ErrorExit<typeof StreamDisplayView>
  | RpcClientError.RpcClientError
  | DaemonError;

export type WatchFileFailure =
  | Rpc.ErrorExit<typeof WatchFile>
  | RpcClientError.RpcClientError
  | DaemonError;
