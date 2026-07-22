import { Schema } from "effect"
import { RpcClientError } from "@effect/rpc"

const { RpcClientError: TransportError } = RpcClientError

// ─── Typed protocol errors ──────────────────────────────────────────────────
// These are not all transport errors. Some are HTTP failures and others are
// violations of the RPC or subscription wire protocol. Authoritative
// lifecycle controls are successful outcomes and never enter this channel.

/** Serialization failed on the request side. */
export class RequestEncodeFailed extends Schema.TaggedError<RequestEncodeFailed>()(
  "RequestEncodeFailed",
  { message: Schema.String },
) {}

/** HTTP connection failure (refused, reset, etc). */
export class TransportRequestFailed extends Schema.TaggedError<TransportRequestFailed>()(
  "TransportRequestFailed",
  { message: Schema.String },
) {}

/** Non-2xx response from the daemon. */
export class BadResponseStatus extends Schema.TaggedError<BadResponseStatus>()(
  "BadResponseStatus",
  { status: Schema.Number },
) {}

/** Parser failed to decode server bytes. */
export class ResponseDecodeFailed extends Schema.TaggedError<ResponseDecodeFailed>()(
  "ResponseDecodeFailed",
  { message: Schema.String },
) {}

/** Server sent a message we don't understand. */
export class UnrecognizedMessage extends Schema.TaggedError<UnrecognizedMessage>()(
  "UnrecognizedMessage",
  { message: Schema.String },
) {}

/** A subscription ended without the wrapping protocol's terminal control. */
export class SubscriptionProtocolViolation extends Schema.TaggedError<SubscriptionProtocolViolation>()(
  "SubscriptionProtocolViolation",
  { message: Schema.String },
) {}

/** No data within the liveness window. */
export class StreamLivenessTimeout extends Schema.TaggedError<StreamLivenessTimeout>()(
  "StreamLivenessTimeout",
  { message: Schema.String },
) {}

/** Stream body ended before a terminal message arrived. */
export class StreamEndedWithoutExit extends Schema.TaggedError<StreamEndedWithoutExit>()(
  "StreamEndedWithoutExit",
  { message: Schema.String },
) {}

/** Recovery limit hit after consecutive failed attempts without progress. */
export class RecoveryExhausted extends Schema.TaggedError<RecoveryExhausted>()(
  "RecoveryExhausted",
  { attempts: Schema.Number },
) {}

export type JitRpcAttemptFailure =
  | RequestEncodeFailed
  | TransportRequestFailed
  | BadResponseStatus
  | ResponseDecodeFailed
  | UnrecognizedMessage
  | SubscriptionProtocolViolation
  | StreamLivenessTimeout
  | StreamEndedWithoutExit
  | RecoveryExhausted

// ─── Lifters ─────────────────────────────────────────────────────────────────

const toReason = (error: JitRpcAttemptFailure): "Protocol" | "Unknown" =>
  error instanceof TransportRequestFailed || error instanceof RecoveryExhausted
    ? "Unknown"
    : "Protocol"

/** Lift a typed attempt failure into the `RpcClientError` channel. */
export const toRpcClientError = (error: JitRpcAttemptFailure): RpcClientError.RpcClientError =>
  new TransportError({
    reason: toReason(error),
    message: error._tag,
    cause: error,
  })
