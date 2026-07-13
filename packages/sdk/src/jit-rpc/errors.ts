import { Schema } from "effect"
import { RpcClientError } from "@effect/rpc"

const { RpcClientError: TransportError } = RpcClientError

// ─── Typed protocol errors ──────────────────────────────────────────────────
// Each distinct failure mode in the recovering protocol gets its own
// `Schema.TaggedError`. These flow as the `cause` inside `RpcClientError`
// (the Effect RPC channel type), so callers can distinguish failure modes
// via `error.cause instanceof TransportRequestFailed`, etc.

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

/** Clean/interrupt exit on a resident stream — daemon relinquished it. */
export class ResidentStreamRelinquished extends Schema.TaggedError<ResidentStreamRelinquished>()(
  "ResidentStreamRelinquished",
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

/** Retry limit hit — too many consecutive failures without progress. */
export class TransportExhausted extends Schema.TaggedError<TransportExhausted>()(
  "TransportExhausted",
  { attempts: Schema.Number },
) {}

export type JitRpcTransportError =
  | RequestEncodeFailed
  | TransportRequestFailed
  | BadResponseStatus
  | ResponseDecodeFailed
  | UnrecognizedMessage
  | ResidentStreamRelinquished
  | StreamLivenessTimeout
  | StreamEndedWithoutExit
  | TransportExhausted

// ─── Lifters ─────────────────────────────────────────────────────────────────

const toReason = (error: JitRpcTransportError): "Protocol" | "Unknown" =>
  error instanceof TransportRequestFailed || error instanceof TransportExhausted
    ? "Unknown"
    : "Protocol"

/** Lift a typed protocol error into the `RpcClientError` channel. */
export const toRpcClientError = (error: JitRpcTransportError): RpcClientError.RpcClientError =>
  new TransportError({
    reason: toReason(error),
    message: error._tag,
    cause: error,
  })
