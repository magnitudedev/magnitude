import { Schema } from "effect"
import { JsonValueSchema } from "@magnitudedev/utils/schema"
import * as Display from "./display"

// ---------------------------------------------------------------------------
// Decoded-level patch operations
//
// Paths are arrays of string|number keys (not JSON Pointer strings).
// The server diffs decoded values directly and emits leaf-level ops.
// The client applies ops to its decoded tree with schema awareness.
// ---------------------------------------------------------------------------

const PatchPath = Schema.Array(Schema.Union(Schema.String, Schema.Number))

const PatchReplaceOp = Schema.Struct({
  op: Schema.Literal("replace"),
  path: PatchPath,
  value: JsonValueSchema
})

const PatchRemoveOp = Schema.Struct({
  op: Schema.Literal("remove"),
  path: PatchPath
})

const PatchAddOp = Schema.Struct({
  op: Schema.Literal("add"),
  path: PatchPath,
  value: JsonValueSchema
})

const PatchMoveOp = Schema.Struct({
  op: Schema.Literal("move"),
  from: PatchPath,
  to: PatchPath
})

export const DecodedPatchOp = Schema.Union(PatchReplaceOp, PatchRemoveOp, PatchAddOp, PatchMoveOp)
export type DecodedPatchOp = Schema.Schema.Type<typeof DecodedPatchOp>

export const DisplayViewStateEvent = Schema.TaggedStruct("state", {
  shape: Schema.suspend(() => Display.DisplayViewShape),
  state: Schema.suspend(() => Display.DisplayState)
})
export type DisplayViewStateEvent = Schema.Schema.Type<typeof DisplayViewStateEvent>

export const DisplayViewPatchEvent = Schema.TaggedStruct("patch", {
  ops: Schema.Array(DecodedPatchOp)
})
export type DisplayViewPatchEvent = Schema.Schema.Type<typeof DisplayViewPatchEvent>

export const DisplayViewRestoreQueuedMessagesEvent = Schema.TaggedStruct("restore_queued_messages", {
  forkId: Schema.Union(Schema.String, Schema.Null),
  messages: Schema.Array(Schema.Struct({
    id: Schema.String,
    content: Schema.String,
    taskMode: Schema.Boolean
  }))
})
export type DisplayViewRestoreQueuedMessagesEvent = Schema.Schema.Type<typeof DisplayViewRestoreQueuedMessagesEvent>

export const StreamEvent = Schema.Union(
  DisplayViewStateEvent,
  DisplayViewPatchEvent,
  DisplayViewRestoreQueuedMessagesEvent
)
export type StreamEvent = Schema.Schema.Type<typeof StreamEvent>

/**
 * Transport-level liveness heartbeat. Every long-lived server stream emits one
 * at a fixed cadence so clients can distinguish "daemon dead" from "no events".
 * The SDK filters heartbeats out before events reach consumers — `StreamEvent`
 * (without heartbeat) stays the consumer-facing type; the `*Wire*` unions are
 * what actually crosses the RPC boundary.
 */
export const StreamHeartbeat = Schema.TaggedStruct("heartbeat", {})
export type StreamHeartbeat = Schema.Schema.Type<typeof StreamHeartbeat>

export const StreamWireEvent = Schema.Union(
  DisplayViewStateEvent,
  DisplayViewPatchEvent,
  DisplayViewRestoreQueuedMessagesEvent,
  StreamHeartbeat
)
export type StreamWireEvent = Schema.Schema.Type<typeof StreamWireEvent>

/** Cadence at which the ACN emits heartbeats on open streams. */
export const STREAM_HEARTBEAT_INTERVAL_MS = 5000
/**
 * A client that sees no stream data (heartbeat or otherwise) for this long
 * treats the stream as dead and recovers. 3× the heartbeat cadence.
 */
export const STREAM_LIVENESS_TIMEOUT_MS = 15000
