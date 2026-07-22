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
