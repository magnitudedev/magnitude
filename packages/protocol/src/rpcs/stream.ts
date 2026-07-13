import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { SessionError } from "../errors"
import { DisplayViewShape } from "../schemas/display"
import { StreamWireEvent } from "../schemas/events"

export const StreamDisplayView = Rpc.make("StreamDisplayView", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    viewId: Schema.String,
    shape: DisplayViewShape
  }),
  success: StreamWireEvent,
  error: SessionError,
  stream: true
})

export const ResyncDisplayView = Rpc.make("ResyncDisplayView", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    viewId: Schema.String,
  }),
  success: Schema.Literal("ok"),
  error: SessionError
})

export const SetDisplayViewShape = Rpc.make("SetDisplayViewShape", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    viewId: Schema.String,
    shape: DisplayViewShape
  }),
  success: Schema.Literal("ok"),
  error: SessionError
})

export const CloseDisplayView = Rpc.make("CloseDisplayView", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    viewId: Schema.String,
  }),
  success: Schema.Literal("ok"),
  error: SessionError
})
