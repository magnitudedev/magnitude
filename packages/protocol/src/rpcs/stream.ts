import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { SessionError } from "../errors"
import { DisplayViewShape } from "../schemas/display"
import { DisplayViewStateEvent, StreamEvent } from "../schemas/events"
import { makeAcnSubscriptionRpc } from "./subscription"

export const StreamDisplayView = makeAcnSubscriptionRpc("StreamDisplayView", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    viewId: Schema.String,
    shape: DisplayViewShape,
  }),
  success: StreamEvent,
  error: SessionError,
  scope: "session",
})

export const ResyncDisplayView = Rpc.make("ResyncDisplayView", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    viewId: Schema.String,
  }),
  success: DisplayViewStateEvent,
  error: SessionError,
})

export const SetDisplayViewShape = Rpc.make("SetDisplayViewShape", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    viewId: Schema.String,
    shape: DisplayViewShape,
  }),
  success: DisplayViewStateEvent,
  error: SessionError,
})
