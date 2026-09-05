import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { SessionError } from "../errors"
import { DisplayViewShape } from "../schemas/display"
import { StreamEvent } from "../schemas/events"

const StreamDisplayView = Rpc.make("StreamDisplayView", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    shape: DisplayViewShape,
  }),
  success: StreamEvent,
  error: SessionError,
  stream: true,
})

export const Display = {
  streamDisplayView: StreamDisplayView,
}
