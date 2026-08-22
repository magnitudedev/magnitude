import { Schema } from "effect"
import { Group, Subscription } from "@magnitudedev/effect-query"
import { SessionError } from "../errors"
import { DisplayViewShape } from "../schemas/display"
import { StreamEvent } from "../schemas/events"

/**
 * The accepted display view of one session for one requested shape.
 *
 * Shape is the observation argument: a different shape is a different
 * subscription. Opening the subscription materializes the view (a complete
 * `state` event first, then `patch` events); reopening it rereads a complete
 * snapshot. `restore_queued_messages` is a server event addressed to the
 * composer, not display state.
 */
const StreamDisplayView = Subscription.make("StreamDisplayView", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    shape: DisplayViewShape,
  }),
  success: StreamEvent,
  error: SessionError,
  // A superseded shape's stream closes shortly after its last observer leaves.
  gcTime: "5 seconds",
})

export const Display = Group.make({ StreamDisplayView })
