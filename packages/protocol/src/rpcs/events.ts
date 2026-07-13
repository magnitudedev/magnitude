import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { JsonValueSchema } from "@magnitudedev/utils/schema"

export const StreamEvents = Rpc.make("StreamEvents", {
  payload: Schema.Struct({ sessionId: Schema.String }),
  success: JsonValueSchema,
  error: Schema.Never,
  stream: true
})
