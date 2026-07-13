import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { SessionError } from "../errors"
import { RawMessageAttachment } from "../schemas/attachments"

export const SendMessage = Rpc.make("SendMessage", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    messageId: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
    content: Schema.String,
    visibleMessage: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
    taskMode: Schema.Boolean,
    attachments: Schema.Array(RawMessageAttachment)
  }),
  success: Schema.Struct({}),
  error: SessionError
})

export const StartGoal = Rpc.make("StartGoal", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    objective: Schema.String
  }),
  success: Schema.Struct({}),
  error: SessionError
})

export const InterruptTarget = Schema.Union(
  Schema.TaggedStruct("all", {}),
  Schema.TaggedStruct("fork", { forkId: Schema.NullOr(Schema.String) }),
)
export type InterruptTarget = Schema.Schema.Type<typeof InterruptTarget>

export const Interrupt = Rpc.make("Interrupt", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    target: InterruptTarget,
  }),
  success: Schema.Struct({}),
  error: SessionError
})
