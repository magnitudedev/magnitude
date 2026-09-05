import { Rpc } from "@effect/rpc"
import { atMostOnce } from "../transport/recovery"
import { Schema } from "effect"
import { SessionError } from "../errors"
import { RawMessageUploads, RawMentionOccurrence } from "../schemas/attachments"

const SendMessage = Rpc.make("SendMessage", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    messageId: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
    content: Schema.String,
    visibleMessage: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
    taskMode: Schema.Boolean,
    uploads: RawMessageUploads,
    mentions: Schema.Array(RawMentionOccurrence),
  }),
  success: Schema.Struct({}),
  error: SessionError,
}).pipe(atMostOnce)

const StartGoal = Rpc.make("StartGoal", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    objective: Schema.String,
  }),
  success: Schema.Struct({}),
  error: SessionError,
}).pipe(atMostOnce)

export const InterruptTarget = Schema.Union(
  Schema.TaggedStruct("all", {}),
  Schema.TaggedStruct("fork", { forkId: Schema.NullOr(Schema.String) }),
)
export type InterruptTarget = Schema.Schema.Type<typeof InterruptTarget>

const Interrupt = Rpc.make("Interrupt", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    target: InterruptTarget,
  }),
  success: Schema.Struct({}),
  error: SessionError,
}).pipe(atMostOnce)

export const Agent = {
  sendMessage: SendMessage,
  startGoal: StartGoal,
  interrupt: Interrupt,
}
