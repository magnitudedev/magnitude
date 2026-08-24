import { Schema } from "effect"
import { Group, Mutation } from "@magnitudedev/effect-query"
import { SessionError } from "../errors"
import { RawMessageUploads, RawMentionOccurrence } from "../schemas/attachments"
import { turnAdmissionScope } from "./configuration"

const SendMessage = Mutation.make("SendMessage", {
  policy: { recovery: "AtMostOnce" },
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
  scope: () => turnAdmissionScope,
})

const StartGoal = Mutation.make("StartGoal", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({
    sessionId: Schema.String,
    objective: Schema.String,
  }),
  success: Schema.Struct({}),
  error: SessionError,
  scope: () => turnAdmissionScope,
})

export const InterruptTarget = Schema.Union(
  Schema.TaggedStruct("all", {}),
  Schema.TaggedStruct("fork", { forkId: Schema.NullOr(Schema.String) }),
)
export type InterruptTarget = Schema.Schema.Type<typeof InterruptTarget>

const Interrupt = Mutation.make("Interrupt", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({
    sessionId: Schema.String,
    target: InterruptTarget,
  }),
  success: Schema.Struct({}),
  error: SessionError,
})

export const Agent = Group.make({ SendMessage, StartGoal, Interrupt })
