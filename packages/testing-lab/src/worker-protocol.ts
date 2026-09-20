import { Schema } from "effect"
import { WorkAssignment, WorkClaim, WorkResult } from "./work-store"

/** Sent by the trusted coordinator. No provider token or general-purpose API credential belongs here. */
export const WorkerInvocation = Schema.Struct({ schemaVersion: Schema.Literal(1), assignment: WorkAssignment,
  disposable: Schema.Boolean, port: Schema.Int.pipe(Schema.between(1024, 65535)), model: Schema.NonEmptyString })
export const WorkerReply = Schema.Struct({ schemaVersion: Schema.Literal(1), claim: WorkClaim, result: WorkResult })
