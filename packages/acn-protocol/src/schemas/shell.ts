import { Schema } from "effect"

export const RunBashPayload = Schema.Struct({
  sessionId: Schema.String,
  command: Schema.String,
  stdin: Schema.optional(Schema.String)
})
export type RunBashPayload = Schema.Schema.Type<typeof RunBashPayload>

export const RunBashResult = Schema.Struct({
  stdout: Schema.String,
  stderr: Schema.String,
  exitCode: Schema.Number,
  cwd: Schema.String
})
export type RunBashResult = Schema.Schema.Type<typeof RunBashResult>
