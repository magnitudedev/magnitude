import { Schema } from "effect"
import { isAbsolute } from "node:path"

export const TerminalColumns = Schema.Int.pipe(Schema.between(20, 400))
export const TerminalRows = Schema.Int.pipe(Schema.between(5, 200))
export const TerminalLaunch = Schema.Struct({ executable: Schema.NonEmptyString.pipe(Schema.filter(isAbsolute)), args: Schema.Array(Schema.String),
  cwd: Schema.NonEmptyString.pipe(Schema.filter(isAbsolute)), environment: Schema.Record({ key: Schema.String, value: Schema.String }),
  columns: TerminalColumns, rows: TerminalRows })
export const TerminalStart = Schema.TaggedStruct("Start", { launch: TerminalLaunch })
export const TerminalCommand = Schema.Union(
  Schema.TaggedStruct("Write", { text: Schema.String.pipe(Schema.maxLength(65536)) }),
  Schema.TaggedStruct("Resize", { columns: TerminalColumns, rows: TerminalRows }),
  Schema.TaggedStruct("Stop", { force: Schema.Boolean }),
)
export const TerminalExit = Schema.TaggedStruct("Exited", { code: Schema.Int,
  signal: Schema.optionalWith(Schema.String, { as: "Option", exact: true }) })
export const TerminalEvent = Schema.Union(
  Schema.TaggedStruct("Started", { pid: Schema.Int.pipe(Schema.positive()) }),
  Schema.TaggedStruct("Output", { text: Schema.String }), TerminalExit,
  Schema.TaggedStruct("Failed", { message: Schema.String }),
)

export class TerminalBridgeFailure extends Schema.TaggedError<TerminalBridgeFailure>()("TerminalBridgeFailure", { message: Schema.String }) {}
