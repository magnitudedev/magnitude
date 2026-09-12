import { Schema } from "effect"

/** Application updates belong to the desktop host, independently of ACN readiness. */
export const DesktopUpdateState = Schema.Union(
  Schema.TaggedStruct("Idle", {}),
  Schema.TaggedStruct("Checking", {}),
  Schema.TaggedStruct("Current", {}),
  Schema.TaggedStruct("Available", { version: Schema.String, bytes: Schema.Number }),
  Schema.TaggedStruct("Downloading", { version: Schema.String, completed: Schema.Number, total: Schema.Number }),
  Schema.TaggedStruct("Staging", { version: Schema.String }),
  Schema.TaggedStruct("Ready", { version: Schema.String }),
  Schema.TaggedStruct("Failed", { message: Schema.String }),
  Schema.TaggedStruct("Unavailable", { message: Schema.String }),
  Schema.TaggedStruct("Closed", {}),
)
export type DesktopUpdateState = typeof DesktopUpdateState.Type
