import { Schema } from "effect"

/** Application updates belong to the desktop host, independently of ACN readiness. */
const Transfer = Schema.Union(
  Schema.TaggedStruct("Idle", {}),
  Schema.TaggedStruct("Available", { version: Schema.String, bytes: Schema.Number }),
  Schema.TaggedStruct("Downloading", { version: Schema.String, completed: Schema.Number, total: Schema.Number }),
  Schema.TaggedStruct("Cancelling", {}),
  Schema.TaggedStruct("Staging", { version: Schema.String }),
  Schema.TaggedStruct("Ready", { version: Schema.String }),
  Schema.TaggedStruct("Failed", { message: Schema.String }),
  Schema.TaggedStruct("Unavailable", { message: Schema.String }),
  Schema.TaggedStruct("Closed", {}),
)
export const DesktopUpdateState = Schema.Struct({
  transfer: Transfer,
  check: Schema.Union(
    Schema.TaggedStruct("Idle", {}),
    Schema.TaggedStruct("Checking", {}),
    Schema.TaggedStruct("Succeeded", { at: Schema.Number }),
    Schema.TaggedStruct("Failed", { message: Schema.String }),
  ),
  preference: Schema.Union(
    Schema.TaggedStruct("Known", { autoDownload: Schema.Boolean }),
    Schema.TaggedStruct("Unavailable", { message: Schema.String }),
  ),
})
export type DesktopUpdateState = typeof DesktopUpdateState.Type
