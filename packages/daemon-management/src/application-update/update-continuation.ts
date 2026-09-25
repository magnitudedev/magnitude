import { Schema } from "effect"

/** Desktop can relaunch independently; a foreground or finite caller retains its own lifetime. */
export const UpdateContinuation = Schema.Union(
  Schema.TaggedStruct("Desktop", { showWindow: Schema.Boolean }),
  Schema.TaggedStruct("Caller", {}),
)
export type UpdateContinuation = typeof UpdateContinuation.Type
