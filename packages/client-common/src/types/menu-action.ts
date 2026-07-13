import { Schema } from "effect"

export const MenuActionSchema = Schema.Union(
  Schema.TaggedStruct("new-session", {}),
  Schema.TaggedStruct("toggle-sidebar-search", {}),
  Schema.TaggedStruct("toggle-transcript-mode", {}),
  Schema.TaggedStruct("open-settings", {}),
  Schema.TaggedStruct("quit", {}),
)

export type MenuAction = Schema.Schema.Type<typeof MenuActionSchema>
