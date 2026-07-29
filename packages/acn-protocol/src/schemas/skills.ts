import { Schema } from "effect"

export const SkillListEntry = Schema.Struct({
  name: Schema.String,
  description: Schema.String,
  path: Schema.String
})
export type SkillListEntry = Schema.Schema.Type<typeof SkillListEntry>

export const SkillContent = Schema.Struct({
  name: Schema.String,
  content: Schema.String
})
export type SkillContent = Schema.Schema.Type<typeof SkillContent>
