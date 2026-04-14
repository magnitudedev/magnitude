import { Schema } from 'effect'

export const ThinkingLensSchema = Schema.Struct({
  lens: Schema.String,
  trigger: Schema.String,
  description: Schema.String,
})
export type ThinkingLens = typeof ThinkingLensSchema.Type

export const SkillSectionsSchema = Schema.Struct({
  shared: Schema.String,
  lead: Schema.String,
  worker: Schema.String,
  handoff: Schema.String,
})
export type SkillSections = typeof SkillSectionsSchema.Type

export const SkillSchema = Schema.Struct({
  name: Schema.String,
  description: Schema.String,
  thinking: Schema.Array(ThinkingLensSchema),
  sections: SkillSectionsSchema,
})
export type Skill = typeof SkillSchema.Type
