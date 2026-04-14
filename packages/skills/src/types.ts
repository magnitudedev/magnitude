import { Schema } from 'effect'

export interface SkillsetInfo {
  name: string
  path: string
  scope: 'project' | 'global'
}



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

export const ParsedSkillSchema = Schema.Struct({
  name: Schema.String,
  description: Schema.String,
  thinking: Schema.Array(ThinkingLensSchema),
  sections: SkillSectionsSchema,
})
export type ParsedSkill = typeof ParsedSkillSchema.Type

export const SkillSchema = Schema.extend(ParsedSkillSchema, Schema.Struct({
  path: Schema.String,
}))
export type Skill = typeof SkillSchema.Type

export const SkillsetSchema = Schema.Struct({
  path: Schema.String,
  content: Schema.String,
  skills: Schema.Record({ key: Schema.String, value: SkillSchema }),
})
export type Skillset = typeof SkillsetSchema.Type

export const EmptySkillset: Skillset = {
  path: '',
  content: '',
  skills: {},
}
