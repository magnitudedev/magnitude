/**
 * Skill Tool
 *
 * skill(name) - Activate a skill by name, loading its full content into context.
 * Resolves against core skills first, then user-provided SKILL.md files.
 */

import { Effect, Context } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool, ToolErrorSchema } from '@magnitudedev/tools'
import { defineXmlBinding } from '@magnitudedev/xml-act'
import type { SkillMetadata } from '../util/skill-scanner'

// =============================================================================
// Skill State Reader Service
// =============================================================================

export interface SkillStateReader {
  readonly getUserSkills: () => Effect.Effect<readonly SkillMetadata[]>
}

export class SkillStateReaderTag extends Context.Tag('SkillStateReader')<
  SkillStateReaderTag,
  SkillStateReader
>() {}

const SkillErrorSchema = ToolErrorSchema('SkillError', {})

// =============================================================================
// skill() - Activate a skill by name
// =============================================================================

export const skillTool = defineTool({
  name: 'skill',
  group: 'default',
  description: 'Activate a user-defined skill by name to load its full methodology into context.',
  inputSchema: Schema.Struct({
    name: Schema.String.annotations({
      description: 'Name of the skill to activate'
    })
  }),
  outputSchema: Schema.String,
  errorSchema: SkillErrorSchema,

  // TODO: rewrite for new Skill type (Effect-based parseSkill, updated event shape)
  execute: (_input, _ctx) => Effect.fail({ _tag: 'SkillError' as const, message: 'Skill tool not yet implemented.' }),
  label: (input) => input.name ? `Activating ${input.name}` : 'Activating skill…',
})

export const skillXmlBinding = defineXmlBinding(skillTool, {
  input: { attributes: [{ field: 'name', attr: 'name' }] },
  output: {},
} as const)

export const SKILL_TOOLS = ['skill'] as const
