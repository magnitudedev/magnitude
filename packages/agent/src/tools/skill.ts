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
import { resolveSkill } from '../skills'
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
  description: 'Activate a skill by name to load its full methodology into context. Use when a task clearly matches a skill type (feature, bug, refactor) or a project-specific skill.',
  inputSchema: Schema.Struct({
    name: Schema.String.annotations({
      description: 'Name of the skill to activate (e.g., "feature", "bug", "refactor")'
    })
  }),
  outputSchema: Schema.String,
  errorSchema: SkillErrorSchema,

  execute: (input, _ctx) => Effect.gen(function* () {
    const skillReader = yield* SkillStateReaderTag
    const userSkills = yield* skillReader.getUserSkills()

    const resolved = yield* Effect.tryPromise({
      try: () => resolveSkill(input.name, userSkills),
      catch: (e) => ({ _tag: 'SkillError' as const, message: e instanceof Error ? e.message : String(e) }),
    })

    if (!resolved) {
      return yield* Effect.fail({
        _tag: 'SkillError' as const,
        message: `Skill "${input.name}" not found. Check available skills in your session context.`
      })
    }

    return `<skill_activated name="${resolved.name}">\n${resolved.content}\n</skill_activated>`
  }),
  label: (input) => input.name ? `Activating ${input.name}` : 'Activating skill…',
})

export const skillXmlBinding = defineXmlBinding(skillTool, {
  input: { attributes: [{ field: 'name', attr: 'name' }] },
  output: {},
} as const)

export const SKILL_TOOLS = ['skill'] as const
