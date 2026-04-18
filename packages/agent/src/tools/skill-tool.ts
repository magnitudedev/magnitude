/**
 * Skill Tool
 *
 * Activates a skill by name, returning its full content for context.
 * Skills provide detailed methodologies for specific types of work.
 */

import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool, ToolErrorSchema } from '@magnitudedev/tools'
import { defineXmlBinding } from '@magnitudedev/xml-act'
import { AmbientServiceTag } from '@magnitudedev/event-core'
import { SkillsAmbient } from '../ambient/skills-ambient'

const SkillErrorSchema = ToolErrorSchema('SkillError', {})

// =============================================================================
// Skill Tool
// =============================================================================

/** Execute logic for skill tool */
function executeSkill({ name }: { name: string }) {
  return Effect.gen(function* () {
    const ambientService = yield* AmbientServiceTag
    const skills = ambientService.getValue(SkillsAmbient)

    const skill = skills.get(name)
    if (!skill) {
      const available = Array.from(skills.keys()).sort().join(', ')
      return yield* Effect.fail({
        _tag: 'SkillError' as const,
        message: `Skill "${name}" not found. Available skills: ${available || 'none'}`,
      })
    }

    // Format skill content with all sections
    const sections = skill.sections
    const parts = [
      `# Skill: ${skill.name}`,
      '',
      skill.description,
      '',
      '## Shared',
      sections.shared,
      '',
      '## Lead',
      sections.lead,
      '',
      '## Worker',
      sections.worker,
      '',
      '## Handoff',
      sections.handoff,
    ]

    const content = parts.join('\n')

    return { content, skillPath: skill.path }
  })
}

export const skillTool = defineTool({
  name: 'skill' as const,
  description: 'Activate a skill by name to load its full methodology into context. Returns the skill content — observe the result before acting on the skill\'s guidance. Skills provide detailed methodologies for specific types of work (e.g., "research", "plan", "implement").',
  inputSchema: Schema.Struct({
    name: Schema.String.annotations({ description: 'Skill name to activate (e.g., "research", "plan", "implement")' }),
  }),
  outputSchema: Schema.Struct({
    content: Schema.String.annotations({ description: 'Full skill content with all sections (shared, lead, worker, handoff)' }),
    skillPath: Schema.String.annotations({ description: 'Resolved file path of the skill' }),
  }),
  errorSchema: SkillErrorSchema,
  execute: (input, _ctx) => executeSkill(input),
  label: (input) => input.name ? `Activating skill: ${input.name}` : 'Activating skill…',
})

export const skillXmlBinding = defineXmlBinding(skillTool, {
  input: {
    attributes: [{ field: 'name', attr: 'name' }],
  },
  output: {
    childTags: [{ field: 'content', tag: 'content' }, { field: 'skillPath', tag: 'skillPath' }],
  },
} as const)
