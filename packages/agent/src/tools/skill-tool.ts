/**
 * Skill Tool
 *
 * Activates a skill by name, returning its full content for context.
 */

import { Effect, Schema } from 'effect'
import { defineHarnessTool } from '@magnitudedev/harness'
import { AmbientServiceTag } from '@magnitudedev/event-core'
import { SkillsAmbient } from '../ambient/skills-ambient'
import { ToolErrorSchema } from './errors'

const SkillErrorSchema = ToolErrorSchema('SkillError', {})

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

export const skillTool = defineHarnessTool({
  definition: {
    name: 'skill',
    description: 'Activate a skill by name to load its full methodology into context. Returns the skill content — observe the result before acting on the skill\'s guidance. Skills provide detailed methodologies for specific types of work (e.g., "research", "plan", "implement").',
    inputSchema: Schema.Struct({
      name: Schema.String.annotations({ description: 'Skill name to activate (e.g., "research", "plan", "implement")' }),
    }),
    outputSchema: Schema.Struct({
      content: Schema.String.annotations({ description: 'Full skill content with all sections (shared, lead, worker, handoff)' }),
      skillPath: Schema.String.annotations({ description: 'Resolved file path of the skill' }),
    }),
  },
  errorSchema: SkillErrorSchema,
  execute: (input, _ctx) => executeSkill(input),
})
