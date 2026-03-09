/**
 * Skill Tool
 *
 * skill(name) - Activate a skill by name, loading its full content into context.
 * Resolves against core skills first, then user-provided SKILL.md files.
 */

import { Effect, Context } from 'effect'
import { Schema } from '@effect/schema'
import { createTool, ToolErrorSchema } from '@magnitudedev/tools'
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

// =============================================================================
// Errors
// =============================================================================

const SkillError = ToolErrorSchema('SkillError', {})

// =============================================================================
// skill() - Activate a skill by name
// =============================================================================

export const skillTool = createTool({
  name: 'skill',
  group: 'default',
  description: 'Activate a skill by name to load its full methodology into context. Use when a task clearly matches a skill type (feature, bug, refactor) or a project-specific skill.',
  inputSchema: Schema.Struct({
    name: Schema.String.annotations({
      description: 'Name of the skill to activate (e.g., "feature", "bug", "refactor")'
    })
  }),
  outputSchema: Schema.String,
  errorSchema: SkillError,
  argMapping: ['name'],
  bindings: {
    xmlInput: { type: 'tag', attributes: ['name'], selfClosing: true },
    xmlOutput: { type: 'tag' as const },
  } as const,

  execute: (input) => Effect.gen(function* () {
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
})

export const SKILL_TOOLS = ['skill'] as const
