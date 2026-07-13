/**
 * System prompt builder.
 *
 * Constructs the final system prompt string from a RoleDefinition,
 * skills configuration, and an optional override.
 *
 * Note: response protocol, tool docs, and few-shot note were previously
 * injected here but no shipped role prompt uses those placeholders and
 * the few-shot note was retired. Tool definitions reach the model via a
 * separate `tools` array on the chat-completions request. Dead code
 * removed 2026-06-21.
 */

import type { RoleDefinition } from '@magnitudedev/roles'
import type { Skill } from '@magnitudedev/skills'
import { renderSkillReferenceTable } from './tasks/index'
import checkpointSectionRaw from './checkpoint-section.txt' with { type: 'text' }
import skillsIntroRaw from './skills-intro.txt' with { type: 'text' }
import headlessSectionRaw from './headless-section.txt' with { type: 'text' }

/**
 * Build the complete system prompt for an agent.
 *
 * Uses the role's prompt template, injects the skills section,
 * the checkpoint section, the few-shot note, and the optional headless section.
 *
 * When `systemPromptOverride` is set, it is used verbatim in place of the
 * role's prompt template. The headless section is still appended when
 * `headless: true`. No placeholder validation is performed.
 */
export function buildSystemPrompt(opts: {
  roleDef: RoleDefinition
  skills: Map<string, Skill>
  vcsAvailable?: boolean
  headless?: boolean
  /** Override the shipped role-prompt template with raw text. */
  systemPromptOverride?: string
}): string {
  const { roleDef, skills, vcsAvailable, headless, systemPromptOverride } = opts

  const headlessSection = headless
    ? '\n\n' + headlessSectionRaw
    : ''

  const skillsSection = skills.size > 0
    ? skillsIntroRaw + '\n\n' + renderSkillReferenceTable(skills)
    : ''

  // Override path: use the provided text as the base prompt verbatim.
  // Skills section is appended to the end when skills are loaded,
  // so the agent still knows what skills are available. Checkpoint section
  // is not appended — override prompts are not expected to use the
  // checkpoint system, and the override is not a role template.
  if (systemPromptOverride !== undefined) {
    return systemPromptOverride
      + skillsSection
      + headlessSection
  }

  const checkpointSection = vcsAvailable !== false
    ? checkpointSectionRaw
    : ''

  // The roles package prompt template expects SKILLS_SECTION, THINKING_LIMIT,
  // and CHECKPOINT_SECTION as runtime vars.
  const basePrompt = roleDef.prompt.render({
    SKILLS_SECTION: skillsSection,
    THINKING_LIMIT: String(roleDef.maxThoughtChars ?? 3000),
    CHECKPOINT_SECTION: checkpointSection,
  })

  return basePrompt
    + headlessSection
}
