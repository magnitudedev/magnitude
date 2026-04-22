import type { RoleDefinition } from '@magnitudedev/roles'
import type { Skill } from '@magnitudedev/skills'

import { PROSE_DELIM_OPEN, PROSE_DELIM_CLOSE } from '../constants'
import { getProtocol } from './protocol'
import toolingSectionRaw from '../agents/prompts/lead-tooling.txt' with { type: 'text' }
import subagentBaseRaw from '../agents/prompts/subagent-base.txt' with { type: 'text' }
import { renderSkillReferenceTable } from './tasks/index'
import fewShotNoteRaw from './protocol/few-shot-note.txt' with { type: 'text' }
import type { ResolvedToolSet } from '../tools/resolved-toolset'
import { renderToolDocs } from '@magnitudedev/tools'

export function compilePromptTemplate(raw: string): string {
  return raw
    .replaceAll('{{PROSE_OPEN}}', PROSE_DELIM_OPEN)
    .replaceAll('{{PROSE_CLOSE}}', PROSE_DELIM_CLOSE)
    .replaceAll('{{TOOLING_SECTION}}', toolingSectionRaw)
}

function mapProtocolMode(roleDef: RoleDefinition): 'lead' | 'subagent' | 'oneshot' {
  if (roleDef.protocolRole === 'oneshot-lead') return 'oneshot'
  if (roleDef.protocolRole === 'lead') return 'lead'
  return 'subagent'
}

export function renderSystemPrompt(
  roleDef: RoleDefinition,
  skills: Map<string, Skill>,
  toolSet: ResolvedToolSet,
  options?: { implicitTools?: readonly string[] },
): string {
  // Render tool docs from the resolved tool set
  const availableTools = [...toolSet.availableKeys]
    .map(key => toolSet.agentDef.tools.entries[key]?.tool)
    .filter(Boolean)
  const toolDocs = availableTools.length > 0
    ? renderToolDocs(availableTools)
    : ''
  return roleDef.systemPrompt
    .replaceAll(
      '{{RESPONSE_PROTOCOL}}',
      getProtocol(roleDef.lenses, mapProtocolMode(roleDef), roleDef.defaultRecipient),
    )
    .replaceAll('{{TOOL_DOCS}}', toolDocs)
    .replaceAll('{{SUBAGENT_BASE}}', subagentBaseRaw)
    .replaceAll(
      '{{SKILLS_SECTION}}',
      skills.size > 0
        ? `## Available skills\n\nSkills provide detailed methodologies for specific types of work. Use the \`skill\` tool to activate a skill and load its full guidance into context.\n\n${renderSkillReferenceTable(skills)}`
        : '',
    )
    //.replaceAll('{{WORKSPACE_SECTION}}', workspaceRaw)
    + '\n\n' + fewShotNoteRaw
}
