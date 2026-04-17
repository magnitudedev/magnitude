import type { RoleDefinition } from '@magnitudedev/roles'
import type { Skill } from '@magnitudedev/skills'

import { PROSE_DELIM_OPEN, PROSE_DELIM_CLOSE } from '../constants'
import { getXmlActProtocol } from './protocol'
import { generateXmlActToolDocs } from '../tools/xml-tool-docs'
import toolingSectionRaw from '../agents/prompts/lead-tooling.txt' with { type: 'text' }
import subagentBaseRaw from '../agents/prompts/subagent-base.txt' with { type: 'text' }
import { renderTaskTypeReferenceTable } from './tasks/index'
import fewShotNoteRaw from './protocol/few-shot-note.txt' with { type: 'text' }
//import workspaceRaw from '../agents/prompts/workspace.txt' with { type: 'text' }

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
  options?: { implicitTools?: readonly string[] },
): string {
  const toolDocs = generateXmlActToolDocs(roleDef, options?.implicitTools ?? [])
  return roleDef.systemPrompt
    .replaceAll(
      '{{RESPONSE_PROTOCOL}}',
      getXmlActProtocol(roleDef.lenses, mapProtocolMode(roleDef), roleDef.defaultRecipient),
    )
    .replaceAll('{{TOOL_DOCS}}', toolDocs)
    .replaceAll('{{SUBAGENT_BASE}}', subagentBaseRaw)
    .replaceAll(
      '{{TASK_TYPES_SECTION}}',
      skills.size > 0
        ? `## Available skills\n\nSkills provide detailed methodologies for specific types of work. Use the \`skill\` tool to activate a skill and load its full guidance into context.\n\n${renderTaskTypeReferenceTable(skills)}`
        : '',
    )
    //.replaceAll('{{WORKSPACE_SECTION}}', workspaceRaw)
    + '\n\n' + fewShotNoteRaw
}
