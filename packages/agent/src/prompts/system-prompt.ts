import type { RoleDefinition, ToolSet } from '@magnitudedev/roles'
import { actionsTagOpen, actionsTagClose, thinkTagOpen, thinkTagClose, commsTagOpen, commsTagClose } from '@magnitudedev/xml-act'
import { PROSE_DELIM_OPEN, PROSE_DELIM_CLOSE } from '../constants'
import { getXmlActProtocol } from './protocol'
import { generateXmlActToolDocs } from '../tools/xml-tool-docs'
import toolingSectionRaw from '../agents/prompts/lead-tooling.txt' with { type: 'text' }
import subagentBaseRaw from '../agents/prompts/subagent-base.txt' with { type: 'text' }
import workspaceRaw from '../agents/prompts/workspace.txt' with { type: 'text' }

export function compilePromptTemplate(raw: string): string {
  return raw
    .replaceAll('{{PROSE_OPEN}}', PROSE_DELIM_OPEN)
    .replaceAll('{{PROSE_CLOSE}}', PROSE_DELIM_CLOSE)
    .replaceAll('{{ACTIONS_OPEN}}', actionsTagOpen())
    .replaceAll('{{ACTIONS_CLOSE}}', actionsTagClose())
    .replaceAll('{{THINK_OPEN}}', thinkTagOpen())
    .replaceAll('{{THINK_CLOSE}}', thinkTagClose())
    .replaceAll('{{COMMS_OPEN}}', commsTagOpen())
    .replaceAll('{{COMMS_CLOSE}}', commsTagClose())
    .replaceAll('{{TOOLING_SECTION}}', toolingSectionRaw)
}

function mapProtocolMode(roleDef: RoleDefinition<ToolSet, string, any>): 'lead' | 'subagent' | 'oneshot' {
  if (roleDef.protocolRole === 'oneshot-lead') return 'oneshot'
  if (roleDef.protocolRole === 'lead') return 'lead'
  return 'subagent'
}

export function renderSystemPrompt(
  roleDef: RoleDefinition<ToolSet, string, any>,
  options?: { implicitTools?: readonly string[] },
): string {
  const toolDocs = generateXmlActToolDocs(roleDef, options?.implicitTools ?? [])
  return roleDef.systemPrompt
    .replaceAll(
      '{{RESPONSE_PROTOCOL}}',
      getXmlActProtocol(roleDef.defaultRecipient, roleDef.lenses, mapProtocolMode(roleDef)),
    )
    .replaceAll('{{TOOL_DOCS}}', toolDocs)
    .replaceAll('{{SUBAGENT_BASE}}', subagentBaseRaw)
    .replaceAll('{{WORKSPACE_SECTION}}', workspaceRaw)
}
