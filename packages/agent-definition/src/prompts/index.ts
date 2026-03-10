import { actionsTagOpen, actionsTagClose, thinkTagOpen, thinkTagClose, commsTagOpen, commsTagClose, TURN_CONTROL_NEXT, TURN_CONTROL_YIELD } from '@magnitudedev/xml-act'
import xmlActProtocolRaw from './xml-act-protocol.txt'
import turnControlOrchestratorRaw from './turn-control-orchestrator.txt'
import turnControlSubagentRaw from './turn-control-subagent.txt'
import type { ThinkingLens } from '../thinking-lens'

const XML_ACT_PROTOCOL_RAW = xmlActProtocolRaw
const TURN_CONTROL_ORCHESTRATOR_RAW = turnControlOrchestratorRaw
const TURN_CONTROL_SUBAGENT_RAW = turnControlSubagentRaw

function renderThinkingLenses(lenses: ThinkingLens[]): string {
  return lenses.map((lens) => `#### ${lens.name}
> When to use: ${lens.trigger}

${lens.description}`).join('\n\n')
}

function renderLensesExample(lenses: ThinkingLens[]): string {
  return lenses
    .map((lens) => `<lens name="${lens.name}">...${lens.name} reasoning if relevant</lens>`)
    .join('\n')
}

export function getXmlActProtocol(
  defaultRecipient: string = 'user',
  lenses: ThinkingLens[],
  role: 'orchestrator' | 'subagent' = 'orchestrator',
): string {
  const turnControlSection = role === 'subagent'
    ? TURN_CONTROL_SUBAGENT_RAW
    : TURN_CONTROL_ORCHESTRATOR_RAW

  return XML_ACT_PROTOCOL_RAW
    // Inject turn control section first so its template vars get replaced by subsequent calls
    .replaceAll('{{TURN_CONTROL_SECTION}}', turnControlSection)
    .replaceAll('{{ACTIONS_OPEN}}', actionsTagOpen())
    .replaceAll('{{ACTIONS_CLOSE}}', actionsTagClose())
    .replaceAll('{{THINK_OPEN}}', thinkTagOpen())
    .replaceAll('{{THINK_CLOSE}}', thinkTagClose())
    .replaceAll('{{COMMS_OPEN}}', commsTagOpen())
    .replaceAll('{{COMMS_CLOSE}}', commsTagClose())
    .replaceAll('{{DEFAULT_RECIPIENT}}', defaultRecipient)
    .replaceAll('{{LENSES_EXAMPLE}}', renderLensesExample(lenses))
    .replaceAll('{{THINKING_LENSES}}', renderThinkingLenses(lenses))
    .replaceAll('{{TURN_CONTROL_NEXT}}', TURN_CONTROL_NEXT)
    .replaceAll('{{TURN_CONTROL_YIELD}}', TURN_CONTROL_YIELD)
}

export function buildAckTurn(lenses: ThinkingLens[]): string {
  return `<lenses>
</lenses>
<comms>
<message>I understand the response format and am ready.
</message>
</comms>
<yield/>`
}