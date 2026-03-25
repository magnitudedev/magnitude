import type { ThinkingLens } from '@magnitudedev/roles'
import {
  ACTIONS_CLOSE,
  ACTIONS_OPEN,
  COMMS_CLOSE,
  COMMS_OPEN,
  LENSES_CLOSE,
  LENSES_OPEN,
  TURN_CONTROL_YIELD,
} from '@magnitudedev/xml-act'
import xmlActProtocolRaw from './protocol/xml-act-protocol.txt'
import turnControlOneshotRaw from './protocol/turn-control-oneshot.txt'
import turnControlLeadRaw from './protocol/turn-control-lead.txt'
import turnControlSubagentRaw from './protocol/turn-control-subagent.txt'

const XML_ACT_PROTOCOL_RAW = xmlActProtocolRaw
const TURN_CONTROL_ONESHOT_RAW = turnControlOneshotRaw
const TURN_CONTROL_LEAD_RAW = turnControlLeadRaw
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
  role: 'lead' | 'subagent' | 'oneshot' = 'lead',
): string {
  const turnControlSection = role === 'subagent'
    ? TURN_CONTROL_SUBAGENT_RAW
    : role === 'oneshot'
    ? TURN_CONTROL_ONESHOT_RAW
    : TURN_CONTROL_LEAD_RAW

  return XML_ACT_PROTOCOL_RAW
    .replaceAll('{{TURN_CONTROL_SECTION}}', turnControlSection)
    .replaceAll('{{ACTIONS_OPEN}}', ACTIONS_OPEN)
    .replaceAll('{{ACTIONS_CLOSE}}', ACTIONS_CLOSE)
    .replaceAll('{{THINK_OPEN}}', LENSES_OPEN)
    .replaceAll('{{THINK_CLOSE}}', LENSES_CLOSE)
    .replaceAll('{{COMMS_OPEN}}', COMMS_OPEN)
    .replaceAll('{{COMMS_CLOSE}}', COMMS_CLOSE)
    .replaceAll('{{DEFAULT_RECIPIENT}}', defaultRecipient)
    .replaceAll('{{LENSES_EXAMPLE}}', renderLensesExample(lenses))
    .replaceAll('{{THINKING_LENSES}}', renderThinkingLenses(lenses))
    .replaceAll('{{TURN_CONTROL_NEXT}}', 'next')
    .replaceAll('{{TURN_CONTROL_YIELD}}', 'yield')
    .replaceAll('{{TURN_CONTROL_FINISH}}', 'finish')
}

export function buildAckTurn(_lenses: ThinkingLens[]): string {
  return `${LENSES_OPEN}
${LENSES_CLOSE}
${COMMS_OPEN}
<message>I understand the response format and am ready.
</message>
${COMMS_CLOSE}
${TURN_CONTROL_YIELD}`
}
