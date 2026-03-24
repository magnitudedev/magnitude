import { actionsTagOpen, actionsTagClose, thinkTagOpen, thinkTagClose, commsTagOpen, commsTagClose, TURN_CONTROL_FINISH, TURN_CONTROL_NEXT, TURN_CONTROL_YIELD } from '@magnitudedev/xml-act'
import xmlActProtocolRaw from './protocol/xml-act-protocol.txt'
import turnControlOneshotRaw from './protocol/turn-control-oneshot.txt'
import turnControlLeadRaw from './protocol/turn-control-lead.txt'
import turnControlSubagentRaw from './protocol/turn-control-subagent.txt'
import type { ThinkingLens } from '@magnitudedev/roles'

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
    .replaceAll('{{TURN_CONTROL_FINISH}}', TURN_CONTROL_FINISH)
}

export function buildAckTurn(_lenses: ThinkingLens[]): string {
  return `<lenses>
</lenses>
<comms>
<message>I understand the response format and am ready.
</message>
</comms>
<yield/>`
}
