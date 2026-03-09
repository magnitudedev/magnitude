import { actionsTagOpen, actionsTagClose, thinkTagOpen, thinkTagClose, commsTagOpen, commsTagClose, TURN_CONTROL_NEXT, TURN_CONTROL_YIELD } from '@magnitudedev/xml-act'
import xmlActProtocolRaw from './xml-act-protocol.txt'
import type { ThinkingLens } from '../thinking-lens'

const XML_ACT_PROTOCOL_RAW = xmlActProtocolRaw

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

export function getXmlActProtocol(defaultRecipient: string = 'user', lenses: ThinkingLens[]): string {
  return XML_ACT_PROTOCOL_RAW
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