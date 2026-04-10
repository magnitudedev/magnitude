import type { ThinkingLens } from '@magnitudedev/roles'
import { TURN_CONTROL_IDLE, TURN_CONTROL_CONTINUE, TURN_CONTROL_IDLE_TAG, TURN_CONTROL_CONTINUE_TAG, TURN_CONTROL_FINISH_TAG, END_TURN_TAG, MESSAGE_TAG, LENS_TAG } from '@magnitudedev/xml-act'
import xmlActProtocolRaw from './protocol/xml-act-protocol.txt'
import turnControlOneshotRaw from './protocol/turn-control-oneshot.txt'
import turnControlLeadRaw from './protocol/turn-control-lead.txt'
import turnControlSubagentRaw from './protocol/turn-control-subagent.txt'
import taskRoutingLeadRaw from './protocol/task-routing-lead.txt'
import taskRoutingWorkerRaw from './protocol/task-routing-worker.txt'

const XML_ACT_PROTOCOL_RAW = xmlActProtocolRaw
const TURN_CONTROL_ONESHOT_RAW = turnControlOneshotRaw
const TURN_CONTROL_LEAD_RAW = turnControlLeadRaw
const TURN_CONTROL_SUBAGENT_RAW = turnControlSubagentRaw
const TASK_ROUTING_LEAD_RAW = taskRoutingLeadRaw
const TASK_ROUTING_WORKER_RAW = taskRoutingWorkerRaw

function renderThinkingLenses(lenses: ThinkingLens[]): string {
  return lenses.map((lens) => `#### ${lens.name}
> When to use: ${lens.trigger}

${lens.description}`).join('\n\n')
}

function renderLensesExample(lenses: ThinkingLens[]): string {
  return lenses
    .map((lens) => `<${LENS_TAG} name="${lens.name}">...${lens.name} reasoning if relevant</${LENS_TAG}>`)
    .join('\n')
}

export function getXmlActProtocol(
  lenses: ThinkingLens[],
  role: 'lead' | 'subagent' | 'oneshot' = 'lead',
  defaultRecipient: 'user' | 'parent' = 'user',
): string {
  const turnControlSection = role === 'subagent'
    ? TURN_CONTROL_SUBAGENT_RAW
    : role === 'oneshot'
    ? TURN_CONTROL_ONESHOT_RAW
    : TURN_CONTROL_LEAD_RAW
  const taskAndRoutingSection = role === 'subagent'
    ? TASK_ROUTING_WORKER_RAW
    : TASK_ROUTING_LEAD_RAW

  return XML_ACT_PROTOCOL_RAW
    .replaceAll('{{TURN_CONTROL_SECTION}}', turnControlSection)
    .replaceAll('{{TASK_AND_ROUTING_SECTION}}', taskAndRoutingSection)
    .replaceAll('{{LENSES_EXAMPLE}}', renderLensesExample(lenses))
    .replaceAll('{{THINKING_LENSES}}', renderThinkingLenses(lenses))
    .replaceAll('{{LENS_TAG}}', LENS_TAG)
    .replaceAll('{{MESSAGE_TAG}}', MESSAGE_TAG)
    .replaceAll('{{END_TURN_TAG}}', END_TURN_TAG)
    .replaceAll('{{IDLE_TAG}}', TURN_CONTROL_IDLE_TAG)
    .replaceAll('{{CONTINUE_TAG}}', TURN_CONTROL_CONTINUE_TAG)
    .replaceAll('{{TURN_CONTROL_FINISH}}', TURN_CONTROL_FINISH_TAG)
    .replaceAll('{{TURN_CONTROL_IDLE}}', TURN_CONTROL_IDLE)
    .replaceAll('{{TURN_CONTROL_CONTINUE}}', TURN_CONTROL_CONTINUE)
    .replaceAll('{{DEFAULT_RECIPIENT}}', defaultRecipient)
}

export function buildAckTurn(
  lenses: ThinkingLens[],
  defaultRecipient: 'user' | 'parent' = 'user',
): string {
  const lensName = lenses.find(lens => lens.name === 'turn')?.name ?? lenses[0]?.name ?? 'turn'

  return `<${LENS_TAG} name="${lensName}">Acknowledge readiness and continue.</${LENS_TAG}>
<${MESSAGE_TAG} to="${defaultRecipient}">Ready.</${MESSAGE_TAG}>
${TURN_CONTROL_CONTINUE}
`
}
