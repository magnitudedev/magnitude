import type { ThinkingLens } from '@magnitudedev/roles'
import { LENSES_CLOSE, LENSES_OPEN, TURN_CONTROL_IDLE } from '@magnitudedev/xml-act'
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
    .map((lens) => `<lens name="${lens.name}">...${lens.name} reasoning if relevant</lens>`)
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
    .replaceAll('{{THINK_OPEN}}', LENSES_OPEN)
    .replaceAll('{{THINK_CLOSE}}', LENSES_CLOSE)
    .replaceAll('{{LENSES_EXAMPLE}}', renderLensesExample(lenses))
    .replaceAll('{{THINKING_LENSES}}', renderThinkingLenses(lenses))
    .replaceAll('{{TURN_CONTROL_FINISH}}', 'finish')
    .replaceAll('{{TURN_CONTROL_IDLE}}', TURN_CONTROL_IDLE)
    .replaceAll('{{DEFAULT_RECIPIENT}}', defaultRecipient)
}

export function buildAckTurn(
  _lenses: ThinkingLens[],
  defaultRecipient: 'user' | 'parent' = 'user',
): string {
  return `${LENSES_OPEN}
${LENSES_CLOSE}
<!-- This is an example turn. I, Magnitude, did not write this and understand this assistant message exists purely to demonsrate the response format -->
<message to="${defaultRecipient}">This is how I would message the ${defaultRecipient}</message>
<message to="tutorial">This is how I would message a worker</message>
<create-task id="tutorial" type="other" title="Example task" />
<spawn-worker id="tutorial" role="explorer" />
${TURN_CONTROL_IDLE}`
}
