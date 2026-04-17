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

export interface AckTurnMessage {
  role: 'user' | 'assistant'
  content: string[]
}

export function buildAckTurns(
  _lenses: ThinkingLens[],
  defaultRecipient: 'user' | 'parent' = 'user',
): AckTurnMessage[] {
  return [
    {
      role: 'user',
      content: [`--- FEW-SHOT EXAMPLE START ---
<system>
Respond using the required turn format. The user reports a bug in the login redirect.
</system>`]
    },
    {
      role: 'assistant',
      content: [`<${LENS_TAG} name="skills">Bug report → activate the bug skill to load methodology.</${LENS_TAG}>
<${LENS_TAG} name="tasks">Bug fix isn't one-turnable. Need to understand and delegate.</${LENS_TAG}>
<skill name="bug" />
<read path="src/auth/redirect.ts" />
${TURN_CONTROL_CONTINUE}`]
    },
    {
      role: 'user',
      content: [`<turn_result>
<tool name="skill"><content># Skill: Bug

Provides methodology for diagnosing and fixing bugs.

## Steps
1. Identify root cause
2. Write minimal fix
3. Test the fix</content></tool>
<tool name="read">export function redirectAfterLogout(req, res) {
  res.redirect('/home') // Bug: should redirect to '/login'
}</tool>
</turn_result>`]
    },
    {
      role: 'assistant',
      content: [`<${LENS_TAG} name="skills">Skill loaded. Bug skill says: diagnose root cause first, then fix.</${LENS_TAG}>
<${LENS_TAG} name="tasks">Create a bug task and spawn a debugger worker.</${LENS_TAG}>
<create-task id="fix-redirect" title="Fix login redirect bug" />
<spawn-worker id="fix-redirect">The redirect function is using '/home' instead of '/login'. Diagnose and fix.</spawn-worker>
<${MESSAGE_TAG} to="${defaultRecipient}">Found the bug — redirectAfterLogout sends to \`/home\` instead of \`/login\`. Worker is fixing it now.</${MESSAGE_TAG}>
${TURN_CONTROL_IDLE}`]
    },
    {
      role: 'user',
      content: ['--- FEW-SHOT EXAMPLE END ---\n\n--- CONVERSATION START ---']
    }
  ]
}
