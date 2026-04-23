import type { ThinkingLens } from '@magnitudedev/roles'
import { YIELD_USER, YIELD_INVOKE, YIELD_WORKER, YIELD_PARENT, LEAD_YIELD_TAGS, SUBAGENT_YIELD_TAGS } from '@magnitudedev/xml-act'
import protocolRaw from './protocol/xml-act-protocol.txt'
import turnControlOneshotRaw from './protocol/turn-control-oneshot.txt'
import turnControlLeadRaw from './protocol/turn-control-lead.txt'
import turnControlSubagentRaw from './protocol/turn-control-subagent.txt'
import taskRoutingLeadRaw from './protocol/task-routing-lead.txt'
import taskRoutingWorkerRaw from './protocol/task-routing-worker.txt'

const PROTOCOL_RAW = protocolRaw
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
    .map((lens) => `<reason about="${lens.name}">...${lens.name} reasoning if relevant</reason>`)
    .join('\n')
}

/**
 * Generate the protocol prompt for an agent.
 */
export function getProtocol(
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

  const yieldTags = role === 'subagent'
    ? SUBAGENT_YIELD_TAGS
    : LEAD_YIELD_TAGS
  const yieldOptions = yieldTags.map(t => `<${t}/>`).join(' | ')

  return PROTOCOL_RAW
    .replaceAll('{{TAG_REASON}}', 'reason')
    .replaceAll('{{TAG_MESSAGE}}', 'message')
    .replaceAll('{{TAG_INVOKE}}', 'invoke')
    .replaceAll('{{TAG_PARAMETER}}', 'parameter')
    .replaceAll('{{TAG_FILTER}}', 'filter')
    .replaceAll('{{YIELD_OPTIONS}}', yieldOptions)
    .replaceAll('{{TURN_CONTROL_SECTION}}', turnControlSection)
    .replaceAll('{{TASK_AND_ROUTING_SECTION}}', taskAndRoutingSection)
    .replaceAll('{{LENSES_EXAMPLE}}', renderLensesExample(lenses))
    .replaceAll('{{THINKING_LENSES}}', renderThinkingLenses(lenses))
    .replaceAll('{{DEFAULT_RECIPIENT}}', defaultRecipient)
    .replaceAll('{{YIELD_USER}}', YIELD_USER)
    .replaceAll('{{YIELD_INVOKE}}', YIELD_INVOKE)
    .replaceAll('{{YIELD_WORKER}}', YIELD_WORKER)
    .replaceAll('{{YIELD_PARENT}}', YIELD_PARENT)
}

/**
 * Build an acknowledgement turn for the agent to signal readiness.
 */
export function buildAckTurn(
  lenses: ThinkingLens[],
  defaultRecipient: 'user' | 'parent' = 'user',
): string {
  const lensName = lenses.find(lens => lens.name === 'turn')?.name ?? lenses[0]?.name ?? 'turn'

  return [
    `<reason about="${lensName}">`,
    `Acknowledge readiness and continue.`,
    `</reason>`,
    `<message to="${defaultRecipient}">`,
    `Ready.`,
    `</message>`,
    YIELD_INVOKE,
    '',
  ].join('\n')
}

export interface AckTurnMessage {
  role: 'user' | 'assistant'
  content: string[]
}

/**
 * Build the few-shot example turns for protocol demonstration.
 */
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
      content: [[
        '<reason about="skills">',
        'Bug report → activate the bug skill to load methodology.',
        '</reason>',
        '',
        '<reason about="tasks">',
        "Bug fix isn't one-turnable. Need to understand and delegate.",
        '</reason>',
        '',
        '<message to="user">',
        'Looking into the login redirect bug. Let me pull up the auth code and the bug methodology.',
        '</message>',
        '',
        '<invoke tool="skill">',
        '<parameter name="name">bug</parameter>',
        '</invoke>',
        '',
        '<invoke tool="read">',
        '<parameter name="path">src/auth/redirect.ts</parameter>',
        '</invoke>',
        '',
        YIELD_INVOKE,
      ].join('\n')]
    },
    {
      role: 'user',
      content: [`<turn_result>
<result tool="skill">
<out field="content">
# Skill: Bug

Provides methodology for diagnosing and fixing bugs.

## Steps
1. Identify root cause
2. Write minimal fix
3. Test the fix
</out>
</result>
<result tool="read">
export function redirectAfterLogout(req, res) {
  res.redirect('/home') // Bug: should redirect to '/login'
}
</result>
</turn_result>`]
    },
    {
      role: 'assistant',
      content: [[
        '<reason about="skills">',
        'Skill loaded. Bug skill says: diagnose root cause first, then fix.',
        '</reason>',
        '',
        '<reason about="tasks">',
        'Create a bug task and spawn a debugger worker.',
        '</reason>',
        '',
        '<invoke tool="create_task">',
        '<parameter name="id">fix-redirect</parameter>',
        '<parameter name="title">Fix login redirect bug</parameter>',
        '</invoke>',
        '',
        '<invoke tool="spawn_worker">',
        '<parameter name="id">fix-redirect</parameter>',
        "<parameter name=\"message\">The redirect function is using '/home' instead of '/login'. Diagnose and fix.</parameter>",
        '</invoke>',
        '',
        `<message to="${defaultRecipient}">`,
        'Found the bug — redirectAfterLogout sends to `/home` instead of `/login`. Worker is fixing it now.',
        '</message>',
        '',
        YIELD_USER,
      ].join('\n')]
    },
    {
      role: 'user',
      content: ['--- FEW-SHOT EXAMPLE END ---\n\n--- CONVERSATION START ---']
    }
  ]
}
