import type { ThinkingLens } from '@magnitudedev/roles'
import { YIELD_USER, YIELD_INVOKE, YIELD_WORKER, YIELD_PARENT, LEAD_YIELD_TAGS, SUBAGENT_YIELD_TAGS, TAG_REASON, TAG_MESSAGE, TAG_INVOKE, TAG_PARAMETER, TAG_ESCAPE } from '@magnitudedev/xml-act'
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
    .map((lens) => `<magnitude:reason about="${lens.name}">...${lens.name} reasoning if relevant</magnitude:reason>`)
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
    .replaceAll('{{TAG_REASON}}', 'magnitude:reason')
    .replaceAll('{{TAG_MESSAGE}}', 'magnitude:message')
    .replaceAll('{{TAG_INVOKE}}', 'magnitude:invoke')
    .replaceAll('{{TAG_PARAMETER}}', 'magnitude:parameter')
    .replaceAll('{{TAG_FILTER}}', 'magnitude:filter')
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
    `<magnitude:reason about="${lensName}">`,
    `Acknowledge readiness and continue.`,
    `</magnitude:reason>`,
    `<magnitude:message to="${defaultRecipient}">`,
    `Ready.`,
    `</magnitude:message>`,
    YIELD_INVOKE,
    '',
  ].join('\n')
}

export interface AckTurnMessage {
  role: 'user' | 'assistant'
  content: string[]
}

// Tag helpers for few-shot construction
const reason = (about: string, content: string) => `<${TAG_REASON} about="${about}">${content}</${TAG_REASON}>`
const msg = (to: string, ...lines: string[]) => `<${TAG_MESSAGE} to="${to}">\n${lines.join('\n')}\n</${TAG_MESSAGE}>`
const invoke = (tool: string, ...params: string[]) => `<${TAG_INVOKE} tool="${tool}">\n${params.join('\n')}\n</${TAG_INVOKE}>`
const param = (name: string, value: string) => `<${TAG_PARAMETER} name="${name}">${value}</${TAG_PARAMETER}>`
const esc = (content: string) => `<${TAG_ESCAPE}>${content}</${TAG_ESCAPE}>`

/**
 * Build the few-shot example turns for protocol demonstration.
 */
export function buildAckTurns(
  _lenses: ThinkingLens[],
  defaultRecipient: 'user' | 'parent' = 'user',
): AckTurnMessage[] {
  return [
    // Turn 1: User asks about the response format
    {
      role: 'user',
      content: [`--- FEW-SHOT EXAMPLE START ---\n<${TAG_MESSAGE} from="user">How does your response format work?</${TAG_MESSAGE}>`]
    },
    // Turn 2: Explain format using escape blocks to show tag examples
    {
      role: 'assistant',
      content: [[
        reason('alignment', 'Format question — explain with examples using escape blocks.'),
        '',
        msg(defaultRecipient,
          'I respond in a structured format with these elements:',
          '',
          `- **Reasons** for internal thinking: ${esc(`\`<${TAG_REASON} about="topic">thinking</${TAG_REASON}>\``)}`,
          `- **Messages** to communicate: ${esc(`\`<${TAG_MESSAGE} to="user">content</${TAG_MESSAGE}>\``)}`,
          `- **Tool calls** to take action: ${esc(`\`<${TAG_INVOKE} tool="name"><${TAG_PARAMETER} name="key">value</${TAG_PARAMETER}></${TAG_INVOKE}>\``)}`,
          '- **Yield** to end my turn and wait',
          '',
          'I think first, then communicate and act, then yield.',
        ),
        '',
        YIELD_USER,
      ].join('\n')]
    },
    // Turn 3: User asks to write a reference doc
    {
      role: 'user',
      content: [`<${TAG_MESSAGE} from="user">Write that up as a quick reference doc.</${TAG_MESSAGE}>`]
    },
    // Turn 4: Activate skill
    {
      role: 'assistant',
      content: [[
        reason('skills', 'Clear objective — activate the example skill.'),
        '',
        msg(defaultRecipient, 'On it — let me pull up the methodology first.'),
        '',
        invoke('skill', param('name', 'example')),
        '',
        YIELD_INVOKE,
      ].join('\n')]
    },
    // Turn 5: Skill results
    {
      role: 'user',
      content: [`<turn_result>
<result tool="skill">
<out field="content">
# Skill: Example

Provides methodology for the example task.

## Steps
1. Understand the objective
2. Execute precisely
3. Verify the result
</out>
</result>
</turn_result>`]
    },
    // Turn 6: Write doc with escape in parameter value + message
    {
      role: 'assistant',
      content: [[
        reason('skills', 'Skill loaded. Write the doc to $M.'),
        '',
        invoke('write',
          param('path', '$M/reports/response-format.md'),
          param('content', [
            '# Response Format Quick Reference',
            '',
            '## Reasons',
            esc(`\`<${TAG_REASON} about="topic">internal thinking</${TAG_REASON}>\``),
            '',
            '## Messages',
            esc(`\`<${TAG_MESSAGE} to="recipient">content</${TAG_MESSAGE}>\``),
            '',
            '## Tool Calls',
            esc([
              '```',
              `<${TAG_INVOKE} tool="name">`,
              `<${TAG_PARAMETER} name="key">value</${TAG_PARAMETER}>`,
              `</${TAG_INVOKE}>`,
              '```',
            ].join('\n')),
            '',
            '## Yield',
            'End each turn with a yield tag to wait for results or user input.',
          ].join('\n')),
        ),
        '',
        msg(defaultRecipient, 'Done — written to [$M/reports/response-format.md]($M/reports/response-format.md).'),
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
