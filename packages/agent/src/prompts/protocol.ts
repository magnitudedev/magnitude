import type { ThinkingLens } from '@magnitudedev/roles'

// Protocol constants — inlined from xml-act to eliminate the live dependency.
const P = 'magnitude:'
const YIELD_USER     = '<' + P + 'yield_user/>'
const YIELD_INVOKE   = '<' + P + 'yield_invoke/>'
const YIELD_WORKER   = '<' + P + 'yield_worker/>'
const YIELD_PARENT   = '<' + P + 'yield_parent/>'
const LEAD_YIELD_TAGS    = [P + 'yield_user', P + 'yield_invoke', P + 'yield_worker'] as const
const SUBAGENT_YIELD_TAGS = [P + 'yield_parent', P + 'yield_invoke'] as const
const TAG_THINK     = P + 'think'
const TAG_MESSAGE   = P + 'message'
const TAG_INVOKE    = P + 'invoke'
const TAG_PARAMETER = P + 'parameter'
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
    .map((lens) => `<magnitude:think about="${lens.name}">...${lens.name} thinking if relevant</magnitude:think>`)
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
    .replaceAll('{{TAG_THINK}}', 'magnitude:think')
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
    `<magnitude:think about="${lensName}">`,
    `Acknowledge readiness and continue.`,
    `</magnitude:think>`,
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
const think = (about: string, content: string) => `<${TAG_THINK} about="${about}">${content}</${TAG_THINK}>`
const msg = (to: string, ...lines: string[]) => `<${TAG_MESSAGE} to="${to}">\n${lines.join('\n')}\n</${TAG_MESSAGE}>`
const invoke = (tool: string, ...params: string[]) => `<${TAG_INVOKE} tool="${tool}">\n${params.join('\n')}\n</${TAG_INVOKE}>`
const param = (name: string, value: string) => `<${TAG_PARAMETER} name="${name}">${value}</${TAG_PARAMETER}>`
/**
 * Build the few-shot example turns for protocol demonstration.
 */
export function buildAckTurns(
  _lenses: ThinkingLens[],
  defaultRecipient: 'user' | 'parent' = 'user',
): AckTurnMessage[] {
  return [
    // Turn 1: User asks something
    {
      role: 'user',
      content: [`--- FEW-SHOT EXAMPLE START ---\n<${TAG_MESSAGE} from="user">Write an HTML guide page for foo.</${TAG_MESSAGE}>`]
    },
    // Turn 2: Activate skill, make tool call
    {
      role: 'assistant',
      content: [[
        think('turn', 'Need to write an HTML guide. Let me activate the foobar skill first.'),
        '',
        invoke('skill', param('name', 'foobar')),
        '',
        YIELD_INVOKE,
      ].join('\n')]
    },
    // Turn 3: Skill result
    {
      role: 'user',
      content: [`<turn_result>
<result tool="skill">
<out field="content">
# Skill: Foobar

Methodology for fooing bars into bazzes.

## Steps
1. Read the bar
2. Foo it
3. Output the baz
</out>
</result>
</turn_result>`]
    },
    // Turn 4: Write the HTML file with both raw tags and escaped entities
    {
      role: 'assistant',
      content: [[
        think('turn', 'Skill loaded. Write the HTML guide page for foo.'),
        '',
        invoke('write',
          param('path', 'foo-guide.html'),
          param('content', '<html>\n<body>\n<h1>Foo Guide</h1>\n<p>Links use &lt;a&gt; tags: &lt;a href="url"&gt;text&lt;/a&gt;</p>\n</body>\n</html>'),
        ),
        '',
        YIELD_INVOKE,
      ].join('\n')]
    },
    // Turn 5: Write result
    {
      role: 'user',
      content: [`<turn_result>
<result tool="write">
</result>
</turn_result>`]
    },
    // Turn 6: Read the file back
    {
      role: 'assistant',
      content: [[
        think('turn', 'File written. Read it back to verify.'),
        '',
        invoke('read', param('path', 'foo-guide.html')),
        '',
        YIELD_INVOKE,
      ].join('\n')]
    },
    // Turn 7: Read result
    {
      role: 'user',
      content: [`<turn_result>
<result tool="read">
<html>
<body>
<h1>Foo Guide</h1>
<p>Links use &lt;a&gt; tags: &lt;a href="url"&gt;text&lt;/a&gt;</p>
</body>
</html>
</result>
</turn_result>`]
    },
    // Turn 8: Confirm to user
    {
      role: 'assistant',
      content: [[
        msg(defaultRecipient, 'Done — wrote and verified foo-guide.html.'),
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
