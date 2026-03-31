import { describe, expect, it } from 'bun:test'
import { Effect, Stream } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool, type ToolDefinition } from '@magnitudedev/tools'
import {
  createXmlRuntime,
  type RegisteredTool,
  type XmlRuntimeConfig,
  type XmlRuntimeEvent,
  type XmlTagBinding,
} from '../index'
import {
  ACTIONS_CLOSE,
  ACTIONS_OPEN,
  AGENT_CREATE_TAG,
  COMMS_CLOSE,
  COMMS_OPEN,
  LENSES_CLOSE,
  LENSES_OPEN,
  MESSAGE_CLOSE,
  MESSAGE_OPEN,
  MESSAGE_TAG,
  TITLE_CLOSE,
  TITLE_OPEN,
  TITLE_TAG,
  TURN_CONTROL_NEXT,
  agentCreateOpen,
  lensClose,
  lensOpen,
  messageClose,
  messageOpen,
} from '../constants'

function registered(
  tool: ToolDefinition,
  tagName: string,
  binding: XmlTagBinding,
): RegisteredTool {
  return {
    tool,
    tagName,
    groupName: 'default',
    binding,
  }
}

function config(tools: RegisteredTool[]): XmlRuntimeConfig {
  return {
    tools: new Map(tools.map(t => [t.tagName, t])),
  }
}

function eventsOfType<T extends XmlRuntimeEvent['_tag']>(
  events: XmlRuntimeEvent[],
  tag: T,
): Extract<XmlRuntimeEvent, { _tag: T }>[] {
  return events.filter(e => e._tag === tag) as Extract<XmlRuntimeEvent, { _tag: T }>[]
}

describe('behavior: action tool call executes as a tool lifecycle (not prose/comms)', () => {
  const agentCreateTool = defineTool({
    name: 'agent_create',
    description: 'create agent',
    inputSchema: Schema.Struct({
      id: Schema.String,
      type: Schema.String,
      title: Schema.String,
      message: Schema.String,
    }),
    outputSchema: Schema.Struct({ ok: Schema.Boolean }),
    execute: () => Effect.succeed({ ok: true }),
    label: ({ id }) => `agent_create:${id}`,
  })

  const binding: XmlTagBinding = {
    tag: AGENT_CREATE_TAG,
    attributes: [
      { field: 'id', attr: 'id' },
      { field: 'type', attr: 'type' },
    ],
    childTags: [
      { field: 'title', tag: TITLE_TAG },
      { field: 'message', tag: MESSAGE_TAG },
    ],
  }

  const xml = `${LENSES_OPEN}
${lensOpen('intent')}User wants research first, then a plan. I'll have Explorer research best stacks for todo apps, then use Planner to create a concrete plan.${lensClose()}
${lensOpen('workflow')}Explorer research → Planner creates plan. User approves plan → Builder implements.${lensClose()}
${lensOpen('turn')}Starting with Explorer to gather stack research.${lensClose()}
${LENSES_CLOSE}
${COMMS_OPEN}
${messageOpen('user')}Got it — researching best stacks for todo apps now. I'll look at popular options across different platforms and bring back findings for planning.${messageClose()}
${COMMS_CLOSE}
${ACTIONS_OPEN}
${agentCreateOpen({ id: 'explorer-todo-stack', type: 'explorer' }, { multiline: true })}
${TITLE_OPEN}Research best tech stacks for todo apps${TITLE_CLOSE}
${MESSAGE_OPEN}Research the best technology stacks for building a todo app in 2024-2025. Consider:

1. **Frontend frameworks** - React, Vue, Svelte, Next.js, Remix, etc.
2. **Backend options** - Node.js, Supabase, Firebase, PocketBase, or serverless
3. **Database choices** - PostgreSQL, SQLite, MongoDB, or simple JSON/local storage
4. **State management** - Redux, Zustand, Jotai, React Query, etc.

For each option, consider:
- Developer experience and learning curve
- Performance and scalability
- Cost (free tier availability)
- Ease of deployment
- Real-world examples of todo apps built with that stack

Provide a concise comparison with recommendations for different scenarios (solo dev, quick MVP, production-grade, etc.).${MESSAGE_CLOSE}
</agent-create>
${ACTIONS_CLOSE}
${TURN_CONTROL_NEXT}
`

  it('executes the action tool call and emits full tool lifecycle', async () => {
    const runtime = createXmlRuntime(config([registered(agentCreateTool, AGENT_CREATE_TAG, binding)]))
    const events = await Effect.runPromise(Stream.runCollect(runtime.streamWith(Stream.make(xml)))).then(c => Array.from(c))

    expect(eventsOfType(events, 'ToolInputStarted')).toHaveLength(1)
    expect(eventsOfType(events, 'ToolInputReady')).toHaveLength(1)
    expect(eventsOfType(events, 'ToolExecutionStarted')).toHaveLength(1)
    expect(eventsOfType(events, 'ToolExecutionEnded')).toHaveLength(1)
    expect(eventsOfType(events, 'ToolInputParseError')).toHaveLength(0)
  })
})
