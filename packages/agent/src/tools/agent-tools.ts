/**
 * Agent Tools
 *
 * Tool group for managing agents (workers).
 * Agents are WHO does the work — separate from tasks (WHAT needs doing).
 * Context flows through message-based communication.
 *
 * Tools: create
 */

import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { createTool, ToolErrorSchema } from '@magnitudedev/tools'
import { Fork, WorkerBusTag } from '@magnitudedev/event-core'
import type { AgentVariant } from '../agents'


import { ConversationStateReaderTag } from './memory-reader'
import { AgentStateReaderTag } from './fork'
import { buildAgentContext, buildConversationSummary } from '../prompts'
import type { AppEvent } from '../events'
import { getActiveAgent } from '../projections/agent-status'

const { ForkContext } = Fork

// =============================================================================
// Errors
// =============================================================================

const AgentError = ToolErrorSchema('AgentError', {})

// =============================================================================
// agent.create — Create and dispatch an agent
// =============================================================================

/** Execute logic for agent.create */
function executeAgentCreate({ agentId, options }: {
  agentId: string;
  options: { type: AgentVariant; title: string; message: string };
}) {
  return Effect.gen(function* () {
    const { type: agentType, title, message } = options

    // For reviewer agents, inject user↔orchestrator conversation context
    let conversationContext = ''
    if (agentType === 'reviewer') {
      const conversationReader = yield* ConversationStateReaderTag
      const conversationState = yield* conversationReader.getState()
      const summary = buildConversationSummary(conversationState.entries)
      if (summary) {
        conversationContext = summary
      }
    }

    // Build context from title + message + conversation
    const context = buildAgentContext(title, message, conversationContext)

    const { ExecutionManager } = yield* Effect.tryPromise({
      try: () => import('../execution/execution-manager'),
      catch: (e) => ({
        _tag: 'AgentError' as const,
        message: e instanceof Error ? e.message : String(e),
      }),
    })
    const execManager = yield* ExecutionManager
    const { forkId: parentForkId } = yield* ForkContext
    // Use a synthetic task ID for the fork (agents no longer require pre-existing tasks)
    const taskId = `agent-${agentId}`

    const forkId = yield* execManager.fork({
      parentForkId,
      name: title,
      agentId,
      prompt: context,
      message,
      mode: 'spawn',
      role: agentType,
      taskId,
    })

    return { agentId, forkId }
  })
}


export const agentCreateTool = createTool({
  name: 'create' as const,
  group: 'agent' as const,
  description: 'Create a new agent and dispatch it with a title and message.',
  inputSchema: Schema.Struct({
    agentId: Schema.String.annotations({ description: 'Unique agent ID. Must be prefixed with the type (e.g. explorer-auth, builder-api)' }),
    options: Schema.Struct({
      type: Schema.Literal('explorer', 'planner', 'builder', 'debugger', 'reviewer', 'browser').annotations({ description: 'Agent type' }),
      title: Schema.String.annotations({ description: 'Concise title of what this agent should accomplish' }),
      message: Schema.String.annotations({ description: 'Detailed message/instructions for the agent' }),
    }),
  }),
  outputSchema: Schema.Struct({ agentId: Schema.String, forkId: Schema.String }),
  errorSchema: AgentError,
  argMapping: ['agentId', 'options'] as const,
  bindings: {
    xmlInput: {
      type: 'tag' as const,
      attributes: [
        { field: 'agentId', attr: 'agentId' },
        { field: 'options.type', attr: 'type' },
      ],
      childTags: [
        { field: 'options.type', tag: 'type' },
        { field: 'options.title', tag: 'title' },
        { field: 'options.message', tag: 'message' },
      ],
    },
    xmlOutput: { type: 'tag' as const, childTags: [{ field: 'agentId', tag: 'agentId' }, { field: 'forkId', tag: 'forkId' }] },
  } as const,
  execute: executeAgentCreate,
})

export const agentKillTool = createTool({
  name: 'kill' as const,
  group: 'agent' as const,
  description: 'Kill an active subagent that was started accidentally or no longer needed. Not meant for idle subagents.',
  inputSchema: Schema.Struct({
    agentId: Schema.String.annotations({ description: 'Direct-child subagent ID to kill' }),
    reason: Schema.optional(Schema.String.annotations({ description: 'Optional reason for the kill event record' })),
  }),
  outputSchema: Schema.Struct({ agentId: Schema.String, forkId: Schema.String }),
  errorSchema: AgentError,
  argMapping: ['agentId', 'reason'] as const,
  bindings: {
    xmlInput: {
      type: 'tag' as const,
      attributes: [{ field: 'agentId', attr: 'agentId' }],
      childTags: [{ field: 'reason', tag: 'reason' }],
    },
    xmlOutput: { type: 'tag' as const, childTags: [{ field: 'agentId', tag: 'agentId' }, { field: 'forkId', tag: 'forkId' }] },
  } as const,
  execute: ({ agentId, reason }) => Effect.gen(function* () {
    const { forkId: parentForkId } = yield* ForkContext
    const agentStateReader = yield* AgentStateReaderTag
    const agentState = yield* agentStateReader.getAgentState()
    const target = getActiveAgent(agentState, agentId)

    if (!target) {
      return yield* Effect.fail({
        _tag: 'AgentError' as const,
        message: `Cannot kill unknown subagent "${agentId}".`,
      })
    }

    if (target.parentForkId !== parentForkId) {
      return yield* Effect.fail({
        _tag: 'AgentError' as const,
        message: `Cannot kill "${agentId}": target is not a direct child of this agent.`,
      })
    }

    if (target.status !== 'starting' && target.status !== 'working') {
      return yield* Effect.fail({
        _tag: 'AgentError' as const,
        message: `Cannot kill "${agentId}": only starting or working subagents can be killed (current status: ${target.status}).`,
      })
    }

    const bus = yield* WorkerBusTag<AppEvent>()
    yield* bus.publish({
      type: 'agent_killed',
      forkId: target.forkId,
      parentForkId,
      agentId: target.agentId,
      reason: reason?.trim() || 'Killed by parent via agent.kill',
    })

    return { agentId: target.agentId, forkId: target.forkId }
  }),
})
// =============================================================================
// Tool Group Export
// =============================================================================

export const agentTools = [
  agentCreateTool,
  agentKillTool,
]
