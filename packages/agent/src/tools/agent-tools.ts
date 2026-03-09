/**
 * Agent Tools
 *
 * Tool group for managing agents (workers).
 * Agents are WHO does the work — separate from tasks (WHAT needs doing).
 * Context flows through message-based communication.
 *
 * Tools: create, pause, dismiss
 */

import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { createTool, ToolErrorSchema } from '@magnitudedev/tools'
import { Fork, WorkerBusTag } from '@magnitudedev/event-core'
import type { AppEvent, WorkAgentType } from '../events'


import { ForkStateReaderTag } from './fork'
import { ExecutionManager } from '../execution/execution-manager'
import { AgentRegistryStateReaderTag } from './agent-registry-reader'
import { ConversationStateReaderTag } from './memory-reader'
import { buildAgentContext, buildConversationSummary } from '../prompts'

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
  options: { type: WorkAgentType; title: string; message: string };
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

    const execManager = yield* ExecutionManager
    const { forkId: parentForkId } = yield* ForkContext
    const bus = yield* WorkerBusTag<AppEvent>()

    // Use a synthetic task ID for the fork (agents no longer require pre-existing tasks)
    const taskId = `agent-${agentId}`

    const forkId = yield* execManager.fork({
      parentForkId,
      name: title,
      agentId,
      prompt: context,
      mode: 'spawn',
      role: agentType,
      taskId,
    })

    yield* bus.publish({
      type: 'agent_created',
      forkId: parentForkId,
      agentId,
      taskId,
      agentType,
      agentForkId: forkId,
      message,
    })

    return { agentId, forkId }
  })
}


export const agentCreateTool = createTool({
  name: 'create' as const,
  group: 'agent' as const,
  description: 'Create a new agent and dispatch it with a title and message.',
  inputSchema: Schema.Struct({
    agentId: Schema.String.annotations({ description: 'Unique agent ID (e.g. db-builder, researcher-1)' }),
    options: Schema.Struct({
      type: Schema.Literal('researcher', 'planner', 'builder', 'debugger', 'reviewer', 'scout', 'browser').annotations({ description: 'Agent type' }),
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
      attributes: ['agentId'],
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
// =============================================================================
// agent.pause — Soft interrupt an agent
// =============================================================================

export const agentPauseTool = createTool({
  name: 'pause',
  group: 'agent',
  description: 'Soft interrupt an agent — current turn completes, then it stops. Partial output is retained.',
  inputSchema: Schema.Struct({
    agentId: Schema.String.annotations({ description: 'Agent ID' }),
  }),
  outputSchema: Schema.Struct({ agentId: Schema.String }),
  errorSchema: AgentError,
  argMapping: ['agentId'],
  bindings: {
    xmlInput: { type: 'tag', attributes: ['agentId'], selfClosing: true },
    xmlOutput: { type: 'tag' as const, childTags: [{ field: 'agentId', tag: 'agentId' }] },
  } as const,
  execute: ({ agentId }) =>
    Effect.gen(function* () {
      const registryReader = yield* AgentRegistryStateReaderTag
      const registryState = yield* registryReader.getState()
      const agent = registryState.agents.get(agentId)

      if (!agent || agent.status === 'dismissed') {
        return yield* Effect.fail({ _tag: 'AgentError' as const, message: `No active agent "${agentId}" found` })
      }

      const bus = yield* WorkerBusTag<AppEvent>()
      yield* bus.publish({ type: 'soft_interrupt', forkId: agent.forkId })
      yield* bus.publish({
        type: 'agent_paused',
        forkId: null,
        agentId,
        agentForkId: agent.forkId,
      })

      return { agentId }
    }),
})

// =============================================================================
// agent.dismiss — Dismiss an agent
// =============================================================================

export const agentDismissTool = createTool({
  name: 'dismiss',
  group: 'agent',
  description: "Dismiss an agent — stops it and removes it. Use when the agent's work is no longer relevant.",
  inputSchema: Schema.Struct({
    agentId: Schema.String.annotations({ description: 'Agent ID' }),
  }),
  outputSchema: Schema.Struct({ agentId: Schema.String }),
  errorSchema: AgentError,
  argMapping: ['agentId'],
  bindings: {
    xmlInput: { type: 'tag', attributes: ['agentId'], selfClosing: true },
    xmlOutput: { type: 'tag' as const, childTags: [{ field: 'agentId', tag: 'agentId' }] },
  } as const,
  execute: ({ agentId }) =>
    Effect.gen(function* () {
      const registryReader = yield* AgentRegistryStateReaderTag
      const registryState = yield* registryReader.getState()
      const agent = registryState.agents.get(agentId)

      if (!agent || agent.status === 'dismissed') {
        return yield* Effect.fail({ _tag: 'AgentError' as const, message: `No active agent "${agentId}" found` })
      }

      // Get parent fork ID from fork state
      const forkStateReader = yield* ForkStateReaderTag
      const forkState = yield* forkStateReader.getForkState()
      const fork = forkState.forks.get(agent.forkId)
      const parentForkId = fork?.parentForkId ?? null

      const bus = yield* WorkerBusTag<AppEvent>()

      // fork_completed triggers full cleanup: Cortex fiber interrupted (completeOn),
      // sandbox disposed (ForkOrchestrator), working state cleared (WorkingState).
      // No separate interrupt event needed — that would cause ForkOrchestrator.interrupt
      // to publish a duplicate fork_completed.
      yield* bus.publish({
        type: 'fork_completed',
        forkId: agent.forkId,
        parentForkId,
        result: { dismissed: true },
      })

      yield* bus.publish({
        type: 'agent_dismissed',
        forkId: null,
        agentId,
        agentForkId: agent.forkId,
      })

      return { agentId }
    }),
})

// =============================================================================
// Tool Group Export
// =============================================================================

export const agentTools = [
  agentCreateTool,
  agentPauseTool,
  agentDismissTool,
]
