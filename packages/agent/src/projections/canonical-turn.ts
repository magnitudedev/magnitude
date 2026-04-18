import { Projection } from '@magnitudedev/event-core'
import type { AppEvent, TurnResultError, MessageDestination } from '../events'

import { serializeCanonicalTurn, type CanonicalTrace } from './canonical-xml'
import { getBindingRegistry } from '../tools/binding-registry'
import { buildResolvedToolSet, type ResolvedToolSet } from '../tools/resolved-toolset'
import { ConfigAmbient } from '../ambient/config-ambient'
import { isValidVariant, type AgentVariant } from '../agents/variants'
import { getAgentDefinition, getAgentSlot } from '../agents/registry'
import { AgentStatusProjection, getAgentByForkId } from './agent-status'


export interface ThinkBlock {
  about: string | null
  content: string
}

export interface CanonicalTurnState {
  turnId: string | null
  lenses: readonly { name: string; content: string }[] | null
  thinkBlocks: ThinkBlock[]
  messages: Array<{ id: string; destination: MessageDestination; text: string; order: number }>
  messageMap: Map<string, number>
  toolCalls: Array<{ toolCallId: string; tagName: string; input: unknown; query: string; order: number }>
  toolCallMap: Map<string, number>
  hasParseError: boolean
  hasStructuralError: boolean
  rawResponse: string
  orderCounter: number
  lastCompleted: { turnId: string; canonicalXml: string; clean: boolean } | null
  resolvedTurnDecision: 'continue' | 'idle' | null
}

export const createInitialCanonicalTurnState = (): CanonicalTurnState => ({
  turnId: null,
  lenses: null,
  thinkBlocks: [],
  messages: [],
  messageMap: new Map(),
  toolCalls: [],
  toolCallMap: new Map(),
  hasParseError: false,
  hasStructuralError: false,
  rawResponse: '',
  orderCounter: 0,
  lastCompleted: null,
  resolvedTurnDecision: null,
})

function hasStructuralTurnError(errors?: readonly TurnResultError[]): boolean {
  if (!errors || errors.length === 0) return false
  return errors.some((error) => error.code === 'unclosed_think')
}

function resetActive(state: CanonicalTurnState): CanonicalTurnState {
  return {
    ...state,
    turnId: null,
    lenses: null,
    thinkBlocks: [],
    messages: [],
    messageMap: new Map(),
    toolCalls: [],
    toolCallMap: new Map(),
    hasParseError: false,
    hasStructuralError: false,
    rawResponse: '',
    orderCounter: 0,
    resolvedTurnDecision: null,
  }
}

export const CanonicalTurnProjection = Projection.defineForked<AppEvent, CanonicalTurnState>()({
  name: 'CanonicalTurn',
  reads: [AgentStatusProjection] as const,
  ambients: [ConfigAmbient] as const,
  initialFork: createInitialCanonicalTurnState(),
  eventHandlers: {
    turn_started: ({ event, fork }) => ({
      ...createInitialCanonicalTurnState(),
      turnId: event.turnId,
      lastCompleted: fork.lastCompleted,
    }),

    thinking_chunk: ({ event, fork }) => {
      if (fork.turnId !== event.turnId) return fork
      const blocks = [...fork.thinkBlocks]
      if (blocks.length === 0) {
        blocks.push({ about: null, content: event.text })
      } else {
        const last = blocks[blocks.length - 1]
        blocks[blocks.length - 1] = { ...last, content: last.content + event.text }
      }
      return { ...fork, thinkBlocks: blocks }
    },

    thinking_end: ({ event, fork }) => {
      if (fork.turnId !== event.turnId) return fork
      if (fork.thinkBlocks.length === 0) return fork
      const blocks = [...fork.thinkBlocks]
      const last = blocks[blocks.length - 1]
      blocks[blocks.length - 1] = { ...last, about: event.about }
      return { ...fork, thinkBlocks: blocks }
    },

    lens_start: ({ event, fork }) => {
      if (fork.turnId !== event.turnId) return fork
      const nextLenses = [...(fork.lenses ?? []), { name: event.name, content: '' }]
      return { ...fork, lenses: nextLenses }
    },

    lens_chunk: ({ event, fork }) => {
      if (fork.turnId !== event.turnId) return fork
      if (!fork.lenses || fork.lenses.length === 0) return fork
      const nextLenses = [...fork.lenses]
      const last = nextLenses[nextLenses.length - 1]
      nextLenses[nextLenses.length - 1] = {
        ...last,
        content: last.content + event.text,
      }
      return { ...fork, lenses: nextLenses }
    },

    lens_end: ({ event, fork }) => {
      if (fork.turnId !== event.turnId) return fork
      if (!fork.lenses || fork.lenses.length === 0) return fork
      const nextLenses = [...fork.lenses]
      const last = nextLenses[nextLenses.length - 1]
      if (last.name !== event.name) return fork
      const trimmed = last.content.trim()
      nextLenses[nextLenses.length - 1] = { ...last, content: trimmed }
      return { ...fork, lenses: nextLenses }
    },

    message_start: ({ event, fork }) => {
      if (fork.turnId !== event.turnId) return fork
      const idx = fork.messages.length
      const nextMessages = [...fork.messages, {
        id: event.id,
        destination: event.destination,
        text: '',
        order: fork.orderCounter,
      }]
      const nextMap = new Map(fork.messageMap)
      nextMap.set(event.id, idx)
      return {
        ...fork,
        messages: nextMessages,
        messageMap: nextMap,
        orderCounter: fork.orderCounter + 1,
      }
    },

    message_chunk: ({ event, fork }) => {
      if (fork.turnId !== event.turnId) return fork
      const idx = fork.messageMap.get(event.id)
      if (idx === undefined) return fork
      const next = [...fork.messages]
      next[idx] = { ...next[idx], text: next[idx].text + event.text }
      return { ...fork, messages: next }
    },

    message_end: ({ fork }) => fork,

    raw_response_chunk: ({ event, fork }) => {
      if (fork.turnId !== event.turnId) return fork
      return { ...fork, rawResponse: fork.rawResponse + event.text }
    },

    tool_event: ({ event, fork }) => {
      if (fork.turnId !== event.turnId) return fork

      switch (event.event._tag) {
        case 'ToolInputStarted': {
          const idx = fork.toolCalls.length
          const nextToolCalls = [...fork.toolCalls, {
            toolCallId: event.toolCallId,
            // Canonical XML is model-facing, so preserve the parsed XML tag here
            // instead of the internal catalog key carried alongside the app event.
            tagName: event.event.tagName,
            input: {},
            query: '.',
            order: fork.orderCounter,
          }]
          const nextMap = new Map(fork.toolCallMap)
          nextMap.set(event.toolCallId, idx)
          return {
            ...fork,
            toolCalls: nextToolCalls,
            toolCallMap: nextMap,
            orderCounter: fork.orderCounter + 1,
          }
        }

        case 'ToolInputReady': {
          const idx = fork.toolCallMap.get(event.toolCallId)
          if (idx === undefined) return fork
          const next = [...fork.toolCalls]
          next[idx] = { ...next[idx], input: event.event.input }
          return { ...fork, toolCalls: next }
        }

        case 'ToolObservation': {
          const idx = fork.toolCallMap.get(event.toolCallId)
          if (idx === undefined) return fork
          const nextToolCalls = [...fork.toolCalls]
          nextToolCalls[idx] = { ...nextToolCalls[idx], query: event.event.query }
          return { ...fork, toolCalls: nextToolCalls }
        }

        case 'ToolInputParseError':
          return { ...fork, hasParseError: true }

        default:
          return fork
      }
    },

    turn_completed: ({ event, fork, read, ambient }) => {
      if (fork.turnId !== event.turnId) return fork

      const hasStructuralError = fork.hasStructuralError || (event.result.success ? hasStructuralTurnError(event.result.errors) : false)
      const clean = !fork.hasParseError && !hasStructuralError && event.result.success === true

      let canonicalXml: string
      if (clean) {
        const agentState = read(AgentStatusProjection)
        const variant: AgentVariant = event.forkId
          ? (() => {
              const role = getAgentByForkId(agentState, event.forkId)?.role
              return role && isValidVariant(role) ? role : 'worker'
            })()
          : 'lead'
        const agentDef = getAgentDefinition(variant)
        const slot = getAgentSlot(variant)
        
        // Build toolSet for getBindingRegistry using ConfigAmbient
        const configState = ambient.get(ConfigAmbient)
        const toolSet = buildResolvedToolSet(agentDef, configState, slot)
        const bindings = getBindingRegistry(toolSet)
        
        const trace: CanonicalTrace = {
          lenses: fork.lenses,
          thinkBlocks: fork.thinkBlocks,
          messages: [...fork.messages]
            .sort((a, b) => a.order - b.order)
            .map(({ text, destination }) => ({ text, destination })),
          toolCalls: [...fork.toolCalls]
            .sort((a, b) => a.order - b.order)
            .map(({ tagName, input, query }) => ({ tagName, input, query })),
          turnDecision: event.result.turnDecision === 'idle' ? 'idle' : 'continue',
        }
        canonicalXml = serializeCanonicalTurn(trace, bindings)
      } else {
        canonicalXml = fork.rawResponse
      }

      const finalized: CanonicalTurnState = {
        ...fork,
        hasStructuralError,
        lastCompleted: {
          turnId: event.turnId,
          canonicalXml,
          clean,
        }
      }

      return resetActive(finalized)
    },

    turn_unexpected_error: ({ fork }) => resetActive(fork),

    interrupt: ({ fork }) => fork,

  }
})