/**
 * MemoryProjection (Forked)
 *
 * LLM conversation history, per-fork.
 * Each fork has independent message history.
 */

import { Projection } from '@magnitudedev/event-core'
import type { AppEvent, ResponsePart, StrategyId, Attachment } from '../events'
import { ForkProjection } from './fork'

import { SubagentActivityProjection } from './subagent-activity'
import { AgentRegistryProjection } from './agent-registry'

import { CanonicalTurnProjection } from './canonical-turn'
import { SessionContextProjection } from './session-context'
import { OutboundMessagesProjection } from './outbound-messages'

import {
  compactionSummaryTag, buildSessionContextContent,

  formatCommsInbox, formatSystemInbox,
} from '../prompts'
import { UserPresenceProjection } from './user-presence'
import { formatUserPresence, formatUserReturnedAfterAbsence } from '../prompts/presence'
import { ContentPart, textParts, wrapTextParts } from '../content'
import type { CommsAttachment, CommsEntry, SystemEntry } from '../prompts/agents'
import { ArtifactAwarenessProjection } from './artifact-awareness'

export type MessageSource = 'user' | 'agent' | 'system'

export type Message =
  | { readonly type: 'session_context'; readonly source: 'system'; readonly content: ContentPart[] }
  | { readonly type: 'assistant_turn'; readonly source: 'agent'; readonly content: ContentPart[]; readonly strategyId: StrategyId; readonly responseParts: readonly ResponsePart[] }
  | { readonly type: 'compacted'; readonly source: 'system'; readonly content: ContentPart[] }
  | { readonly type: 'fork_context'; readonly source: 'system'; readonly content: ContentPart[] }
  | { readonly type: 'comms_inbox'; readonly source: 'system'; readonly entries: readonly CommsEntry[] }
  | { readonly type: 'system_inbox'; readonly source: 'system'; readonly entries: readonly SystemEntry[] }

export interface LLMMessage {
  readonly role: 'user' | 'assistant'
  readonly content: ContentPart[]
}

export type Perspective = 'agent' | 'autopilot'

export interface QueuedCommsMessage {
  readonly kind: 'comms'
  readonly entry: CommsEntry
}

export interface QueuedSystemMessage {
  readonly kind: 'system'
  readonly timestamp: number
  readonly entry: SystemEntry
  readonly coalesceKey?: string
}

export type QueuedMessage = QueuedCommsMessage | QueuedSystemMessage

export interface ForkMemoryState {
  readonly messages: readonly Message[]
  readonly queuedMessages: readonly QueuedMessage[]
  readonly currentTurnId: string | null
  readonly currentTurnToolCalls: readonly []
  readonly currentChainId: string | null
  readonly pendingPresenceText: string | null
}

function flattenResponseText(parts: readonly ResponsePart[]): string {
  return parts
    .filter((p): p is Extract<ResponsePart, { type: 'text' }> => p.type === 'text')
    .map(p => p.content)
    .join('')
}

function extractText(parts: readonly ContentPart[]): string {
  return parts
    .filter((p): p is Extract<ContentPart, { type: 'text' }> => p.type === 'text')
    .map(p => p.text)
    .join('')
}

function toCommsImageAttachment(attachment: Attachment): CommsAttachment {
  return { kind: 'image', base64: attachment.base64, mediaType: attachment.mediaType, width: attachment.width, height: attachment.height }
}



function transformMessage(message: Message, timezone: string | null, perspective: Perspective): LLMMessage {
  const content = message.type === 'comms_inbox'
    ? formatCommsInbox(message.entries, timezone)
    : message.type === 'system_inbox'
      ? formatSystemInbox(message.entries)
      : message.content

  if (message.type === 'compacted') {
    return { role: 'user', content: wrapTextParts(content, text => compactionSummaryTag(text)) }
  }

  switch (perspective) {
    case 'agent':
      switch (message.source) {
        case 'agent': return { role: 'assistant', content }
        case 'system': return { role: 'user', content }
      }
    case 'autopilot':
      switch (message.source) {
        case 'agent': return { role: 'user', content: wrapTextParts(content, text => `<agent>\n${text}\n</agent>`) }
        case 'system': return { role: 'user', content }
      }
  }
}

export function getView(messages: readonly Message[], timezone: string | null, perspective: Perspective): LLMMessage[] {
  return messages.map(msg => transformMessage(msg, timezone, perspective))
}

export const MemoryProjection = Projection.defineForked<AppEvent, ForkMemoryState>()({
  name: 'Memory',
  reads: [ForkProjection, ArtifactAwarenessProjection, AgentRegistryProjection, SubagentActivityProjection, CanonicalTurnProjection, SessionContextProjection, UserPresenceProjection, OutboundMessagesProjection] as const,
  signals: {},
  initialFork: {
    messages: [],
    queuedMessages: [],
    currentTurnId: null,
    currentTurnToolCalls: [],
    currentChainId: null,
    pendingPresenceText: null,
  },

  eventHandlers: {
    session_initialized: ({ event, fork }) => {
      const content = buildSessionContextContent(event.context)
      const sessionMsg: Message = { type: 'session_context', source: 'system', content: textParts(content) }
      return { ...fork, messages: [sessionMsg, ...fork.messages] }
    },

    user_message: ({ event, fork }) => {
      const text = extractText(event.content)
      const attachments = (event.attachments ?? []).map(toCommsImageAttachment)
      const entry: CommsEntry = { kind: 'user', timestamp: event.timestamp, text, attachments }

      if (fork.currentTurnId !== null) {
        return {
          ...fork,
          queuedMessages: [...fork.queuedMessages, { kind: 'comms', entry }],
        }
      }

      return {
        ...fork,
        messages: [...fork.messages, { type: 'comms_inbox', source: 'system', entries: [entry] }],
      }
    },

    fork_started: ({ fork }) => fork,

    turn_started: ({ event, fork, read }) => {
      const commsEntries = fork.queuedMessages
        .filter((q): q is QueuedCommsMessage => q.kind === 'comms')
        .map(q => q.entry)
        .sort((a, b) => a.timestamp - b.timestamp)

      const systemEntries = fork.queuedMessages
        .filter((q): q is QueuedSystemMessage => q.kind === 'system')
        .sort((a, b) => a.timestamp - b.timestamp)
        .map(q => q.entry)

      const flushedMessages: Message[] = []

      // Inject pending presence notification
      if (event.forkId === null && fork.pendingPresenceText !== null) {
        systemEntries.unshift({ kind: 'reminder', text: fork.pendingPresenceText })
      } else if (event.forkId === null && read(UserPresenceProjection).currentFocusState === false) {
        systemEntries.unshift({ kind: 'reminder', text: formatUserPresence(false) })
      }

      if (systemEntries.length > 0) {
        flushedMessages.push({ type: 'system_inbox', source: 'system', entries: systemEntries })
      }
      if (commsEntries.length > 0) {
        flushedMessages.push({ type: 'comms_inbox', source: 'system', entries: commsEntries })
      }

      return {
        ...fork,
        messages: [...fork.messages, ...flushedMessages],
        queuedMessages: [],
        currentTurnId: event.turnId,
        currentTurnToolCalls: [],
        currentChainId: event.chainId,
        pendingPresenceText: null,
      }
    },

    tool_event: ({ fork }) => fork,

    turn_completed: ({ event, fork, read }) => {
      if (fork.currentTurnId !== event.turnId) return fork

      const newMessages: Message[] = [...fork.messages]
      const isCancelled = !event.result.success && event.result.cancelled

      if (event.responseParts.length > 0) {
        const canonical = read(CanonicalTurnProjection)
        const canonicalText = canonical.lastCompleted?.turnId === event.turnId && canonical.lastCompleted.clean
          ? canonical.lastCompleted.canonicalXml
          : flattenResponseText(event.responseParts)

        newMessages.push({
          type: 'assistant_turn',
          source: 'agent',
          content: textParts(canonicalText),
          strategyId: event.strategyId,
          responseParts: event.responseParts,
        })
      }

      const systemEntries: SystemEntry[] = []
      const hasError = !event.result.success
      const errorMessage = hasError ? event.result.error : undefined
      const inspectResults = isCancelled ? [] : event.inspectResults
      if (event.toolCalls.length > 0 || inspectResults.length > 0 || hasError) {
        systemEntries.push({ kind: 'tool_results', toolCalls: event.toolCalls, inspectResults, error: errorMessage })
      }
      if (event.result.success && event.result.reminder) {
        systemEntries.push({ kind: 'reminder', text: event.result.reminder })
      }
      if (isCancelled) {
        systemEntries.push({ kind: 'interrupted' })
      }

      const newQueuedMessages = systemEntries.map(entry => ({
        kind: 'system' as const,
        timestamp: event.timestamp,
        entry,
      }))

      return {
        ...fork,
        messages: newMessages,
        queuedMessages: [...fork.queuedMessages, ...newQueuedMessages],
        currentTurnId: null,
        currentTurnToolCalls: []
      }
    },

    turn_unexpected_error: ({ event, fork }) => {
      if (fork.currentTurnId !== event.turnId) return fork
      return {
        ...fork,
        messages: [...fork.messages, { type: 'system_inbox', source: 'system', entries: [{ kind: 'error', message: event.message }] }],
        currentTurnId: null,
        currentTurnToolCalls: []
      }
    },

    interrupt: ({ fork }) => {
      return {
        ...fork,
        queuedMessages: fork.queuedMessages.filter(q => !(q.kind === 'comms' && q.entry.kind === 'user'))
      }
    },

    compaction_completed: ({ event, fork }) => {
      const remainingMessages = fork.messages.slice(1 + event.compactedMessageCount)
      const sessionContext: Message = event.refreshedContext
        ? { type: 'session_context', source: 'system', content: textParts(buildSessionContextContent(event.refreshedContext)) }
        : fork.messages[0]

      const summaryMessage: Message = { type: 'compacted', source: 'system', content: textParts(event.summary) }
      return { ...fork, messages: [sessionContext, summaryMessage, ...remainingMessages], currentChainId: null }
    },


  },


  signalHandlers: (on) => [
    on(ForkProjection.signals.forkCreated, ({ value, source, state }) => {
      const { forkId, parentForkId } = value
      const parentState = state.forks.get(parentForkId)
      if (!parentState) throw new Error(`Parent fork ${parentForkId} not found in MemoryProjection`)

      const forkInstance = source.forks.get(forkId)
      const contextMessage: Message[] = forkInstance?.context
        ? [{ type: 'fork_context', source: 'system', content: textParts(forkInstance.context) }]
        : []

      const isSpawn = forkInstance?.mode === 'spawn'
      const baseMessages = isSpawn ? [] : parentState.messages

      const newForkState: ForkMemoryState = {
        messages: [...baseMessages, ...contextMessage],
        queuedMessages: [],
        currentTurnId: null,
        currentTurnToolCalls: [],
        currentChainId: null,
        pendingPresenceText: null,
      }

      return { ...state, forks: new Map(state.forks).set(forkId, newForkState) }
    }),

    on(ForkProjection.signals.forkCompleted, ({ value, state }) => {
      const { parentForkId, name, result, role, taskId } = value
      const parentState = state.forks.get(parentForkId)
      if (!parentState) return state

      const entry: SystemEntry = { kind: 'fork_result', taskId: taskId ?? null, role, name, result }

      return {
        ...state,
        forks: new Map(state.forks).set(parentForkId, {
          ...parentState,
          queuedMessages: [...parentState.queuedMessages, { kind: 'system', timestamp: value.timestamp, entry }]
        })
      }
    }),

    on(OutboundMessagesProjection.signals.messageCompleted, ({ value, state, read }) => {
      if (value.dest === 'user') return state

      const forkState = read(ForkProjection)
      const senderAgentId = value.forkId === null
        ? 'orchestrator'
        : (forkState.forks.get(value.forkId)?.agentId ?? 'orchestrator')

      const targetForkId = value.targetForkId
      if (targetForkId === undefined) return state
      const targetState = state.forks.get(targetForkId)
      if (!targetState) return state

      const entry: CommsEntry = { kind: 'agent', from: senderAgentId, timestamp: value.timestamp, text: value.text }

      return {
        ...state,
        forks: new Map(state.forks).set(targetForkId, {
          ...targetState,
          queuedMessages: [...targetState.queuedMessages, { kind: 'comms', entry }],
        })
      }
    }),

    on(ArtifactAwarenessProjection.signals.artifactFirstMentioned, ({ value, state }) => {
      const forkState = state.forks.get(value.forkId)
      if (!forkState) return state
      const entry: SystemEntry = { kind: 'reminder', text: `<artifact id="${value.artifactId}">\n${value.content}\n</artifact>` }
      return {
        ...state,
        forks: new Map(state.forks).set(value.forkId, {
          ...forkState,
          queuedMessages: [...forkState.queuedMessages, { kind: 'system', timestamp: value.timestamp, entry }]
        })
      }
    }),

    on(ArtifactAwarenessProjection.signals.artifactUpdateNotification, ({ value, state }) => {
      const forkState = state.forks.get(value.forkId)
      if (!forkState) return state
      const entry: SystemEntry = { kind: 'reminder', text: value.text }
      const coalesceKey = `artifact-update:${value.artifactId}`
      return {
        ...state,
        forks: new Map(state.forks).set(value.forkId, {
          ...forkState,
          queuedMessages: [
            ...forkState.queuedMessages.filter(q => !(q.kind === 'system' && (q as QueuedSystemMessage).coalesceKey === coalesceKey)),
            { kind: 'system', timestamp: value.timestamp, entry, coalesceKey }
          ]
        })
      }
    }),

    on(SubagentActivityProjection.signals.unseenActivityAvailable, ({ value, state }) => {
      const parentState = state.forks.get(value.parentForkId)
      if (!parentState) return state

      const entry: SystemEntry = {
        kind: 'agent_activity',
        entries: value.entries.map(item => ({
          agentId: item.agentId,
          prose: item.prose,
          toolsCalled: item.toolsCalled,
          artifactsWritten: item.artifactsWritten,
        }))
      }

      return {
        ...state,
        forks: new Map(state.forks).set(value.parentForkId, {
          ...parentState,
          queuedMessages: [...parentState.queuedMessages, { kind: 'system', timestamp: value.timestamp, entry }]
        })
      }
    }),





    on(UserPresenceProjection.signals.userReturnedAfterAbsence, ({ state }) => {
      const rootState = state.forks.get(null)
      if (!rootState) return state
      return {
        ...state,
        forks: new Map(state.forks).set(null, {
          ...rootState,
          pendingPresenceText: formatUserReturnedAfterAbsence(),
        })
      }
    }),


  ]
})