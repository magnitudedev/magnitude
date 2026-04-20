/**
 * MemoryProjection (Forked)
 *
 * LLM conversation history, per-fork.
 * Each fork has independent message history.
 */

import { Projection } from '@magnitudedev/event-core'
import type { ObservationPart } from '@magnitudedev/roles'
import type { AppEvent, StrategyId, ImageAttachment } from '../events'
import { deriveParameters } from '@magnitudedev/xml-act'
import { catalog } from '../catalog'
import { getAgentByForkId, AgentStatusProjection } from './agent-status'
import { SubagentActivityProjection } from './subagent-activity'
import { CanonicalTurnProjection } from './canonical-turn'

import { OutboundMessagesProjection } from './outbound-messages'
import { compactionSummaryTag } from '../prompts/constants'
import { buildSessionContextContent } from '../prompts/session-context'
import { TASK_TREE_COMPLETION_REMINDER } from '../prompts/task-tree'
import { SkillsAmbient } from '../ambient/skills-ambient'
import { UserPresenceProjection } from './user-presence'
import { UserMessageResolutionProjection } from './user-message-resolution'
import { TaskGraphProjection, type TaskGraphState, type TaskRecord } from './task-graph'
import { isValidVariant } from '../agents/variants'
import { getAgentDefinition } from '../agents/registry'
import { formatUserPresence, formatUserReturnedAfterAbsence } from '../prompts/presence'
import { ContentPart, ImageMediaType, textParts, wrapTextParts } from '../content'

import { EMPTY_RESPONSE_ERROR } from '../prompts/error-states'
import { formatInbox } from '../inbox/render'
import type {
  ResultEntry,
  TimelineEntry,
  TimelineAttachment,
  QueuedEntry,
  AgentAtom,
  TurnResultItem,
} from '../inbox/types'
import {
  toResultTurnResults,
  toResultInterrupted,
  toResultError,
  toResultNoop,
  toTimelineUserMessage,
  toTimelineParentMessage,
  toTimelineUserToAgent,
  toTimelineUserBashCommand,
  toTimelineUserPresence,
  toTimelineObservation,
  toTimelineAgentBlock,
  toTimelineSubagentUserKilled,
  toTimelineTaskTypeHook,
  toTimelineTaskIdleHook,
  toTimelineTaskCompleteHook,
  toTimelineTaskTreeDirty,
  toTimelineTaskTreeView,
  toTimelineTaskUpdate,
} from '../inbox/compose'

export type MessageSource = 'user' | 'agent' | 'system'

export type Message =
  | { readonly type: 'session_context'; readonly source: 'system'; readonly content: ContentPart[] }
  | {
      readonly type: 'assistant_turn'
      readonly source: 'agent'
      readonly content: ContentPart[]
      readonly strategyId: StrategyId
    }
  | { readonly type: 'compacted'; readonly source: 'system'; readonly content: ContentPart[] }
  | { readonly type: 'fork_context'; readonly source: 'system'; readonly content: ContentPart[] }
  | {
      readonly type: 'inbox'
      readonly source: 'system'
      readonly results: readonly ResultEntry[]
      readonly timeline: readonly TimelineEntry[]
    }

export interface LLMMessage {
  readonly role: 'user' | 'assistant'
  readonly content: ContentPart[]
}

export type Perspective = 'agent' | 'autopilot'

type PendingParentMessage = {
  readonly body: string
  readonly destination: 'parent'
}

export interface ForkMemoryState {
  readonly messages: readonly Message[]
  readonly queuedEntries: readonly QueuedEntry[]
  readonly currentTurnId: string | null
  readonly currentChainId: string | null
  readonly pendingPresenceText: string | null
  readonly nextQueueSeq: number
  readonly pendingResultItems: readonly TurnResultItem[]
  readonly pendingParentMessages: ReadonlyMap<string, PendingParentMessage>
}

function resetPendingTurnState(fork: ForkMemoryState): ForkMemoryState {
  return {
    ...fork,
    pendingResultItems: [],
    pendingParentMessages: new Map(),
  }
}



function extractText(parts: readonly ContentPart[]): string {
  return parts
    .filter((p): p is Extract<ContentPart, { type: 'text' }> => p.type === 'text')
    .map(p => p.text)
    .join('')
}

function appendNewInbox(messages: readonly Message[], options: { results?: readonly ResultEntry[], timeline?: readonly TimelineEntry[] }): readonly Message[] {
  const results = options.results ?? []
  const timeline = options.timeline ?? []
  if (results.length === 0 && timeline.length === 0) return messages
  return [...messages, { type: 'inbox', source: 'system', results: [...results], timeline: [...timeline] }]
}

function enqueueResult(
  fork: ForkMemoryState,
  entry: ResultEntry,
  timestamp: number,
  coalesceKey?: string,
): ForkMemoryState {
  const seq = fork.nextQueueSeq
  const queued: QueuedEntry = { lane: 'result', timestamp, seq, entry, coalesceKey }
  const queuedEntries = coalesceKey
    ? [...fork.queuedEntries.filter(q => q.coalesceKey !== coalesceKey), queued]
    : [...fork.queuedEntries, queued]
  return { ...fork, queuedEntries, nextQueueSeq: seq + 1 }
}

function enqueueTimeline(
  fork: ForkMemoryState,
  entry: TimelineEntry,
  timestamp: number,
  coalesceKey?: string,
): ForkMemoryState {
  const seq = fork.nextQueueSeq
  const queued: QueuedEntry = { lane: 'timeline', timestamp, seq, entry, coalesceKey }
  const queuedEntries = coalesceKey
    ? [...fork.queuedEntries.filter(q => q.coalesceKey !== coalesceKey), queued]
    : [...fork.queuedEntries, queued]
  return { ...fork, queuedEntries, nextQueueSeq: seq + 1 }
}

function flushQueue(fork: ForkMemoryState, taskGraphState: TaskGraphState): ForkMemoryState {
  const sorted = [...fork.queuedEntries].sort((a, b) => (a.timestamp - b.timestamp) || (a.seq - b.seq))
  const results: ResultEntry[] = []
  const timeline: TimelineEntry[] = []
  const dirtyTaskIds = new Set<string>()
  let latestDirtyTimestamp: number | null = null

  for (const queued of sorted) {
    if (queued.lane === 'result') {
      results.push(queued.entry)
      continue
    }

    if (queued.entry.kind === 'task_tree_dirty') {
      dirtyTaskIds.add(queued.entry.taskId)
      latestDirtyTimestamp = latestDirtyTimestamp === null
        ? queued.entry.timestamp
        : Math.max(latestDirtyTimestamp, queued.entry.timestamp)
      continue
    }

    timeline.push(queued.entry)
  }

  if (dirtyTaskIds.size > 0 && latestDirtyTimestamp !== null) {
    const renderedTree = renderTaskTreesForTaskIds(taskGraphState, Array.from(dirtyTaskIds))
    if (renderedTree) {
      timeline.push(
        toTimelineTaskTreeView({
          timestamp: latestDirtyTimestamp,
          renderedTree,
        }),
      )
    }
  }

  return {
    ...fork,
    messages: appendNewInbox(fork.messages, { results, timeline }),
    queuedEntries: [],
  }
}

function enqueueAgentAtomBlock(
  fork: ForkMemoryState,
  args: {
    timestamp: number
    agentId: string
    role: string
    atoms: readonly AgentAtom[]
  },
): ForkMemoryState {
  if (args.atoms.length === 0) return fork

  const last = fork.queuedEntries[fork.queuedEntries.length - 1]
  if (
    last
    && last.lane === 'timeline'
    && last.entry.kind === 'agent_block'
    && last.entry.agentId === args.agentId
  ) {
    const mergedEntry = toTimelineAgentBlock({
      timestamp: last.entry.timestamp,
      firstAtomTimestamp: last.entry.firstAtomTimestamp,
      lastAtomTimestamp: args.atoms[args.atoms.length - 1]!.timestamp,
      agentId: last.entry.agentId,
      role: last.entry.role,
      atoms: [...last.entry.atoms, ...args.atoms],
    })

    const mergedQueued: QueuedEntry = {
      ...last,
      timestamp: Math.min(last.timestamp, args.timestamp),
      entry: mergedEntry,
    }

    return {
      ...fork,
      queuedEntries: [...fork.queuedEntries.slice(0, -1), mergedQueued],
    }
  }

  return enqueueTimeline(
    fork,
    toTimelineAgentBlock({
      timestamp: args.timestamp,
      firstAtomTimestamp: args.atoms[0]!.timestamp,
      lastAtomTimestamp: args.atoms[args.atoms.length - 1]!.timestamp,
      agentId: args.agentId,
      role: args.role,
      atoms: args.atoms,
    }),
    args.timestamp,
  )
}

function toContentPartFromObservation(part: ObservationPart): ContentPart {
  if (part.type === 'text') {
    return { type: 'text', text: part.text }
  }

  return {
    type: 'image',
    base64: part.base64,
    mediaType: part.mediaType as ImageMediaType,
    width: part.width,
    height: part.height,
  }
}

function findRootTaskId(state: TaskGraphState, taskId: string): string {
  let current = state.tasks.get(taskId)
  while (current && current.parentId) {
    const parent = state.tasks.get(current.parentId)
    if (!parent) break
    current = parent
  }
  return current?.id ?? taskId
}

function renderTaskSubtree(state: TaskGraphState, taskId: string, depth: number): string[] {
  const task = state.tasks.get(taskId)
  if (!task) return []

  const indent = '  '.repeat(depth)

  const status = task.status === 'completed' ? 'done' : task.status
  const assignedRoleStr = task.worker && task.worker.role !== 'user'
    ? `, assigned: ${task.worker.role}`
    : ''
  const assigneeStr = task.assignee === 'user' ? ', user' : ''
  const line = `${indent}[${status}] ${task.title} (${task.id}${assignedRoleStr}${assigneeStr})`

  const childLines = task.childIds.flatMap(childId => renderTaskSubtree(state, childId, depth + 1))


  return [line, ...childLines]
}

function renderTaskTreesForTaskIds(state: TaskGraphState, taskIds: readonly string[]): string {
  const roots = new Set<string>()
  for (const taskId of taskIds) {
    roots.add(findRootTaskId(state, taskId))
  }

  const renderedTrees = Array.from(roots)
    .map(rootId => renderTaskSubtree(state, rootId, 0).join('\n'))
    .filter(Boolean)
    .join('\n')

  if (!renderedTrees) return ''

  return `${renderedTrees}\n${TASK_TREE_COMPLETION_REMINDER}`
}

function findTaskForAgent(state: TaskGraphState, args: { agentId: string, forkId: string }): TaskRecord | null {
  for (const task of Array.from(state.tasks.values())) {
    if (task.worker?.agentId === args.agentId || task.worker?.forkId === args.forkId) return task
  }
  return null
}

function getAgentDefinitionForFork(read: <T>(projection: T) => any, forkId: string | null) {
  if (forkId === null) return getAgentDefinition('lead')
  const role = getAgentByForkId(read(AgentStatusProjection), forkId)?.role
  return role && isValidVariant(role) ? getAgentDefinition(role) : undefined
}

/** Lazy map from tagName to correctToolShape (XML format), built once from catalog */
const toolShapeByTagName: Map<string, string> = (() => {
  const map = new Map<string, string>()
  for (const [, entry] of Object.entries(catalog.entries)) {
    const e = entry as { tool: { name: string; inputSchema: { ast: unknown } } }
    try {
      const params = deriveParameters(e.tool.inputSchema.ast as import('@effect/schema/AST').AST)
      let shape = '<' + e.tool.name
      for (const [name] of params.parameters) {
        shape += '\n' + name + '="..."'
      }
      shape += '\n/>'
      map.set(e.tool.name, shape)
    } catch {
      // skip tools that fail to generate
    }
  }
  return map
})()

function toToolErrorResult(args: {
  tagName: string
  status: Extract<TurnResultItem, { kind: 'tool_error' }>['status']
  message?: string
  correctToolShape?: string
}): TurnResultItem {
  return {
    kind: 'tool_error',
    tagName: args.tagName,
    status: args.status,
    message: args.message,
    correctToolShape: args.correctToolShape ?? toolShapeByTagName.get(args.tagName),
  }
}

function transformMessage(message: Message, timezone: string | null, perspective: Perspective): LLMMessage {
  const content = message.type === 'inbox'
    ? formatInbox({ results: message.results, timeline: message.timeline, timezone })
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
  reads: [AgentStatusProjection, SubagentActivityProjection, CanonicalTurnProjection, UserPresenceProjection, OutboundMessagesProjection, UserMessageResolutionProjection, TaskGraphProjection] as const,
  ambients: [SkillsAmbient] as const,
  signals: {},
  initialFork: {
    messages: [],
    queuedEntries: [],
    currentTurnId: null,
    currentChainId: null,
    pendingPresenceText: null,
    nextQueueSeq: 0,
    pendingResultItems: [],
    pendingParentMessages: new Map(),
  },

  eventHandlers: {
    session_initialized: ({ event, fork }) => {
      const content = buildSessionContextContent(event.context)
      const sessionMsg: Message = { type: 'session_context', source: 'system', content: textParts(content) }
      return { ...fork, messages: [sessionMsg, ...fork.messages] }
    },

    oneshot_task: ({ event, fork }) => {
      const taskMessage: Message = {
        type: 'session_context',
        source: 'system',
        content: textParts(
          '<task>\n'
          + event.prompt
          + '\n</task>\n\n<critical-reminder>Thoroughly and efficiently complete this task. Be strategic and creative, try multiple approaches in parallel when appropriate, but ensure that the task is fully complete and the environment is clean before you finish.</critical-reminder>',
        ),
      }
      return { ...fork, messages: [...fork.messages, taskMessage] }
    },

    skill_activated: ({ event, fork }) => {
      if (event.source !== 'user') return fork
      const text = event.message ? `/${event.skillName} ${event.message}` : `/${event.skillName}`
      const entry = toTimelineUserMessage({ timestamp: event.timestamp, text, attachments: [] })
      return enqueueTimeline(fork, entry, event.timestamp)
    },

    user_bash_command: ({ event, fork }) =>
      enqueueTimeline(
        fork,
        toTimelineUserBashCommand({
          timestamp: event.timestamp,
          command: event.command,
          cwd: event.cwd,
          exitCode: event.exitCode,
          stdout: event.stdout,
          stderr: event.stderr,
        }),
        event.timestamp,
      ),

    turn_started: ({ event, fork, read }) => {
      let nextFork = fork

      if (event.forkId === null && nextFork.pendingPresenceText !== null) {
        nextFork = enqueueTimeline(
          nextFork,
          toTimelineUserPresence({ timestamp: event.timestamp, text: nextFork.pendingPresenceText, confirmed: true }),
          event.timestamp,
        )
      } else if (event.forkId === null && read(UserPresenceProjection).currentFocusState === false) {
        nextFork = enqueueTimeline(
          nextFork,
          toTimelineUserPresence({ timestamp: event.timestamp, text: formatUserPresence(false), confirmed: false }),
          event.timestamp,
        )
      }

      const preFlushMessageCount = nextFork.messages.length
      const taskGraphState = read(TaskGraphProjection)
      const flushed = flushQueue(nextFork, taskGraphState)

      let messages = flushed.messages
      const flushProducedInbox = messages.length > preFlushMessageCount
      const lastMessage = messages[messages.length - 1]
      if (!flushProducedInbox && lastMessage?.source === 'agent') {
        messages = [...messages, { type: 'inbox', source: 'system', results: [toResultNoop()], timeline: [] }]
      }

      return {
        ...resetPendingTurnState(flushed),
        messages,
        currentTurnId: event.turnId,
        currentChainId: event.chainId,
        pendingPresenceText: null,
      }
    },

    tool_event: ({ event, fork, read }) => {
      if (fork.currentTurnId !== event.turnId) return fork

      switch (event.event._tag) {
        case 'ToolExecutionEnded': {
          const result = event.event.result
          const tagName = event.event.tagName ?? event.toolKey
          switch (result._tag) {
            case 'Success':
              return fork
            case 'Error':
              return {
                ...fork,
                pendingResultItems: [
                  ...fork.pendingResultItems,
                  toToolErrorResult({
                    tagName,
                    status: 'error',
                    message: result.error,
                  }),
                ],
              }
            case 'Rejected':
              return {
                ...fork,
                pendingResultItems: [
                  ...fork.pendingResultItems,
                  toToolErrorResult({
                    tagName,
                    status: 'rejected',
                    message: typeof result.rejection === 'string' ? result.rejection : undefined,
                  }),
                ],
              }
            case 'Interrupted':
              return {
                ...fork,
                pendingResultItems: [
                  ...fork.pendingResultItems,
                  toToolErrorResult({
                    tagName,
                    status: 'interrupted',
                  }),
                ],
              }
          }
        }

        case 'ToolObservation':
          return {
            ...fork,
            pendingResultItems: [
              ...fork.pendingResultItems,
              {
                kind: 'tool_observation',
                tagName: event.event.tagName,
                query: event.event.query,
                content: event.event.content,
              },
            ],
          }

        case 'ToolInputParseError':
          return {
            ...fork,
            pendingResultItems: [
              ...fork.pendingResultItems,
              toToolErrorResult({
                tagName: event.event.tagName,
                status: 'error',
                message: `Invalid tool input: ${event.event.error.detail}`,
                correctToolShape: event.event.correctToolShape,
              }),
            ],
          }

        default:
          return fork
      }
    },

    message_start: ({ event, fork }) => {
      if (fork.currentTurnId !== event.turnId) return fork
      if (event.forkId === null || event.destination.kind !== 'parent') return fork
      return {
        ...fork,
        pendingParentMessages: new Map(fork.pendingParentMessages).set(event.id, {
          body: '',
          destination: 'parent',
        }),
      }
    },

    message_chunk: ({ event, fork }) => {
      if (fork.currentTurnId !== event.turnId) return fork
      const pending = fork.pendingParentMessages.get(event.id)
      if (!pending) return fork

      return {
        ...fork,
        pendingParentMessages: new Map(fork.pendingParentMessages).set(event.id, {
          ...pending,
          body: pending.body + event.text,
        }),
      }
    },

    message_end: ({ event, fork }) => {
      if (fork.currentTurnId !== event.turnId) return fork
      const pending = fork.pendingParentMessages.get(event.id)
      if (!pending) return fork

      const pendingParentMessages = new Map(fork.pendingParentMessages)
      pendingParentMessages.delete(event.id)

      return {
        ...fork,
        pendingParentMessages,
        pendingResultItems: [
          ...fork.pendingResultItems,
          {
            kind: 'message_ack',
            destination: pending.destination,
            chars: pending.body.length,
          },
        ],
      }
    },

    observations_captured: ({ event, fork, read }) => {
      if (fork.currentTurnId !== event.turnId) return fork
      const nextFork = enqueueTimeline(
        fork,
        toTimelineObservation({ timestamp: event.timestamp, parts: event.parts.map(toContentPartFromObservation) }),
        event.timestamp,
      )
      return flushQueue(nextFork, read(TaskGraphProjection))
    },

    turn_completed: ({ event, fork, read }) => {
      if (fork.currentTurnId !== event.turnId) return fork

      const newMessages: Message[] = [...fork.messages]
      const isCancelled = !event.result.success && 'cancelled' in event.result && event.result.cancelled
      const canonical = read(CanonicalTurnProjection)
      const canonicalText = canonical.lastCompleted?.turnId === event.turnId
        ? canonical.lastCompleted.canonicalMact
        : ''
      const hasAssistantContent = canonicalText.trim().length > 0

      if (hasAssistantContent) {
        newMessages.push({
          type: 'assistant_turn',
          source: 'agent',
          content: textParts(canonicalText),
          strategyId: event.strategyId,
        })
      }

      let nextFork: ForkMemoryState = { ...fork, messages: newMessages, currentTurnId: null }

      if (fork.pendingResultItems.length === 0 && !isCancelled && event.result.success) {
        nextFork = {
          ...nextFork,
          pendingResultItems: [
            ...nextFork.pendingResultItems,
            { kind: 'no_tools_or_messages' },
          ],
        }
      }

      const hasSubstantivePendingResults = nextFork.pendingResultItems.some(item => item.kind !== 'no_tools_or_messages')

      if (!hasAssistantContent && !isCancelled && event.result.success && !hasSubstantivePendingResults) {
        nextFork = {
          ...nextFork,
          messages: [
            ...nextFork.messages,
            {
              type: 'assistant_turn',
              source: 'agent',
              content: textParts('(empty response)'),
              strategyId: event.strategyId,
            },
          ],
        }
        nextFork = enqueueResult(nextFork, toResultError({ message: EMPTY_RESPONSE_ERROR }), event.timestamp)
      }

      const hasError = !event.result.success
      const errorMessage = hasError && 'error' in event.result ? event.result.error : undefined
      if (nextFork.pendingResultItems.length > 0) {
        nextFork = enqueueResult(
          nextFork,
          toResultTurnResults({ items: nextFork.pendingResultItems }),
          event.timestamp,
        )
      }
      if (errorMessage) {
        nextFork = enqueueResult(nextFork, toResultError({ message: errorMessage }), event.timestamp)
      }
      if (event.result.success && event.result.errors && event.result.errors.length > 0) {
        for (const err of event.result.errors) {
          nextFork = enqueueResult(nextFork, toResultError({ message: err.message }), event.timestamp)
        }
      }
      if (event.result.success && event.result.oneshotLivenessTriggered) {
        nextFork = enqueueResult(nextFork, { kind: 'oneshot_liveness' }, event.timestamp)
      }
      if (event.result.success && event.result.yieldWorkerRetriggered) {
        nextFork = enqueueResult(nextFork, { kind: 'yield_worker_retrigger' }, event.timestamp)
      }
      if (isCancelled) {
        nextFork = enqueueResult(nextFork, toResultInterrupted(), event.timestamp)
      }

      return resetPendingTurnState(nextFork)
    },

    turn_unexpected_error: ({ event, fork, read }) => {
      if (fork.currentTurnId !== event.turnId) return fork
      const nextFork = flushQueue(
        enqueueResult(fork, toResultError({ message: event.message }), event.timestamp),
        read(TaskGraphProjection),
      )
      return resetPendingTurnState({ ...nextFork, currentTurnId: null })
    },

    interrupt: ({ fork }) => fork,

    compaction_completed: ({ event, fork }) => {
      const remainingMessages = fork.messages.slice(1 + event.compactedMessageCount)
      const sessionContext: Message = event.refreshedContext
        ? { type: 'session_context', source: 'system', content: textParts(buildSessionContextContent(event.refreshedContext)) }
        : fork.messages[0]

      const summaryMessage: Message = { type: 'compacted', source: 'system', content: textParts(event.summary) }
      return { ...fork, messages: [sessionContext, summaryMessage, ...remainingMessages], currentChainId: null }
    },
  },

  globalEventHandlers: {
    task_assigned: ({ event, state }) => {
      if (!event.workerInfo) return state
      return state
    },

    agent_created: ({ event, state }) => {
      const { forkId, parentForkId } = event
      const parentState = state.forks.get(parentForkId)
      if (!parentState) throw new Error(`Parent fork ${parentForkId} not found in MemoryProjection`)

      const normalizedContext = typeof event.context === 'string' ? event.context : ''
      const contextMessage: Message[] = normalizedContext
        ? [{ type: 'fork_context', source: 'system', content: textParts(normalizedContext) }]
        : []

      let newForkState: ForkMemoryState = {
        messages: [...contextMessage],
        queuedEntries: [],
        currentTurnId: null,
        currentChainId: null,
        pendingPresenceText: null,
        nextQueueSeq: 0,
        pendingResultItems: [],
        pendingParentMessages: new Map(),
      }

      newForkState = enqueueTimeline(
        newForkState,
        toTimelineParentMessage({ timestamp: event.timestamp, text: event.message }),
        event.timestamp,
      )

      return { ...state, forks: new Map(state.forks).set(forkId, newForkState) }
    },
  },

  signalHandlers: on => [
    on(OutboundMessagesProjection.signals.messageCompleted, ({ value, state, read }) => {
      if (value.userFacing) return state

      const targetForkId = value.targetForkId
      if (targetForkId === undefined) return state
      const targetState = state.forks.get(targetForkId)
      if (!targetState) return state

      const agentState = read(AgentStatusProjection)
      const sender = value.forkId === null ? null : getAgentByForkId(agentState, value.forkId)
      const senderAgentId = sender?.agentId ?? 'lead'

      if (value.destination.kind === 'worker') {
        return {
          ...state,
          forks: new Map(state.forks).set(
            targetForkId,
            enqueueTimeline(
              targetState,
              toTimelineParentMessage({ timestamp: value.timestamp, text: value.text }),
              value.timestamp,
            ),
          ),
        }
      }

      const atom: AgentAtom = {
        kind: 'message',
        timestamp: value.timestamp,
        direction: 'to_lead',
        text: value.text,
      }

      return {
        ...state,
        forks: new Map(state.forks).set(
          targetForkId,
          enqueueAgentAtomBlock(targetState, {
            timestamp: value.timestamp,
            agentId: senderAgentId,
            role: sender?.role ?? 'lead',
            atoms: [atom],
          }),
        ),
      }
    }),

    on(SubagentActivityProjection.signals.unseenActivityAvailable, ({ value, state, read }) => {
      const parentState = state.forks.get(value.parentForkId)
      if (!parentState) return state

      let nextParent = parentState
      for (const item of value.entries) {
        const atoms: AgentAtom[] = []
        if (item.prose) {
          atoms.push({ kind: 'thought', timestamp: value.timestamp, text: item.prose })
        }

        // Skip empty agent blocks — nothing to show
        if (atoms.length === 0) continue

        // Resolve role from agent registry
        const agentState = read(AgentStatusProjection)
        const agent = agentState.agents.get(item.agentId)
        if (!agent) continue

        nextParent = enqueueAgentAtomBlock(nextParent, {
          timestamp: value.timestamp,
          agentId: item.agentId,
          role: agent.role,
          atoms,
        })
      }

      return {
        ...state,
        forks: new Map(state.forks).set(value.parentForkId, nextParent),
      }
    }),

    on(AgentStatusProjection.signals.agentBecameIdle, ({ value, state, read }) => {
      const parentState = state.forks.get(value.parentForkId)
      if (!parentState) return state

      const idleAtom: AgentAtom = {
        kind: 'idle',
        timestamp: value.timestamp,
        reason: value.reason === 'error' ? 'error' : value.reason === 'interrupt' ? 'interrupt' : 'stable',
      }

      let nextParent = enqueueAgentAtomBlock(parentState, {
        timestamp: value.timestamp,
        agentId: value.agentId,
        role: value.role,
        atoms: [idleAtom],
      })

      const taskGraphState = read(TaskGraphProjection)
      const linkedTask = findTaskForAgent(taskGraphState, { agentId: value.agentId, forkId: value.forkId })
      if (linkedTask) {
        nextParent = enqueueTimeline(
          nextParent,
          toTimelineTaskIdleHook({
            timestamp: value.timestamp,
            taskId: linkedTask.id,
            title: linkedTask.title,
            agentId: value.agentId,
          }),
          value.timestamp,
        )

        nextParent = enqueueTimeline(
          nextParent,
          toTimelineTaskTreeDirty({ timestamp: value.timestamp, taskId: linkedTask.id }),
          value.timestamp,
        )
      }

      return {
        ...state,
        forks: new Map(state.forks).set(value.parentForkId, nextParent),
      }
    }),

    on(AgentStatusProjection.signals.subagentUserKilled, ({ value, state }) => {
      const parentState = state.forks.get(value.parentForkId)
      if (!parentState) return state
      return {
        ...state,
        forks: new Map(state.forks).set(
          value.parentForkId,
          enqueueTimeline(
            parentState,
            toTimelineSubagentUserKilled({ timestamp: value.timestamp, agentId: value.agentId, agentType: value.role }),
            value.timestamp,
          ),
        ),
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
        }),
      }
    }),

    on(UserMessageResolutionProjection.signals.userMessageResolved, ({ value, state, read }) => {
      const targetFork = state.forks.get(value.forkId)
      if (!targetFork) return state

      const text = extractText(value.content)
      const imageAttachments: TimelineAttachment[] = (value.attachments ?? [])
        .filter((a): a is ImageAttachment => a.type === 'image')
        .map(a => ({ kind: 'image' as const, image: { type: 'image' as const, base64: a.base64, mediaType: a.mediaType, width: a.width, height: a.height } }))
      const mentionAttachments: TimelineAttachment[] = value.resolvedMentions.map(m => ({
        kind: 'mention' as const,
        ...m,
      }))
      const attachments = [...imageAttachments, ...mentionAttachments]
      const userEntry = toTimelineUserMessage({ timestamp: value.timestamp, text, attachments })

      let nextFork = enqueueTimeline(targetFork, userEntry, value.timestamp)

      if (value.forkId !== null) {
        const agent = getAgentByForkId(read(AgentStatusProjection), value.forkId)
        if (agent) {
          nextFork = enqueueTimeline(
            nextFork,
            toTimelineUserToAgent({ timestamp: value.timestamp, agentId: agent.agentId, text }),
            value.timestamp,
          )
        }
      }

      return {
        ...state,
        forks: new Map(state.forks).set(value.forkId, nextFork),
      }
    }),

    on(TaskGraphProjection.signals.taskCreated, ({ value, state, read }) => {
      const leadFork = state.forks.get(null)
      if (!leadFork) return state

      const taskGraphState = read(TaskGraphProjection)
      const task = taskGraphState.tasks.get(value.taskId)
      if (!task) return state

      let nextLead = enqueueTimeline(
        leadFork,
        toTimelineTaskTypeHook({
          timestamp: value.timestamp,
          taskId: task.id,
          title: task.title,
        }),
        value.timestamp,
      )

      nextLead = enqueueTimeline(
        nextLead,
        toTimelineTaskUpdate({
          timestamp: value.timestamp,
          action: 'created',
          taskId: task.id,
          title: task.title,
        }),
        value.timestamp,
      )

      nextLead = enqueueTimeline(
        nextLead,
        toTimelineTaskTreeDirty({ timestamp: value.timestamp, taskId: task.id }),
        value.timestamp,
      )

      return {
        ...state,
        forks: new Map(state.forks).set(null, nextLead),
      }
    }),

    on(TaskGraphProjection.signals.taskCompleted, ({ value, state, read, ambient }) => {
      const leadFork = state.forks.get(null)
      if (!leadFork) return state

      const task = read(TaskGraphProjection).tasks.get(value.taskId)

      let nextLead = enqueueTimeline(
        leadFork,
        toTimelineTaskUpdate({
          timestamp: value.timestamp,
          action: 'completed',
          taskId: value.taskId,
          title: task?.title,
        }),
        value.timestamp,
      )

      if (task) {
        nextLead = enqueueTimeline(
          nextLead,
          toTimelineTaskCompleteHook({
            timestamp: value.timestamp,
            taskId: task.id,
            title: task.title,
          }),
          value.timestamp,
        )
      }

      nextLead = enqueueTimeline(
        nextLead,
        toTimelineTaskTreeDirty({ timestamp: value.timestamp, taskId: value.taskId }),
        value.timestamp,
      )

      return {
        ...state,
        forks: new Map(state.forks).set(null, nextLead),
      }
    }),

    on(TaskGraphProjection.signals.taskCancelled, ({ value, state }) => {
      const leadFork = state.forks.get(null)
      if (!leadFork) return state

      let nextLead = enqueueTimeline(
        leadFork,
        toTimelineTaskUpdate({
          timestamp: value.timestamp,
          action: 'cancelled',
          taskId: value.taskId,
          cancelledCount: value.cancelledSubtree.length,
        }),
        value.timestamp,
      )

      nextLead = enqueueTimeline(
        nextLead,
        toTimelineTaskTreeDirty({ timestamp: value.timestamp, taskId: value.taskId }),
        value.timestamp,
      )

      return {
        ...state,
        forks: new Map(state.forks).set(null, nextLead),
      }
    }),

    on(TaskGraphProjection.signals.taskStatusChanged, ({ value, state, read }) => {
      const leadFork = state.forks.get(null)
      if (!leadFork) return state
      if (value.next === 'completed') return state

      const task = read(TaskGraphProjection).tasks.get(value.taskId)
      const action = 'status_changed'

      const nextLead = enqueueTimeline(
        leadFork,
        toTimelineTaskUpdate({
          timestamp: value.timestamp,
          action,
          taskId: value.taskId,
          title: task?.title,
          previousStatus: value.previous,
          nextStatus: value.next,
        }),
        value.timestamp,
      )

      return {
        ...state,
        forks: new Map(state.forks).set(null, nextLead),
      }
    }),
  ],
})