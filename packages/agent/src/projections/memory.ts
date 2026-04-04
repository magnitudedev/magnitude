/**
 * MemoryProjection (Forked)
 *
 * LLM conversation history, per-fork.
 * Each fork has independent message history.
 */

import { Projection } from '@magnitudedev/event-core'
import type { ObservationPart } from '@magnitudedev/roles'
import type { AppEvent, ResponsePart, StrategyId, ImageAttachment } from '../events'
import { getAgentByForkId, AgentStatusProjection } from './agent-status'
import { SubagentActivityProjection } from './subagent-activity'
import { CanonicalTurnProjection } from './canonical-turn'
import { OutboundMessagesProjection } from './outbound-messages'
import { compactionSummaryTag, buildSessionContextContent } from '../prompts'
import { UserPresenceProjection } from './user-presence'
import { UserMessageResolutionProjection } from './user-message-resolution'
import { TaskGraphProjection, type TaskGraphState, type TaskRecord } from './task-graph'
import { getAgentDefinition, isValidVariant } from '../agents'
import { formatUserPresence, formatUserReturnedAfterAbsence } from '../prompts/presence'
import { formatSkillInitialPrompt } from '../prompts/skills'
import { ContentPart, ImageMediaType, textParts, wrapTextParts } from '../content'

import { EMPTY_RESPONSE_ERROR } from '../prompts/error-states'
import { formatInbox } from '../inbox/render'
import type {
  ResultEntry,
  TimelineEntry,
  TimelineAttachment,
  QueuedEntry,
  AgentAtom,
  PhaseCriteriaPayload,
  LifecycleReminderFormatterMap,
} from '../inbox/types'
import {
  toResultToolResults,
  toResultInterrupted,
  toResultError,
  toResultNoop,
  toTimelineUserMessage,
  toTimelineUserToAgent,
  toTimelineUserPresence,
  toTimelineObservation,
  toTimelineAgentBlock,
  toTimelineSubagentUserKilled,
  toTimelineSkillStarted,
  toTimelineSkillCompleted,
  toTimelinePhaseCriteria,
  toTimelinePhaseVerdict,
  toTimelineWorkflowPhase,
  toTimelineLifecycleHook,
  toTimelineTaskTypeHook,
  toTimelineTaskIdleHook,
  toTimelineTaskTreeDirty,
  toTimelineTaskTreeView,
} from '../inbox/compose'
import { builderRole } from '../agents/builder'
import { explorerRole } from '../agents/explorer'
import { plannerRole } from '../agents/planner'
import { reviewerRole } from '../agents/reviewer'

const lifecycleReminderFormatters: LifecycleReminderFormatterMap = {
  builder: {
    spawn: builderRole.lifecyclePrompts?.parentOnSpawn,
    idle: builderRole.lifecyclePrompts?.parentOnIdle,
  },
  explorer: {
    spawn: explorerRole.lifecyclePrompts?.parentOnSpawn,
    idle: explorerRole.lifecyclePrompts?.parentOnIdle,
  },
  planner: {
    spawn: plannerRole.lifecyclePrompts?.parentOnSpawn,
    idle: plannerRole.lifecyclePrompts?.parentOnIdle,
  },
  reviewer: {
    spawn: reviewerRole.lifecyclePrompts?.parentOnSpawn,
    idle: reviewerRole.lifecyclePrompts?.parentOnIdle,
  },
}

export type MessageSource = 'user' | 'agent' | 'system'

export type Message =
  | { readonly type: 'session_context'; readonly source: 'system'; readonly content: ContentPart[] }
  | {
      readonly type: 'assistant_turn'
      readonly source: 'agent'
      readonly content: ContentPart[]
      readonly strategyId: StrategyId
      readonly responseParts: readonly ResponsePart[]
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

export interface ForkMemoryState {
  readonly messages: readonly Message[]
  readonly queuedEntries: readonly QueuedEntry[]
  readonly currentTurnId: string | null
  readonly currentChainId: string | null
  readonly pendingPresenceText: string | null
  readonly nextQueueSeq: number
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

function countDescendants(state: TaskGraphState, taskId: string): number {
  const task = state.tasks.get(taskId)
  if (!task) return 0
  let count = 0
  const stack = [...task.childIds]
  while (stack.length > 0) {
    const currentId = stack.pop()
    if (!currentId) continue
    const current = state.tasks.get(currentId)
    if (!current) continue
    count += 1
    stack.push(...current.childIds)
  }
  return count
}

function renderTaskSubtree(state: TaskGraphState, taskId: string, depth: number): string[] {
  const task = state.tasks.get(taskId)
  if (!task) return []

  const indent = '  '.repeat(depth)

  if (task.status === 'archived') {
    const archivedCount = countDescendants(state, task.id)
    return [`${indent}[archived] ${task.title} (${archivedCount} tasks)`]
  }

  const status = task.status === 'completed' ? 'done' : task.status
  const assigneeStr = task.assignee === 'user' ? ', user' : ''
  const line = `${indent}[${status}] ${task.taskType}: ${task.title} (${task.id}${assigneeStr})`

  const childLines = task.childIds.flatMap(childId => renderTaskSubtree(state, childId, depth + 1))
  const directCompletedNonArchivedCount = task.childIds.reduce((count, childId) => {
    const child = state.tasks.get(childId)
    if (!child) return count
    return count + (child.status === 'completed' ? 1 : 0)
  }, 0)

  if (directCompletedNonArchivedCount > 8) {
    childLines.push(`${indent}  Consider archiving completed tasks`)
  }

  return [line, ...childLines]
}

function renderTaskTreesForTaskIds(state: TaskGraphState, taskIds: readonly string[]): string {
  const roots = new Set<string>()
  for (const taskId of taskIds) {
    roots.add(findRootTaskId(state, taskId))
  }

  return Array.from(roots)
    .map(rootId => renderTaskSubtree(state, rootId, 0).join('\n'))
    .filter(Boolean)
    .join('\n')
}

function findTaskForAgent(state: TaskGraphState, args: { agentId: string, forkId: string }): TaskRecord | null {
  for (const task of Array.from(state.tasks.values())) {
    if (task.worker?.agentId === args.agentId || task.worker?.forkId === args.forkId) return task
  }
  return null
}

function transformMessage(message: Message, timezone: string | null, perspective: Perspective): LLMMessage {
  const content = message.type === 'inbox'
    ? formatInbox({ results: message.results, timeline: message.timeline, timezone, lifecycleReminderFormatters })
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
  signals: {},
  initialFork: {
    messages: [],
    queuedEntries: [],
    currentTurnId: null,
    currentChainId: null,
    pendingPresenceText: null,
    nextQueueSeq: 0,
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

      const hadQueuedEntries = nextFork.queuedEntries.length > 0
      const taskGraphState = read(TaskGraphProjection)
      const flushed = flushQueue(nextFork, taskGraphState)

      let messages = flushed.messages
      const lastMessage = messages[messages.length - 1]
      if (!hadQueuedEntries && lastMessage?.source === 'agent') {
        messages = [...messages, { type: 'inbox', source: 'system', results: [toResultNoop()], timeline: [] }]
      }

      return {
        ...flushed,
        messages,
        currentTurnId: event.turnId,
        currentChainId: event.chainId,
        pendingPresenceText: null,
      }
    },

    tool_event: ({ fork }) => fork,

    observations_captured: ({ event, fork }) => {
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

      let nextFork: ForkMemoryState = { ...fork, messages: newMessages, currentTurnId: null }

      if (event.responseParts.length === 0 && !isCancelled && event.result.success) {
        nextFork = {
          ...nextFork,
          messages: [
            ...nextFork.messages,
            {
              type: 'assistant_turn',
              source: 'agent',
              content: textParts('(empty response)'),
              strategyId: event.strategyId,
              responseParts: [],
            },
          ],
        }
        nextFork = enqueueResult(nextFork, toResultError({ message: EMPTY_RESPONSE_ERROR }), event.timestamp)
      }

      const hasError = !event.result.success
      const errorMessage = hasError && 'error' in event.result ? event.result.error : undefined
      const observedResults = isCancelled ? [] : event.observedResults
      if (event.toolCalls.length > 0 || observedResults.length > 0) {
        nextFork = enqueueResult(
          nextFork,
          toResultToolResults({ toolCalls: event.toolCalls, observedResults }),
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
      if (isCancelled) {
        nextFork = enqueueResult(nextFork, toResultInterrupted(), event.timestamp)
      }

      return nextFork
    },

    turn_unexpected_error: ({ event, fork }) => {
      if (fork.currentTurnId !== event.turnId) return fork
      const nextFork = flushQueue(
        enqueueResult(fork, toResultError({ message: event.message }), event.timestamp),
        read(TaskGraphProjection),
      )
      return { ...nextFork, currentTurnId: null }
    },

    skill_started: ({ event, fork }) => {
      const firstPhase = event.skill.phases[0]
      return enqueueTimeline(
        fork,
        toTimelineSkillStarted({
          timestamp: event.timestamp,
          skillName: event.skill.name,
          firstPhase: firstPhase?.name,
          prompt: formatSkillInitialPrompt(event.skill),
        }),
        event.timestamp,
      )
    },

    phase_criteria_verdict: ({ event, fork }) => {
      let payload: PhaseCriteriaPayload
      if (event.criteriaType === 'agent') {
        payload = {
          source: 'agent',
          name: event.criteriaName,
          status: event.status === 'running' ? 'pending' : event.status,
          agentId: event.agentId,
          reason: 'reason' in event ? event.reason : undefined,
        }
      } else if (event.criteriaType === 'shell') {
        payload = {
          source: 'shell',
          name: event.criteriaName,
          status: event.status === 'running' ? 'pending' : event.status,
          command: event.command,
          reason: 'reason' in event ? event.reason : undefined,
        }
      } else {
        payload = {
          source: 'user',
          name: event.criteriaName,
          status: event.status,
          reason: event.reason,
        }
      }

      return enqueueTimeline(fork, toTimelinePhaseCriteria({ timestamp: event.timestamp, payload }), event.timestamp)
    },

    phase_verdict: ({ event, fork }) => {
      const verdictText = event.verdicts
        .map(v => `  <verdict name="${v.criteriaName}" passed="${v.passed}" reason="${v.reason}"/>`)
        .join('\n')

      let nextFork = enqueueTimeline(
        fork,
        toTimelinePhaseVerdict({
          timestamp: event.timestamp,
          passed: event.passed,
          verdictText,
          workflowCompleted: event.workflowCompleted,
        }),
        event.timestamp,
      )

      if (event.passed && !event.workflowCompleted && event.nextPhasePrompt) {
        nextFork = enqueueTimeline(
          nextFork,
          toTimelineWorkflowPhase({
            timestamp: event.timestamp,
            text: event.nextPhasePrompt,
          }),
          event.timestamp,
        )
      }

      return nextFork
    },

    skill_completed: ({ event, fork }) =>
      enqueueTimeline(
        fork,
        toTimelineSkillCompleted({ timestamp: event.timestamp, skillName: event.skillName }),
        event.timestamp,
      ),

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
    agent_created: ({ event, state }) => {
      const { forkId, parentForkId } = event
      const parentState = state.forks.get(parentForkId)
      if (!parentState) throw new Error(`Parent fork ${parentForkId} not found in MemoryProjection`)

      const normalizedContext = typeof event.context === 'string' ? event.context : ''
      const contextMessage: Message[] = normalizedContext
        ? [{ type: 'fork_context', source: 'system', content: textParts(normalizedContext) }]
        : []

      const newForkState: ForkMemoryState = {
        messages: [...contextMessage],
        queuedEntries: [],
        currentTurnId: null,
        currentChainId: null,
        pendingPresenceText: null,
        nextQueueSeq: 0,
      }

      const roleDef = isValidVariant(event.role) ? getAgentDefinition(event.role) : undefined
      const spawnReminder = roleDef?.lifecyclePrompts?.parentOnSpawn
      if (spawnReminder) {
        const updatedParent = enqueueTimeline(
          parentState,
          toTimelineLifecycleHook({ timestamp: event.timestamp, agentId: event.agentId, role: event.role, hookType: 'spawn' }),
          event.timestamp,
        )
        return { ...state, forks: new Map(state.forks).set(parentForkId, updatedParent).set(forkId, newForkState) }
      }

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

      const roleDef = isValidVariant(value.role) ? getAgentDefinition(value.role) : undefined
      const idleReminder = roleDef?.lifecyclePrompts?.parentOnIdle
      if (idleReminder) {
        nextParent = enqueueTimeline(
          nextParent,
          toTimelineLifecycleHook({ timestamp: value.timestamp, agentId: value.agentId, role: value.role, hookType: 'idle' }),
          value.timestamp,
        )
      }

      const taskGraphState = read(TaskGraphProjection)
      const linkedTask = findTaskForAgent(taskGraphState, { agentId: value.agentId, forkId: value.forkId })
      if (linkedTask) {
        nextParent = enqueueTimeline(
          nextParent,
          toTimelineTaskIdleHook({
            timestamp: value.timestamp,
            taskId: linkedTask.id,
            taskType: linkedTask.taskType,
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
          taskType: task.taskType,
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

    on(TaskGraphProjection.signals.taskCompleted, ({ value, state }) => {
      const leadFork = state.forks.get(null)
      if (!leadFork) return state

      const nextLead = enqueueTimeline(
        leadFork,
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

      const nextLead = enqueueTimeline(
        leadFork,
        toTimelineTaskTreeDirty({ timestamp: value.timestamp, taskId: value.taskId }),
        value.timestamp,
      )

      return {
        ...state,
        forks: new Map(state.forks).set(null, nextLead),
      }
    }),
  ],
})