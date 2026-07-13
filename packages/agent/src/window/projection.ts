/**
 * WindowProjection (Forked)
 *
 * LLM conversation history, per-fork.
 * Each fork has independent message history and token budget tracking.
 */

import { Projection, Signal } from '@magnitudedev/event-core'
import { outcomeWillChainContinue, type AppEvent, type StrategyId, type ImageAttachment, type ObservationPart } from '../events'
import { present } from '../errors'
import { getAgentByForkId, AgentLifecycleProjection, hasActiveWorkers } from '../projections/agent-lifecycle'
import { WorkerActivityProjection } from '../projections/worker-activity'
import { OutboundMessagesProjection } from '../projections/outbound-messages'
import { HarnessStateProjection } from '../projections/harness-state'
import { DetachedProcessProjection } from '../projections/detached-process'
import { GoalProjection } from '../projections/goal'
import { buildSessionContextContent } from '../prompts/session-context'
import { TASK_TREE_COMPLETION_REMINDER } from '../prompts/task-tree'
import { renderGoalEarlyStopInjection, renderGoalStartedInjection } from '../prompts/goal'
import { SkillsAmbient } from '../ambient/skills-ambient'
import { ConfigAmbient, getSlotConfigForRole } from '../ambient/config-ambient'
import { SessionOptionsAmbient } from '../ambient/session-ambient'

import { UserMessageResolutionProjection } from '../projections/user-message-resolution'
import { TaskGraphProjection, type TaskGraphState, type TaskRecord } from '../projections/task-graph'
import { TaskAssignmentProjection, type TaskAssignmentRow } from '../projections/task-assignment'

import type { UserPart, ImageMediaType } from '@magnitudedev/ai'
import { Option } from 'effect'
import { textParts } from '../content'

import { EMPTY_RESPONSE_ERROR } from '../prompts/error-states'
import type {
  TimelineEntry,
  TimelineAttachment,
  AgentAtom,
} from './inbox/types'
import type { CompletedTurn, TurnFeedback } from './types'
import {
  toTimelineUserMessage,
  toTimelineCoordinatorMessage,
  toTimelineUserToAgent,
  toTimelineUserBashCommand,
  toTimelineObservation,
  toTimelineAgentBlock,
  toTimelineSubagentUserKilled,
  toTimelineTaskTypeHook,
  toTimelineTaskIdleHook,
  toTimelineTaskCompleteHook,
  toTimelineTaskTreeDirty,
  toTimelineTaskTreeView,
  toTimelineTaskUpdate,
  toTimelineTaskReassigned,
  toTimelineDetachedProcessExited,
  toTimelineTurnStart,
  toTimelineTurnEnd,
  toTimelineEscalation,
  toTimelineBackgroundProcesses,
} from './inbox/compose'

import type { ForkWindowState, WindowEntry, QueuedTimelineEntry } from './types'
import { ForkWindowStateSchema } from './types'
import {
  estimateContentEntry,
  estimateTurnEntry,
  estimateContextEntry,
  estimateSystemPromptTokens,
  estimateObserverTurnEntry,
  estimateAdvisorResponseEntry,
  computeTokenEstimate,
} from './estimate'
import { isRoleId } from '../agents/role-validation'
import { getForkInfo } from '../agents/registry'
import { COMPACTION_FALLBACK_KEEP_RATIO } from '../constants'
import { compactionSignals, type CompactionInjectedSignal } from '../projections/compaction-signals'

const compactionInjectedSignal = Signal.fromDef<
  CompactionInjectedSignal,
  unknown
>(compactionSignals.compactionInjected, 'Compaction')

function extractText(parts: readonly UserPart[]): string {
  return parts
    .filter((p): p is Extract<UserPart, { _tag: 'TextPart' }> => p._tag === 'TextPart')
    .map(p => p.text)
    .join('')
}

function makeGoalInjectionEntry(text: string): Extract<WindowEntry, { type: 'goal_injection' }> {
  const content = textParts(text)
  return {
    type: 'goal_injection',
    source: 'system',
    content,
    estimatedTokens: estimateContentEntry(content),
  }
}

function appendTimeline(
  messages: readonly WindowEntry[],
  timeline: readonly TimelineEntry[],
): { messages: readonly WindowEntry[]; addedTokens: number } {
  if (timeline.length === 0) return { messages, addedTokens: 0 }

  const estimatedTokens = estimateContextEntry(timeline)

  // Always create a new context entry — never merge into an existing one.
  // Merging would mutate a cached message's content, invalidating the prefix cache.
  // Adjacent UserMessages are coalesced at render time (full.ts / shared.ts).
  return {
    messages: [...messages, { type: 'context', source: 'system', timeline: [...timeline], estimatedTokens }],
    addedTokens: estimatedTokens,
  }
}

function enqueueTimeline(
  fork: ForkWindowState,
  entry: TimelineEntry,
  timestamp: number,
  coalesceKey: Option.Option<string> = Option.none(),
): ForkWindowState {
  const seq = fork.nextQueueSeq
  const queued: QueuedTimelineEntry = { timestamp, seq, entry, coalesceKey }
  const queuedTimeline = Option.isSome(coalesceKey)
    ? [...fork.queuedTimeline.filter(q => !Option.isSome(q.coalesceKey) || q.coalesceKey.value !== coalesceKey.value), queued]
    : [...fork.queuedTimeline, queued]
  return { ...fork, queuedTimeline, nextQueueSeq: seq + 1 }
}

function flushQueue(fork: ForkWindowState, taskGraphState: TaskGraphState, taskWorkerState: { rows: ReadonlyMap<string, TaskAssignmentRow> }): ForkWindowState {
  const sorted = [...fork.queuedTimeline].sort((a, b) => (a.timestamp - b.timestamp) || (a.seq - b.seq))
  const timeline: TimelineEntry[] = []
  const dirtyTaskIds = new Set<string>()
  let latestDirtyTimestamp: number | null = null

  for (const queued of sorted) {
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
    const renderedTree = renderTaskTreesForTaskIds(taskGraphState, taskWorkerState, Array.from(dirtyTaskIds))
    if (renderedTree) {
      timeline.push(
        toTimelineTaskTreeView({
          timestamp: latestDirtyTimestamp,
          renderedTree,
        }),
      )
    }
  }

  const { messages, addedTokens } = appendTimeline(fork.messages, timeline)
  const messageTokens = fork.messageTokens + addedTokens
  return {
    ...fork,
    messages,
    queuedTimeline: [],
    messageTokens,
    tokenEstimate: computeTokenEstimate(
      fork.systemPromptTokens, messageTokens,
      fork.lastAnchoredTotal, fork.lastAnchoredMessageTokens,
    ),
  }
}

function enqueueAgentAtomBlock(
  fork: ForkWindowState,
  args: {
    timestamp: number
    agentId: string
    role: string
    status: string
    atoms: readonly AgentAtom[]
  },
): ForkWindowState {
  if (args.atoms.length === 0) return fork

  const last = fork.queuedTimeline[fork.queuedTimeline.length - 1]
  if (
    last
    && last.entry.kind === 'agent_block'
    && last.entry.agentId === args.agentId
  ) {
    const mergedEntry = toTimelineAgentBlock({
      timestamp: last.entry.timestamp,
      firstAtomTimestamp: last.entry.firstAtomTimestamp,
      lastAtomTimestamp: args.atoms[args.atoms.length - 1]!.timestamp,
      agentId: last.entry.agentId,
      role: last.entry.role,
      status: args.status,
      atoms: [...last.entry.atoms, ...args.atoms],
    })

    const mergedQueued: QueuedTimelineEntry = {
      ...last,
      timestamp: Math.min(last.timestamp, args.timestamp),
      entry: mergedEntry,
    }

    return {
      ...fork,
      queuedTimeline: [...fork.queuedTimeline.slice(0, -1), mergedQueued],
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
      status: args.status,
      atoms: args.atoms,
    }),
    args.timestamp,
  )
}

function toUserPartFromObservation(part: ObservationPart): UserPart {
  if (part.type === 'text') {
    return { _tag: 'TextPart', text: part.text }
  }

  return {
    _tag: 'ImagePart',
    data: part.base64,
    mediaType: part.mediaType as ImageMediaType,
    ...(part.dimensions ? { dimensions: part.dimensions } : {}),
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

function renderTaskSubtree(
  state: TaskGraphState,
  taskWorkerState: { rows: ReadonlyMap<string, TaskAssignmentRow> },
  taskId: string,
): string {
  const task = state.tasks.get(taskId)
  if (!task) return ''

  const status = task.status === 'completed' ? 'completed' : 'pending'
  const row = taskWorkerState.rows.get(taskId)
  const workerState = row?.workerState
  const hasWorker = task.worker && task.worker.role !== 'user'

  let xml = `<task title="${escapeXmlAttr(task.title)}" id="${escapeXmlAttr(task.id)}" status="${status}">`

  if (hasWorker && task.worker) {
    const workerStatus = workerState?.status ?? 'idle'
    xml += `\n  <worker id="${escapeXmlAttr(task.worker.agentId)}" role="${task.worker.role}" status="${workerStatus}"/>`
  }

  if (task.childIds.length > 0) {
    for (const childId of task.childIds) {
      const childXml = renderTaskSubtree(state, taskWorkerState, childId)
      if (childXml) xml += '\n' + indentXml(childXml, 1)
    }
  }

  xml += `</task>`
  return xml
}

function escapeXmlAttr(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function indentXml(xml: string, depth: number): string {
  const prefix = '  '.repeat(depth)
  return xml.split('\n').map(line => prefix + line).join('\n')
}

function renderTaskTreesForTaskIds(
  state: TaskGraphState,
  taskWorkerState: { rows: ReadonlyMap<string, TaskAssignmentRow> },
  taskIds: readonly string[],
): string {
  const roots = new Set<string>()
  for (const taskId of taskIds) {
    roots.add(findRootTaskId(state, taskId))
  }

  const renderedTrees = Array.from(roots)
    .map(rootId => renderTaskSubtree(state, taskWorkerState, rootId))
    .filter(Boolean)

  if (renderedTrees.length === 0) return ''

  if (renderedTrees.length === 1) return renderedTrees[0]

  return `<tasks>\n${renderedTrees.map(t => indentXml(t, 1)).join('\n')}\n</tasks>`
}

function findTaskForAgent(state: TaskGraphState, args: { agentId: string, forkId: string }): TaskRecord | null {
  for (const task of Array.from(state.tasks.values())) {
    if (task.worker?.agentId === args.agentId || task.worker?.forkId === args.forkId) return task
  }
  return null
}

import type { TrackedProcess, DetachedProcessState } from '../projections/detached-process'
import type { ForkedState } from '@magnitudedev/event-core'

/** Merge all fork processes into a single map (for root fork view). */
function mergeAllForkProcesses(
  detachedForkedState: ForkedState<DetachedProcessState>,
): ReadonlyMap<number, TrackedProcess> {
  const merged = new Map<number, TrackedProcess>()
  for (const [, forkState] of detachedForkedState.forks) {
    for (const [pid, proc] of forkState.processes) {
      merged.set(pid, proc)
    }
  }
  return merged
}

/** Emit tokenEstimateChanged signal when tokenEstimate changes between old and new fork state. */
function emitIfChanged(
  oldFork: ForkWindowState,
  newFork: ForkWindowState,
  forkId: string | null,
  emit: { tokenEstimateChanged: (v: { forkId: string | null; tokenEstimate: number }) => void },
): void {
  if (newFork.tokenEstimate !== oldFork.tokenEstimate) {
    emit.tokenEstimateChanged({ forkId, tokenEstimate: newFork.tokenEstimate })
  }
}


export const WindowProjection = Projection.defineForked<AppEvent>()({
  name: 'Window',
  forkState: ForkWindowStateSchema,
  reads: [AgentLifecycleProjection, WorkerActivityProjection, OutboundMessagesProjection, UserMessageResolutionProjection, TaskGraphProjection, TaskAssignmentProjection, HarnessStateProjection, DetachedProcessProjection, GoalProjection] as const,
  ambients: [SkillsAmbient, ConfigAmbient, SessionOptionsAmbient] as const,
  signals: {
    tokenEstimateChanged: Signal.create<{ forkId: string | null; tokenEstimate: number }>('Window/tokenEstimateChanged'),
  },
  initialFork: {
    messages: [],
    queuedTimeline: [],
    currentTurnId: null,
    currentChainId: null,
    nextQueueSeq: 0,
    _activeMessageIsCoordinator: false,
    _coordinatorChars: 0,
    tokenEstimate: 0,
    messageTokens: 0,
    systemPromptTokens: 0,
    lastAnchoredTotal: null,
    lastAnchoredMessageTokens: null,
    autopilotEnabled: false,
    consumerAutopilotKnowledge: { advisor: null, leader: null },
  },

  eventHandlers: {
    session_initialized: ({ event, fork, emit, ambient }) => {
      const content = buildSessionContextContent(event.context)
      const contentParts = textParts(content)
      const entryTokens = estimateContentEntry(contentParts)
      const sessionMsg: WindowEntry = { type: 'session_context', source: 'system', content: contentParts, estimatedTokens: entryTokens }

      const skills = ambient.get(SkillsAmbient)
      const configState = ambient.get(ConfigAmbient)
      const sessionOptions = ambient.get(SessionOptionsAmbient)
      const sysPromptTokens = estimateSystemPromptTokens('leader', skills, configState, { solo: sessionOptions.solo, systemPromptOverride: sessionOptions.systemPromptOverride })
      const messageTokens = fork.messageTokens + entryTokens
      const tokenEstimate = sysPromptTokens + messageTokens

      const result: ForkWindowState = {
        ...fork,
        messages: [sessionMsg, ...fork.messages],
        messageTokens,
        systemPromptTokens: sysPromptTokens,
        tokenEstimate,
      }
      emitIfChanged(fork, result, event.forkId, emit)
      return result
    },

    autopilot_toggled: ({ event, fork }) => {
      // TEMPORARILY DISABLED: autopilot state should not enter the context window.
      // return { ...fork, autopilotEnabled: event.enabled }
      return fork
    },

    goal_started: ({ event, fork, emit }) => {
      const entry = makeGoalInjectionEntry(renderGoalStartedInjection(event.objective))
      const messageTokens = fork.messageTokens + entry.estimatedTokens
      const result: ForkWindowState = {
        ...fork,
        messages: [...fork.messages, entry],
        messageTokens,
        tokenEstimate: computeTokenEstimate(
          fork.systemPromptTokens,
          messageTokens,
          fork.lastAnchoredTotal,
          fork.lastAnchoredMessageTokens,
        ),
      }
      emitIfChanged(fork, result, event.forkId, emit)
      return result
    },

    skill_activated: ({ event, fork }) => {
      if (event.source !== 'user') return fork
      const text = event.message ? `/${event.skillName} ${event.message}` : `/${event.skillName}`
      const entry = toTimelineUserMessage({ timestamp: event.timestamp, text, attachments: [], synthetic: Option.none() })
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

    turn_started: ({ event, fork, read, emit }) => {
      let nextFork = { ...fork, _activeMessageIsCoordinator: false, _coordinatorChars: 0 }

      nextFork = enqueueTimeline(
        nextFork,
        toTimelineTurnStart({ timestamp: event.timestamp, turnId: event.turnId }),
        event.timestamp,
      )

      const taskGraphState = read(TaskGraphProjection)
      const taskWorkerState = read(TaskAssignmentProjection)
      const flushed = flushQueue(nextFork, taskGraphState, taskWorkerState)

      const result: ForkWindowState = {
        ...flushed,
        currentTurnId: event.turnId,
        currentChainId: event.chainId,
      }
      emitIfChanged(fork, result, event.forkId, emit)
      return result
    },

    observations_captured: ({ event, fork, read, emit }) => {
      if (fork.currentTurnId !== event.turnId) return fork
      const nextFork = enqueueTimeline(
        fork,
        toTimelineObservation({ timestamp: event.timestamp, parts: event.parts.map(toUserPartFromObservation) }),
        event.timestamp,
      )
      const result = flushQueue(nextFork, read(TaskGraphProjection), read(TaskAssignmentProjection))
      emitIfChanged(fork, result, event.forkId, emit)
      return result
    },

    message_start: ({ event, fork }) => {
      if (fork.currentTurnId !== event.turnId) return fork
      return { ...fork, _activeMessageIsCoordinator: event.destination.kind === 'coordinator' }
    },

    message_chunk: ({ event, fork }) => {
      if (fork.currentTurnId !== event.turnId) return fork
      if (!fork._activeMessageIsCoordinator) return fork
      return { ...fork, _coordinatorChars: fork._coordinatorChars + event.text.length }
    },

    message_end: ({ event, fork }) => {
      if (fork.currentTurnId !== event.turnId) return fork
      return { ...fork, _activeMessageIsCoordinator: false }
    },

    turn_outcome: ({ event, fork, read, emit }) => {
      if (fork.currentTurnId !== event.turnId) return fork

      const outcome = 'outcome' in event
        ? event.outcome
        : 'result' in event
          ? (event as any).result
          : { _tag: 'SystemError' as const, message: (event as any).message ?? 'Unknown error' }

      // Read the harness state to build CompletedTurn
      const harness = read(HarnessStateProjection)
      const canonicalState = harness.canonical

      // Build feedback
      const feedback: TurnFeedback[] = []
      if (fork._coordinatorChars > 0) {
        feedback.push({ kind: 'message_ack', destination: 'coordinator', chars: fork._coordinatorChars })
      }

      const hasContent = Option.isSome(canonicalState.assistantMessage.text)
        || Option.isSome(canonicalState.assistantMessage.reasoning)
        || (Option.isSome(canonicalState.assistantMessage.toolCalls) && canonicalState.assistantMessage.toolCalls.value.length > 0)
        || canonicalState.toolResults.length > 0

      switch (outcome._tag) {
        case 'Completed': {
          if (!hasContent) {
            feedback.push({ kind: 'error', message: EMPTY_RESPONSE_ERROR })
          }

          for (const fb of outcome.completion.feedback) {
            switch (fb._tag) {
              case 'InvalidMessageDestination':
                feedback.push({ kind: 'error', message: fb.message })
                break
            }
          }
          break
        }

        case 'Overthinking': {
          const fb = present(outcome).llmFeedback
          if (fb) feedback.push({ kind: 'overthinking', message: fb })
          break
        }

        case 'ConnectionFailure':
        case 'StreamFailed':
        case 'ProviderNotReady':
        case 'ContextWindowExceeded':
        case 'OutputTruncated':
        case 'SafetyStop':
        case 'UnexpectedError': {
          const fb = present(outcome).llmFeedback
          if (fb) feedback.push({ kind: 'error', message: fb })
          break
        }

        case 'Cancelled':
          feedback.push({ kind: 'interrupted' })
          break

        case 'SystemError':
          feedback.push({ kind: 'error', message: outcome.message })
          break
      }

      const commitPolicy = event.commitPolicy ?? (
        outcome._tag === 'Completed'
          ? { _tag: 'commitCleanTurn' as const }
          : { _tag: 'commitErrorOnly' as const }
      )
      const shouldCommitAssistantContent = commitPolicy._tag === 'commitCleanTurn'

      // Build CompletedTurn from harness canonical state. Failed attempts may
      // have displayed partial assistant chunks; commit policy decides whether
      // those chunks become future prompt history.
      const completedTurn: CompletedTurn = {
        turnId: event.turnId,
        assistant: shouldCommitAssistantContent ? canonicalState.assistantMessage : { _tag: 'AssistantMessage', reasoning: Option.none(), text: Option.none(), toolCalls: Option.none() },
        toolResults: shouldCommitAssistantContent ? [...canonicalState.toolResults] : [],
        feedback,
        clean: outcome._tag === 'Completed',
      }

      const newMessages: WindowEntry[] = [...fork.messages]
      let turnEntryTokens = 0

      if ((shouldCommitAssistantContent && hasContent) || feedback.length > 0) {
        turnEntryTokens = estimateTurnEntry(completedTurn)
        newMessages.push({
          type: 'assistant_turn',
          source: 'agent',
          turn: completedTurn,
          strategyId: event.strategyId,
          estimatedTokens: turnEntryTokens,
        })
      }

      // Insert advisor_response entries for message_advisor tool results
      let advisorResponseTokens = 0
      if (shouldCommitAssistantContent) {
        for (const tr of completedTurn.toolResults) {
          if (tr.toolName === 'message_advisor' && tr.result._tag === 'Success') {
            const content = typeof tr.result.output === 'string' ? tr.result.output : String(tr.result.output)
            const entryTokens = estimateAdvisorResponseEntry({ content })
            advisorResponseTokens += entryTokens
            newMessages.push({
              type: 'advisor_response',
              source: 'system',
              mode: 'advice',
              content,
              estimatedTokens: entryTokens,
            })
          }
        }
      }

      let goalInjectionTokens = 0
      if (event.forkId === null && !outcomeWillChainContinue(event.outcome)) {
        const goalState = read(GoalProjection)
        const agentStatus = read(AgentLifecycleProjection)
        const isUserInterrupt = event.outcome._tag === 'Cancelled' && event.outcome.reason._tag === 'UserInterrupt'
        if (goalState.active && !isUserInterrupt && !hasActiveWorkers(agentStatus)) {
          const entry = makeGoalInjectionEntry(renderGoalEarlyStopInjection(goalState.active.objective))
          goalInjectionTokens = entry.estimatedTokens
          newMessages.push(entry)
        }
      }

      const messageTokens = fork.messageTokens + turnEntryTokens + advisorResponseTokens + goalInjectionTokens

      // Apply API anchor if available
      let nextFork: ForkWindowState
      if (event.inputTokens != null) {
        // Anchor: API measured inputTokens for this turn's prompt.
        // The new turn entry wasn't in the prompt, so anchor total includes it separately.
        nextFork = {
          ...fork,
          messages: newMessages,
          currentTurnId: null,
          messageTokens,
          lastAnchoredTotal: event.inputTokens + turnEntryTokens + advisorResponseTokens + goalInjectionTokens,
          lastAnchoredMessageTokens: messageTokens,
          tokenEstimate: event.inputTokens + turnEntryTokens + advisorResponseTokens + goalInjectionTokens,
        }
      } else {
        nextFork = {
          ...fork,
          messages: newMessages,
          currentTurnId: null,
          messageTokens,
          tokenEstimate: computeTokenEstimate(
            fork.systemPromptTokens, messageTokens,
            fork.lastAnchoredTotal, fork.lastAnchoredMessageTokens,
          ),
        }
      }

      // Flush queued entries on top of (possibly anchored) base
      nextFork = flushQueue(nextFork, read(TaskGraphProjection), read(TaskAssignmentProjection))

      nextFork = enqueueTimeline(
        nextFork,
        toTimelineTurnEnd({ timestamp: event.timestamp, turnId: event.turnId }),
        event.timestamp,
      )
      nextFork = flushQueue(nextFork, read(TaskGraphProjection), read(TaskAssignmentProjection))

      // TEMPORARILY DISABLED: autopilot context tracking.
      // Update consumer autopilot knowledge after advisor response or leader turn.
      // if (advisorResponseTokens > 0) {
      //   nextFork = {
      //     ...nextFork,
      //     consumerAutopilotKnowledge: {
      //       ...nextFork.consumerAutopilotKnowledge,
      //       advisor: nextFork.autopilotEnabled,
      //     },
      //   }
      // }
      // if (event.forkId === null && advisorResponseTokens === 0) {
      //   nextFork = {
      //     ...nextFork,
      //     consumerAutopilotKnowledge: {
      //       ...nextFork.consumerAutopilotKnowledge,
      //       leader: nextFork.autopilotEnabled,
      //     },
      //   }
      // }

      emitIfChanged(fork, nextFork, event.forkId, emit)
      return nextFork
    },

    interrupt: ({ fork }) => fork,

    observer_outcome: ({ event, fork }) => {
      const entry: WindowEntry = {
        type: 'observer_turn',
        source: 'system',
        observerTurnId: event.observerTurnId,
        justification: event.justification,
        escalate: event.escalate,
        reasoning: event.reasoning,
        estimatedTokens: estimateObserverTurnEntry({ justification: event.justification, reasoning: event.reasoning }),
      }
      const messageTokens = fork.messageTokens + entry.estimatedTokens
      return {
        ...fork,
        messages: [...fork.messages, entry],
        messageTokens,
        tokenEstimate: computeTokenEstimate(
          fork.systemPromptTokens,
          messageTokens,
          fork.lastAnchoredTotal,
          fork.lastAnchoredMessageTokens,
        ),
      }
    },

  },

  globalEventHandlers: {
    turn_started: ({ event, state, read }) => {
      // Record background process status at turn time — frozen, never recomputed.
      // Runs after the per-fork handler (which already flushed the queue),
      // so we enqueue + flush here to ensure the status is in the current turn.
      const detachedForkedState = read(DetachedProcessProjection)
      const detachedState = event.forkId === null
        ? mergeAllForkProcesses(detachedForkedState)
        : detachedForkedState.forks.get(event.forkId)?.processes ?? new Map()
      const runningProcesses = [...detachedState.values()].filter(p => p.status === 'running')
      if (runningProcesses.length === 0) return state

      const entry = toTimelineBackgroundProcesses({
        timestamp: event.timestamp,
        processes: runningProcesses.map(proc => ({
          pid: proc.pid,
          command: proc.command,
          elapsedMs: event.timestamp - proc.startedAt,
          cpuPercent: proc.cpuPercent,
          rssBytes: proc.rssBytes,
          ownerAgentId: proc.ownerAgentId,
        })),
      })

      const targetFork = state.forks.get(event.forkId)
      if (!targetFork) return state
      const enqueued = enqueueTimeline(targetFork, entry, event.timestamp)
      const taskGraphState = read(TaskGraphProjection)
      const taskWorkerState = read(TaskAssignmentProjection)
      const flushed = flushQueue(enqueued, taskGraphState, taskWorkerState)
      return { ...state, forks: new Map(state.forks).set(event.forkId, flushed) }
    },

    shell_process_exited: ({ event, state, read }) => {
      const detachedForkedState = read(DetachedProcessProjection)
      // Find the process across all forks to get stdoutPath/stderrPath
      let stdoutPath = ''
      let stderrPath = ''
      for (const [, forkState] of detachedForkedState.forks) {
        const proc = forkState.processes.get(event.pid)
        if (proc) {
          stdoutPath = proc.stdoutPath
          stderrPath = proc.stderrPath
          break
        }
      }
      const entry = toTimelineDetachedProcessExited({
        timestamp: event.timestamp,
        pid: event.pid,
        command: event.command,
        exitCode: event.exitCode,
        stdoutPath,
        stderrPath,
      })

      let nextState = state
      // Enqueue into the owning fork
      const ownerFork = nextState.forks.get(event.forkId)
      if (ownerFork) {
        const nextFork = enqueueTimeline(ownerFork, entry, event.timestamp)
        nextState = { ...nextState, forks: new Map(nextState.forks).set(event.forkId, nextFork) }
      }
      // Also enqueue into root fork if root is different from owner
      if (event.forkId !== null) {
        const rootFork = nextState.forks.get(null)
        if (rootFork) {
          const nextRoot = enqueueTimeline(rootFork, entry, event.timestamp)
          nextState = { ...nextState, forks: new Map(nextState.forks).set(null, nextRoot) }
        }
      }
      return nextState
    },

    task_assigned: ({ event, state }) => {
      if (!event.workerInfo) return state
      return state
    },

    agent_task_changed: ({ event, state }) => {
      const workerFork = state.forks.get(event.forkId)
      if (workerFork) {
        const reassignedEntry = toTimelineTaskReassigned({
          timestamp: event.timestamp,
          oldTaskId: event.oldTaskId,
          newTaskId: event.newTaskId,
        })
        const nextFork = enqueueTimeline(workerFork, reassignedEntry, event.timestamp)
        let nextState: typeof state = { ...state, forks: new Map(state.forks).set(event.forkId, nextFork) }

        // Also inject notification into root fork (leader)
        const rootFork = nextState.forks.get(null)
        if (rootFork) {
          const leaderEntry = toTimelineTaskUpdate({
            timestamp: event.timestamp,
            action: 'status_changed',
            taskId: event.newTaskId,
            title: Option.none(),
            previousStatus: Option.some(`worker ${event.agentId} on ${event.oldTaskId}`),
            nextStatus: Option.some(`worker ${event.agentId} on ${event.newTaskId}`),
            cancelledCount: Option.none(),
          })
          const nextRoot = enqueueTimeline(rootFork, leaderEntry, event.timestamp)
          nextState = { ...nextState, forks: new Map(nextState.forks).set(null, nextRoot) }
        }

        return nextState
      }
      return state
    },

    agent_created: ({ event, state, ambient }) => {
      const { forkId, parentForkId } = event
      const parentState = state.forks.get(parentForkId)
      if (!parentState) throw new Error(`Parent fork ${parentForkId} not found in WindowProjection`)

      const normalizedContext = typeof event.context === 'string' ? event.context : ''
      const contextMessage: WindowEntry[] = normalizedContext
        ? (() => {
            const content = textParts(normalizedContext)
            return [{ type: 'fork_context' as const, source: 'system' as const, content, estimatedTokens: estimateContentEntry(content) }]
          })()
        : []

      const entryTokens = contextMessage.reduce((sum, e) => sum + e.estimatedTokens, 0)

      // Seed systemPromptTokens for the child fork
      const skills = ambient.get(SkillsAmbient)
      const configState = ambient.get(ConfigAmbient)
      const sessionOptions = ambient.get(SessionOptionsAmbient)
      const sysPromptTokens = isRoleId(event.role)
        ? estimateSystemPromptTokens(event.role, skills, configState, { solo: sessionOptions.solo, systemPromptOverride: sessionOptions.systemPromptOverride })
        : 0

      const coordinatorMessageEntry = toTimelineCoordinatorMessage({ timestamp: event.timestamp, text: event.message })

      let newForkState: ForkWindowState = {
        messages: [...contextMessage],
        queuedTimeline: [],
        currentTurnId: null,
        currentChainId: null,
        nextQueueSeq: 0,
        _activeMessageIsCoordinator: false,
        _coordinatorChars: 0,
        messageTokens: entryTokens,
        systemPromptTokens: sysPromptTokens,
        tokenEstimate: sysPromptTokens + entryTokens,
        lastAnchoredTotal: null,
        lastAnchoredMessageTokens: null,
        autopilotEnabled: false,
        consumerAutopilotKnowledge: { advisor: null, leader: null },
      }

      newForkState = enqueueTimeline(
        newForkState,
        coordinatorMessageEntry,
        event.timestamp,
      )

      return { ...state, forks: new Map(state.forks).set(forkId, newForkState) }
    },

    observer_outcome: ({ event, state, read }) => {
      if (!event.escalate) return state

      // Leader escalation: enqueue on root fork for escalation block rendering
      if (event.forkId === null) {
        const rootFork = state.forks.get(null)
        if (!rootFork) return state

        const escalationEntry = toTimelineEscalation({
          timestamp: event.timestamp,
          observedForkId: null,
          observedTurnId: event.observedTurnId,
          justification: event.justification,
          coalesceKey: Option.some('leader-escalation'),
        })

        const nextRoot = enqueueTimeline(rootFork, escalationEntry, event.timestamp, Option.some('leader-escalation'))
        return { ...state, forks: new Map(state.forks).set(null, nextRoot) }
      }

      // Worker escalation: show it on the parent fork so the leader can act.
      const agentState = read(AgentLifecycleProjection)
      const agent = getAgentByForkId(agentState, event.forkId)
      const targetForkId = agent?.parentForkId ?? null

      const targetFork = state.forks.get(targetForkId)
      if (!targetFork) return state

      const escalationEntry = toTimelineEscalation({
        timestamp: event.timestamp,
        observedForkId: event.forkId,
        observedTurnId: event.observedTurnId,
        justification: event.justification,
        coalesceKey: Option.none(),
      })

      const nextTarget = enqueueTimeline(targetFork, escalationEntry, event.timestamp)
      return { ...state, forks: new Map(state.forks).set(targetForkId, nextTarget) }
    },
  },

  signalHandlers: on => [
    on(OutboundMessagesProjection.signals.messageCompleted, ({ value, state, read }) => {
      if (value.userFacing) return state

      const targetForkId = value.targetForkId
      if (targetForkId === undefined) return state
      const targetState = state.forks.get(targetForkId)
      if (!targetState) return state

      const agentState = read(AgentLifecycleProjection)
      const sender = value.forkId === null ? null : getAgentByForkId(agentState, value.forkId)
      const senderAgentId = sender?.agentId ?? 'lead'

      if (value.destination.kind === 'worker') {
        return {
          ...state,
          forks: new Map(state.forks).set(
            targetForkId,
            enqueueTimeline(
              targetState,
              toTimelineCoordinatorMessage({ timestamp: value.timestamp, text: value.text }),
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
            status: sender?.status ?? 'idle',
            atoms: [atom],
          }),
        ),
      }
    }),

    on(WorkerActivityProjection.signals.unseenActivityAvailable, ({ value, state, read }) => {
      const parentState = state.forks.get(value.parentForkId)
      if (!parentState) return state

      let nextParent = parentState
      for (const item of value.entries) {
        const atoms: AgentAtom[] = []
        if (item.prose) {
          atoms.push({ kind: 'thought', timestamp: value.timestamp, text: item.prose })
        }

        if (atoms.length === 0) continue

        const agentState = read(AgentLifecycleProjection)
        const agent = agentState.agents.get(item.agentId)
        if (!agent) continue

        nextParent = enqueueAgentAtomBlock(nextParent, {
          timestamp: value.timestamp,
          agentId: item.agentId,
          role: agent.role,
          status: agent.status,
          atoms,
        })
      }

      return {
        ...state,
        forks: new Map(state.forks).set(value.parentForkId, nextParent),
      }
    }),

    on(AgentLifecycleProjection.signals.agentBecameIdle, ({ value, state, read }) => {
      const parentState = state.forks.get(value.parentForkId)
      if (!parentState) return state

      const idleAtom: AgentAtom = {
        kind: 'idle',
        timestamp: value.timestamp,
        reason: Option.some(value.reason === 'error' ? 'error' : value.reason === 'interrupt' ? 'interrupt' : 'stable'),
      }

      let nextParent = enqueueAgentAtomBlock(parentState, {
        timestamp: value.timestamp,
        agentId: value.agentId,
        role: value.role,
        status: 'idle',
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

    on(AgentLifecycleProjection.signals.subagentUserKilled, ({ value, state }) => {
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

    on(UserMessageResolutionProjection.signals.userMessageResolved, ({ value, state, read, emit }) => {
      const targetFork = state.forks.get(value.forkId)
      if (!targetFork) return state

      const text = extractText(value.content)
      const imageAttachments: TimelineAttachment[] = (value.attachments ?? [])
        .filter((a): a is ImageAttachment => a.type === 'image')
        .map(a => ({
          kind: 'image' as const,
          path: a.path,
          filename: a.filename,
          mediaType: a.mediaType,
          width: a.width,
          height: a.height,
        }))
      const mentionTimelineAttachments: TimelineAttachment[] = value.mentionResolutions.map((mention) => ({
        kind: 'mention' as const,
        attachment: mention.attachment,
        resolution: mention.status === 'resolved'
          ? {
              status: 'resolved' as const,
              content: mention.content,
              truncated: mention.truncated,
              originalBytes: mention.originalBytes,
            }
          : {
              status: 'failed' as const,
              reason: mention.reason,
            },
      }))
      const attachments = [...imageAttachments, ...mentionTimelineAttachments]
      const userEntry = toTimelineUserMessage({ timestamp: value.timestamp, text, attachments, synthetic: value.synthetic ? Option.some(true) : Option.none() })

      let nextFork = enqueueTimeline(targetFork, userEntry, value.timestamp)

      if (value.forkId !== null) {
        const agent = getAgentByForkId(read(AgentLifecycleProjection), value.forkId)
        if (agent) {
          nextFork = enqueueTimeline(
            nextFork,
            toTimelineUserToAgent({ timestamp: value.timestamp, agentId: agent.agentId, text }),
            value.timestamp,
          )
        }
      }

      // TEMPORARILY DISABLED: autopilot context tracking.
      // Update advisor autopilot knowledge after autopilot-generated user message.
      // if (value.synthetic) {
      //   nextFork = {
      //     ...nextFork,
      //     consumerAutopilotKnowledge: {
      //       ...nextFork.consumerAutopilotKnowledge,
      //       advisor: nextFork.autopilotEnabled,
      //     },
      //   }
      // }

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
          title: Option.some(task.title),
          previousStatus: Option.none(),
          nextStatus: Option.none(),
          cancelledCount: Option.none(),
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
          title: task ? Option.some(task.title) : Option.none(),
          previousStatus: Option.none(),
          nextStatus: Option.none(),
          cancelledCount: Option.none(),
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
          title: Option.none(),
          previousStatus: Option.none(),
          nextStatus: Option.none(),
          cancelledCount: Option.some(value.cancelledSubtree.length),
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
          title: task ? Option.some(task.title) : Option.none(),
          previousStatus: Option.some(value.previous),
          nextStatus: Option.some(value.next),
          cancelledCount: Option.none(),
        }),
        value.timestamp,
      )

      return {
        ...state,
        forks: new Map(state.forks).set(null, nextLead),
      }
    }),

    on(compactionInjectedSignal, ({ value, state, emit, ambient, read }) => {
      const fork = state.forks.get(value.forkId)
      if (!fork) return state

      const sessionContext: WindowEntry = value.refreshedContext
        ? (() => {
            const content = textParts(buildSessionContextContent(value.refreshedContext))
            return { type: 'session_context' as const, source: 'system' as const, content, estimatedTokens: estimateContentEntry(content) }
          })()
        : fork.messages[0]

      const remainingMessages = fork.messages.slice(1 + value.compactedMessageCount)

      let newMessages: readonly WindowEntry[]

      if (!value.compactionOutcome.isFallback) {
        // Structured compaction → compacted UserMessage
        const result = value.compactionOutcome.compactResult
        let text = '<compaction_summary>\n'
        text += `## Summary\n${result.summary}\n\n`
        text += `## Reflection\n${result.reflection}`

        if (result.files.length > 0) {
          text += '\n\n## Key Files'
          for (const file of result.files) {
            const ext = file.path.split('.').pop() || ''
            text += `\n\n### ${file.path}\n\`\`\`${ext}\n${file.content}\n\`\`\``
          }
        }

        text += '\n</compaction_summary>'

        const content = textParts(text)
        const compactionEntry: WindowEntry = {
          type: 'compacted',
          source: 'system',
          content,
          estimatedTokens: estimateContentEntry(content),
        }

        newMessages = [sessionContext, compactionEntry, ...remainingMessages]
      } else {
        // Fallback: raw tail preservation — keep latest 25% of softCap worth of messages
        const configState = ambient.get(ConfigAmbient)
        const agentStatus = read(AgentLifecycleProjection)
        const forkInfo = getForkInfo(agentStatus, value.forkId)
        const roleConfig = forkInfo ? getSlotConfigForRole(configState, forkInfo.roleId) : null
        const fallbackBudget = roleConfig
          ? roleConfig.softCap * COMPACTION_FALLBACK_KEEP_RATIO
          : fork.systemPromptTokens * 2 // reasonable fallback if config unavailable

        // Walk backwards from all messages (excluding session context) to find what fits
        const allNonSession = fork.messages.slice(1)
        let accumulated = 0
        let keepFrom = allNonSession.length
        for (let i = allNonSession.length - 1; i >= 0; i--) {
          if (accumulated + allNonSession[i].estimatedTokens > fallbackBudget) break
          accumulated += allNonSession[i].estimatedTokens
          keepFrom = i
        }

        newMessages = [sessionContext, ...allNonSession.slice(keepFrom)]
      }

      const messageTokens = newMessages.reduce((sum, e) => sum + e.estimatedTokens, 0)

      const result: ForkWindowState = {
        ...fork,
        messages: newMessages,
        currentChainId: null,
        messageTokens,
        lastAnchoredTotal: null,
        lastAnchoredMessageTokens: null,
        tokenEstimate: fork.systemPromptTokens + messageTokens,
      }

      const nextForks = new Map(state.forks).set(value.forkId, result)
      if (result.tokenEstimate !== fork.tokenEstimate) {
        emit.tokenEstimateChanged({ forkId: value.forkId, tokenEstimate: result.tokenEstimate })
      }
      return { ...state, forks: nextForks }
    }),
  ],
})
