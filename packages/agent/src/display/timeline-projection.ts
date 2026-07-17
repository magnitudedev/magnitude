/**
 * DisplayTimelineProjection (Forked)
 *
 * UI state with flat messages, per-fork.
 * Each fork has independent display state for its conversation.
 *
 * Key invariants:
 * - Queued messages always appear at the END of the message list
 * - New content (assistant messages, tools) is inserted BEFORE queued messages
 * - Queued messages are promoted to user_message on turn_started
 */

import { Signal, Projection } from '@magnitudedev/event-core'
import { DisplayMessage as ProtocolDisplayMessageSchema } from '@magnitudedev/protocol'
import type { ToolStepPresentation } from '@magnitudedev/protocol'
import { Effect, Option } from 'effect'
import type { AppEvent } from '../events'
import { outcomeWillChainContinue } from '../events'

import { AgentRoutingProjection } from '../projections/agent-routing'
import { AgentLifecycleProjection, getAgentByForkId } from '../projections/agent-lifecycle'
import { GoalProjection } from '../projections/goal'
import { TurnProjection } from '../projections/turn'
import { HarnessStateProjection } from '../projections/harness-state'

import { HIDDEN_TOOLS, isToolKey } from '../tools/toolkits'
import type { ToolKeyErased } from '../tools/types'
import type { ToolHandleFromSchema } from '../models/tool-handle-schema'

import {
  UserMessageDisplay,
  AssistantMessageDisplay,
  ThinkingMessage,
  ToolMessage,
  StatusIndicatorMessage,
  GoalStatusMessage,
  WorkerResumedMessage,
  WorkerFinishedMessage,
  WorkerKilledMessage,
  WorkerUserKilledMessage,
  DisplayTimelineState,
  DisplayMessage,
  ErrorDisplayMessage,
  ForkActivityMessage,
  PendingInboundCommunicationDisplay,
  AgentCommunicationMessage,
  DisplayTimelineStateSchema,
} from './types'

import {
  toErrorDisplayMessage,
  toPreview,
} from './helpers/messages'
import {
  appendDisplayMessage,
  readDisplayMessageById,
  tailDisplayMessageIds,
  type DisplayMessageSequence,
} from './helpers/message-sequence'

import { processThinkingChunk, heldBuffers, flushHeld } from './helpers/thinking'

import {
  incrementToolCount,
  totalToolsUsed,
} from './helpers/fork-activity'

import { presentToolState } from '../display-view/tool-presentation'

import { EMPTY_TOOL_COUNTS } from './constants'

export const DisplayTimelineMessages = Projection.addressed.sequence(ProtocolDisplayMessageSchema)

// ── Tool cluster mapping ────────────────────────────────────────
type DisplayTimelineMessagesHandle = DisplayMessageSequence

function forkIdKey(forkId: string | null): string {
  return forkId ?? 'root'
}

function addActiveToolProducer(
  fork: DisplayTimelineState,
  toolCallId: string
): DisplayTimelineState {
  return fork._activeToolCallIds.includes(toolCallId)
    ? fork
    : { ...fork, _activeToolCallIds: [...fork._activeToolCallIds, toolCallId] }
}

function removeActiveToolProducer(
  fork: DisplayTimelineState,
  toolCallId: string
): DisplayTimelineState {
  return fork._activeToolCallIds.includes(toolCallId)
    ? {
        ...fork,
        _activeToolCallIds: fork._activeToolCallIds.filter((candidate) => candidate !== toolCallId)
      }
    : fork
}

function appendMessageToFork(
  messages: DisplayTimelineMessagesHandle,
  fork: DisplayTimelineState,
  message: DisplayMessage
) {
  return Effect.map(
    appendDisplayMessage(messages, fork.messages, message),
    (index): DisplayTimelineState => ({
      ...fork,
      messages: index,
      _queuedUserMessageCount: message.type === 'queued_user_message'
        ? fork._queuedUserMessageCount + 1
        : fork._queuedUserMessageCount
    })
  )
}

function insertMessageIntoFork(
  messages: DisplayTimelineMessagesHandle,
  fork: DisplayTimelineState,
  message: DisplayMessage
) {
  return fork._queuedUserMessageCount === 0
    ? appendMessageToFork(messages, fork, message)
    : Effect.map(
      messages.replaceRange(
        fork.messages,
        fork.messages.totalCount - fork._queuedUserMessageCount,
        fork.messages.totalCount - fork._queuedUserMessageCount,
        [message]
      ),
      (index): DisplayTimelineState => ({
        ...fork,
        messages: index,
        _queuedUserMessageCount: message.type === 'queued_user_message'
          ? fork._queuedUserMessageCount + 1
          : fork._queuedUserMessageCount
      })
    )
}

function rememberCommunicationMessage(
  fork: DisplayTimelineState,
  streamId: string,
  messageId: string
): DisplayTimelineState {
  return {
    ...fork,
    _communicationMessageIdsByStreamId: {
      ...fork._communicationMessageIdsByStreamId,
      [streamId]: messageId
    }
  }
}

function forgetCommunicationMessage(
  fork: DisplayTimelineState,
  streamId: string
): DisplayTimelineState {
  const { [streamId]: _removed, ...remaining } = fork._communicationMessageIdsByStreamId
  return {
    ...fork,
    _communicationMessageIdsByStreamId: remaining
  }
}

/**
 * Release the fork's active thinking message: flush its held buffer into it,
 * drop it if it ends up empty. The message is located by `_thinkingMessageId`
 * alone — the fork state is the locator.
 */
function releaseActiveThinking(
  messages: DisplayTimelineMessagesHandle,
  fork: DisplayTimelineState
) {
  const thinkingMessageId = fork._thinkingMessageId
  if (!thinkingMessageId) return Effect.succeed(fork)

  return Effect.gen(function* () {
    let emptied = false
    const flushed = yield* messages.updateById(fork.messages, thinkingMessageId, (msg) => {
      if (msg.type !== 'thinking') return msg
      const held = flushHeld(msg.id)
      emptied = msg.content === '' && held === ''
      return held === '' ? msg : { ...msg, content: msg.content + held }
    })
    const index = emptied
      ? yield* messages.removeById(flushed, thinkingMessageId)
      : flushed

    return {
      ...fork,
      messages: index,
      _thinkingMessageId: null,
    }
  })
}

/** Finalize the fork's open tool messages, located by `_activeToolCallIds`. */
function finalizeActiveToolMessages(
  messages: DisplayTimelineMessagesHandle,
  fork: DisplayTimelineState,
  getToolHandle: (toolCallId: string) => ToolHandleFromSchema | undefined = () => undefined,
) {
  return Effect.gen(function* () {
    let index = fork.messages
    for (const toolCallId of fork._activeToolCallIds) {
      index = yield* messages.updateById(index, toolCallId, (msg) => {
        if (msg.type !== 'tool') return msg
        if (Option.isNone(msg.presentation)) return msg
        if (isTerminalToolPhase(msg.presentation.value.phase)) return msg

        const handle = getToolHandle(toolCallId)
        const presentation = handle
          ? interruptPresentation(Option.some(presentToolState(handle)))
          : interruptPresentation(msg.presentation)

        return Option.isSome(presentation)
          ? { ...msg, presentation }
          : msg
      })
    }
    return { ...fork, messages: index }
  })
}

function rememberForkActivityMessage(
  fork: DisplayTimelineState,
  childForkId: string,
  messageId: string
): DisplayTimelineState {
  const existing = fork._forkActivityMessageIdsByForkId[childForkId] ?? []
  return {
    ...fork,
    _forkActivityMessageIdsByForkId: {
      ...fork._forkActivityMessageIdsByForkId,
      [childForkId]: [...existing, messageId]
    }
  }
}

function forgetForkActivityMessages(
  fork: DisplayTimelineState,
  childForkId: string
): DisplayTimelineState {
  const { [childForkId]: _removed, ...remaining } = fork._forkActivityMessageIdsByForkId
  return { ...fork, _forkActivityMessageIdsByForkId: remaining }
}

function latestForkActivityMessageId(
  fork: DisplayTimelineState,
  childForkId: string
): string | undefined {
  const ids = fork._forkActivityMessageIdsByForkId[childForkId]
  return ids?.[ids.length - 1]
}

function stopActiveFork(
  fork: DisplayTimelineState,
  timestamp: number
): DisplayTimelineState {
  const cleared = {
    ...fork,
    _currentTurnId: null,
    mode: 'idle' as const,
    streamingMessageId: null,
    _thinkingMessageId: null,
    _activeToolCallIds: [],
    _communicationMessageIdsByStreamId: {},
  }
  void timestamp
  return cleared
}

function startCommunicationStreamMessage(
  messages: DisplayTimelineMessagesHandle,
  fork: DisplayTimelineState,
  message: Omit<AgentCommunicationMessage, 'id' | 'type' | 'preview' | 'streamId' | 'status' | 'content'>,
  streamId: string,
  textDelta: string
) {
  if (message.forkId === null) return Effect.succeed(fork)

  return Effect.gen(function* () {
    const messageId = streamId
    const content = textDelta
    const nextFork = yield* insertMessageIntoFork(messages, fork, {
      id: messageId,
      type: 'agent_communication',
      streamId: Option.some(streamId),
      ...message,
      content,
      preview: toPreview(content),
      status: Option.some('streaming' as const),
    })
    return rememberCommunicationMessage(nextFork, streamId, messageId)
  })
}

function upsertCommunicationStreamMessage(
  messages: DisplayTimelineMessagesHandle,
  fork: DisplayTimelineState,
  message: Omit<AgentCommunicationMessage, 'id' | 'type' | 'preview' | 'streamId' | 'status' | 'content'>,
  streamId: string,
  textDelta: string
) {
  if (message.forkId === null) return Effect.succeed(fork)

  return Effect.gen(function* () {
    const existingMessageId = fork._communicationMessageIdsByStreamId[streamId]

    if (existingMessageId !== undefined) {
      const index = yield* messages.updateById(
        fork.messages,
        existingMessageId,
        (current) => {
          if (current.type !== 'agent_communication') return current
          const content = current.content + textDelta
          return {
            ...current,
            content,
            preview: toPreview(content),
            status: Option.some('streaming' as const),
          }
        }
      )
      return { ...fork, messages: index }
    }

    const messageId = streamId
    const content = textDelta
    const nextFork = yield* insertMessageIntoFork(messages, fork, {
      id: messageId,
      type: 'agent_communication',
      streamId: Option.some(streamId),
      ...message,
      content,
      preview: toPreview(content),
      status: Option.some('streaming' as const),
    })
    return rememberCommunicationMessage(nextFork, streamId, messageId)
  })
}

function completeCommunicationStreamMessage(
  messages: DisplayTimelineMessagesHandle,
  fork: DisplayTimelineState,
  streamId: string
) {
  return Effect.gen(function* () {
    const existingMessageId = fork._communicationMessageIdsByStreamId[streamId]
    if (existingMessageId === undefined) return fork

    const index = yield* messages.updateById(
      fork.messages,
      existingMessageId,
      (current) => {
        if (current.type !== 'agent_communication') return current
        return {
          ...current,
          preview: toPreview(current.content),
          status: Option.some('completed' as const),
        }
      }
    )

    return forgetCommunicationMessage({ ...fork, messages: index }, streamId)
  })
}

function setForkInState(
  state: { readonly forks: ReadonlyMap<string | null, DisplayTimelineState> },
  forkId: string | null,
  fork: DisplayTimelineState
) {
  return {
    ...state,
    forks: new Map(state.forks).set(forkId, fork)
  }
}

// ── Tool cluster mapping ────────────────────────────────────────

function getToolCluster(toolKey: ToolKeyErased): Option.Option<string> {
  switch (toolKey) {
    case 'fileRead': return Option.some('read')
    case 'fileSearch': return Option.some('search')
    case 'webSearch': return Option.some('web_search')
    case 'webFetch': return Option.some('web_fetch')
    case 'fileTree': return Option.some('tree')
    case 'fileView': return Option.some('view')
    default: return Option.none()
  }
}

// ── Signal definitions ──────────────────────────────────────────

const forkToolStepSignalDef = Signal.create<{ forkId: string | null; toolKey: ToolKeyErased }>('DisplayTimeline/forkToolStep')
const forkToolStepSignal = Signal.fromDef<{ forkId: string | null; toolKey: ToolKeyErased }, unknown>(forkToolStepSignalDef, 'DisplayTimeline')

// ── Tool presentation ───────────────────────────────────────────
// Build the typed presentation from the handle at event time and store it
// durably on the ToolMessage. Handles are transient; message presentation is
// the display timeline's replayable source of truth.

const TERMINAL_TOOL_PHASES: ReadonlySet<ToolStepPresentation['phase']> = new Set([
  'completed',
  'error',
  'rejected',
  'interrupted',
])

function isTerminalToolPhase(phase: ToolStepPresentation['phase']): boolean {
  return TERMINAL_TOOL_PHASES.has(phase)
}

function presentationFromHandle(
  handle: ToolHandleFromSchema | undefined,
): ToolMessage['presentation'] {
  return handle ? Option.some(presentToolState(handle)) : Option.none()
}

function presentationFromHandleOrCurrent(
  handle: ToolHandleFromSchema | undefined,
  current: ToolMessage['presentation'],
): ToolMessage['presentation'] {
  return handle ? Option.some(presentToolState(handle)) : current
}

function interruptPresentation(presentation: ToolMessage['presentation']): ToolMessage['presentation'] {
  if (Option.isNone(presentation)) return presentation
  return Option.some({ ...presentation.value, phase: 'interrupted' as const, running: false, failed: true })
}

// ── Projection ──────────────────────────────────────────────────

export const DisplayTimelineProjection = Projection.defineForked<AppEvent>()({
  name: 'DisplayTimeline',
  forkState: DisplayTimelineStateSchema,
  addressed: {
    messages: DisplayTimelineMessages
  },

  ambients: [],

  reads: [AgentRoutingProjection, AgentLifecycleProjection, GoalProjection, TurnProjection, HarnessStateProjection] as const,

  initialFork: {
    mode: 'idle',
    messages: DisplayTimelineMessages.empty,
    _queuedUserMessageCount: 0,
    _pendingInboundCommunications: [],
    _currentTurnId: null,
    streamingMessageId: null,
    _thinkingMessageId: null,
    _activeToolCallIds: [],
    _communicationMessageIdsByStreamId: {},
    _forkActivityMessageIdsByForkId: {},
  },

  signals: {
    restoreQueuedMessages: Signal.create<{
      forkId: string | null
      messages: { id: string; content: string; taskMode: boolean }[]
    }>('DisplayTimeline/restoreQueuedMessages'),
    forkToolStep: forkToolStepSignalDef,
  },

  eventHandlers: {
    user_message: ({ event, fork, addressed }) => {
      const content = event.text
      const messageType: 'user_message' | 'queued_user_message' =
        fork._currentTurnId !== null ? 'queued_user_message' : 'user_message'

      return appendMessageToFork(addressed.messages, fork, {
        id: event.messageId,
        type: messageType,
        content,
        timestamp: event.timestamp,
        taskMode: event.taskMode,
        attachments: event.attachments
          .filter((attachment): attachment is Extract<typeof attachment, { type: 'image' }> => attachment.type === 'image')
          .map(attachment => ({
            type: attachment.type,
            path: attachment.image.path,
            filename: Option.getOrElse(attachment.image.name, () => attachment.image.path.split('/').pop() ?? 'image'),
            mediaType: attachment.image.mediaType,
            width: attachment.image.dimensions.width,
            height: attachment.image.dimensions.height,
          })),
      })
    },

    skill_activated: ({ event, fork, addressed }) => {
      if (event.source !== 'user') return fork

      const content = event.message ? `/${event.skillName} ${event.message}` : `/${event.skillName}`
      return appendMessageToFork(addressed.messages, fork, {
        id: `skill_activated:${forkIdKey(event.forkId)}:${event.timestamp}:${event.skillName}`,
        type: 'user_message' as const,
        content,
        timestamp: event.timestamp,
        taskMode: false,
        attachments: [],
      })
    },

    goal_started: ({ event, fork, addressed }) => {
      const msg: GoalStatusMessage = {
        id: `goal:${event.goalId}:started`,
        type: 'goal_status',
        status: 'started',
        objective: Option.some(event.objective),
        evidence: Option.none(),
        timestamp: event.timestamp,
      }

      return insertMessageIntoFork(addressed.messages, fork, msg)
    },

    goal_finished: ({ event, fork, read, addressed }) => {
      const goalState = read(GoalProjection)
      const finished = goalState.finished.find((goal) => goal.goalId === event.goalId)
      const msg: GoalStatusMessage = {
        id: `goal:${event.goalId}:finished`,
        type: 'goal_status',
        status: 'finished',
        ...(finished?.objective ? { objective: Option.some(finished.objective) } : { objective: Option.none() }),
        evidence: Option.some(event.evidence),
        timestamp: event.timestamp,
      }

      return insertMessageIntoFork(addressed.messages, fork, msg)
    },

    turn_started: ({ event, fork, addressed }) => Effect.gen(function* () {
      let stateWithMessages = yield* releaseActiveThinking(addressed.messages, fork)

      if (stateWithMessages._queuedUserMessageCount > 0) {
        // Queued messages are by invariant the tail suffix; promote each in
        // place (content-only, same ids and offsets).
        let index = stateWithMessages.messages
        const queuedIds = tailDisplayMessageIds(
          addressed.messages,
          index,
          stateWithMessages._queuedUserMessageCount
        )
        for (const queuedId of queuedIds) {
          index = yield* addressed.messages.updateById(index, queuedId, (msg) =>
            msg.type === 'queued_user_message'
              ? { ...msg, type: 'user_message' as const }
              : msg
          )
        }
        stateWithMessages = { ...stateWithMessages, messages: index, _queuedUserMessageCount: 0 }
      }

      return {
        ...stateWithMessages,
        _currentTurnId: event.turnId,
        mode: 'streaming' as const,
        _pendingInboundCommunications: [],
      }
    }),

    message_start: ({ event, fork, addressed }) => Effect.gen(function* () {
      if (fork._currentTurnId !== event.turnId) return fork
      if (event.destination.kind !== 'user') return fork

      const msgId = event.id
      const assistantMessage: AssistantMessageDisplay = {
        id: msgId,
        type: 'assistant_message',
        content: '',
        timestamp: event.timestamp
      }

      const forkWithFlushedThinking = yield* releaseActiveThinking(addressed.messages, fork)
      const withMessage = yield* insertMessageIntoFork(addressed.messages, forkWithFlushedThinking, assistantMessage)

      const nextFork = {
        ...withMessage,
        streamingMessageId: msgId,
        _thinkingMessageId: null
      }
      return nextFork
    }),

    message_chunk: ({ event, fork, addressed }) => {
      if (fork._currentTurnId !== event.turnId) return fork
      if (!fork.streamingMessageId) return fork
      return Effect.map(
        addressed.messages.updateById(
          fork.messages,
          fork.streamingMessageId,
          (msg) => msg.type === 'assistant_message'
            ? { ...msg, content: msg.content + event.text }
            : msg
        ),
        (messages): DisplayTimelineState => ({ ...fork, messages })
      )
    },

    message_end: ({ event, fork, addressed }) => Effect.gen(function* () {
      if (fork._currentTurnId !== event.turnId) return fork

      if (fork.streamingMessageId) {
        const message = yield* readDisplayMessageById(addressed.messages, fork.messages, fork.streamingMessageId)
        let nextFork = fork
        if (
          Option.isSome(message) &&
          message.value.type === 'assistant_message' &&
          !message.value.content.trim()
        ) {
          nextFork = {
            ...fork,
            messages: yield* addressed.messages.removeById(fork.messages, fork.streamingMessageId)
          }
        }
        return { ...nextFork, streamingMessageId: null }
      }

      return { ...fork, streamingMessageId: null }
    }),

    thinking_chunk: ({ event, fork, addressed }) => Effect.gen(function* () {
      if (fork._currentTurnId !== event.turnId) return fork

      if (fork._thinkingMessageId) {
        const thinkingMessageId = fork._thinkingMessageId
        let shouldSuppressThinking = false
        let nextContentLength: number | null = null

        const messages = yield* addressed.messages.updateById(
          fork.messages,
          thinkingMessageId,
          (msg) => {
            if (msg.type !== 'thinking') return msg
            const { contentToAppend, shouldSuppress } = processThinkingChunk(msg, event.text)

            if (shouldSuppress) {
              shouldSuppressThinking = true
              return msg
            }

            if (contentToAppend === '') return msg

            const updated = { ...msg, content: msg.content + contentToAppend }
            nextContentLength = updated.content.length
            return updated
          }
        )

        if (shouldSuppressThinking) {
          heldBuffers.delete(thinkingMessageId)
          const nextForkWithoutThinking = {
            ...fork,
            messages: yield* addressed.messages.removeById(messages, thinkingMessageId),
            _thinkingMessageId: null,
          }
          return nextForkWithoutThinking
        }

        if (nextContentLength === null) return fork

        void nextContentLength
        return { ...fork, messages }
      }

      const stepId = `thinking:${event.turnId}`
      const tempStep: ThinkingMessage = {
        id: stepId,
        type: 'thinking',
        content: '',
        label: Option.none(),
        timestamp: event.timestamp,
      }
      const { contentToAppend, shouldSuppress } = processThinkingChunk(tempStep, event.text)

      if (shouldSuppress) {
        heldBuffers.delete(stepId)
        return fork
      }

      if (contentToAppend === '' && heldBuffers.has(stepId)) {
        const thinkingMsg: ThinkingMessage = {
          id: stepId,
          type: 'thinking',
          content: '',
          label: Option.none(),
          timestamp: event.timestamp,
        }
        const nextFork = yield* insertMessageIntoFork(addressed.messages, fork, thinkingMsg)
        return { ...nextFork, _thinkingMessageId: stepId }
      }

      const thinkingMsg: ThinkingMessage = {
        id: stepId,
        type: 'thinking',
        content: contentToAppend,
        label: Option.none(),
        timestamp: event.timestamp,
      }
      const withMessage = yield* insertMessageIntoFork(addressed.messages, fork, thinkingMsg)
      const withThinking = { ...withMessage, _thinkingMessageId: stepId }
      return withThinking
    }),

    tool_event: ({ event, fork, read, emit, addressed }) => Effect.gen(function* () {
      const inner = event.event
      const harnessState = read(HarnessStateProjection)
      const handle = harnessState.handles.handles.get(event.toolCallId)
      switch (inner._tag) {
        case 'ToolInputStarted': {
          emit.forkToolStep({ forkId: event.forkId, toolKey: event.toolKey })

          if (fork._currentTurnId !== event.turnId) return fork

          // spawnWorker starting
          if (event.toolKey === 'spawnWorker') {
            const toolMsg: ToolMessage = {
              id: event.toolCallId,
              type: 'tool',
              toolKey: event.toolKey,
              cluster: getToolCluster(event.toolKey),
              presentation: presentationFromHandle(handle),
              filter: Option.none(),
              resultFilePath: Option.none(),
              timestamp: event.timestamp,
            }
            const forkWithFlushedThinking = yield* releaseActiveThinking(addressed.messages, fork)
            const withMessage = yield* insertMessageIntoFork(addressed.messages, forkWithFlushedThinking, toolMsg)
            const withMessages = addActiveToolProducer(
              {
                ...withMessage,
                _thinkingMessageId: null
              },
              event.toolCallId
            )
            return withMessages
          }

          // Skip hidden tools
          if (HIDDEN_TOOLS.has(event.toolKey)) return fork

          // Flush any thinking before adding tool
          const toolMsg: ToolMessage = {
            id: event.toolCallId,
            type: 'tool',
            toolKey: event.toolKey,
            cluster: getToolCluster(event.toolKey),
            presentation: presentationFromHandle(handle),
            filter: Option.none(),
            resultFilePath: Option.none(),
            timestamp: event.timestamp,
          }
          const forkWithFlushedThinking = yield* releaseActiveThinking(addressed.messages, fork)
          const withMessage = yield* insertMessageIntoFork(addressed.messages, forkWithFlushedThinking, toolMsg)
          const withMessages = addActiveToolProducer(
            { ...withMessage, _thinkingMessageId: null },
            event.toolCallId
          )
          return withMessages
        }

        case 'ToolInputReady': {
          if (fork._currentTurnId !== event.turnId) return fork
          if (HIDDEN_TOOLS.has(event.toolKey)) return fork

          const messages = yield* addressed.messages.updateById(
              fork.messages,
              event.toolCallId,
              (msg) => msg.type === 'tool'
                ? {
                    ...msg,
                    presentation: presentationFromHandleOrCurrent(handle, msg.presentation),
                  }
                : msg
            )
          return { ...fork, messages }
        }

        case 'ToolExecutionEnded': {
          if (fork._currentTurnId !== event.turnId) return fork

          if (HIDDEN_TOOLS.has(event.toolKey)) return fork

          const messages = yield* addressed.messages.updateById(
              fork.messages,
              event.toolCallId,
              (msg) => msg.type === 'tool'
                ? {
                    ...msg,
                    presentation: presentationFromHandleOrCurrent(handle, msg.presentation),
                  }
                : msg
            )
          return {
            ...removeActiveToolProducer(fork, event.toolCallId),
            messages
          }
        }

        case 'ToolInputRejected': {
          if (fork._currentTurnId !== event.turnId) return fork

          if (HIDDEN_TOOLS.has(event.toolKey)) return removeActiveToolProducer(fork, event.toolCallId)

          const messages = yield* addressed.messages.updateById(
            fork.messages,
            event.toolCallId,
            (msg) => msg.type === 'tool'
              ? {
                  ...msg,
                  presentation: presentationFromHandleOrCurrent(handle, msg.presentation),
                }
              : msg
          )
          return {
            ...removeActiveToolProducer(fork, event.toolCallId),
            messages
          }
        }

        default: {
          if (fork._currentTurnId !== event.turnId) return fork
          if (HIDDEN_TOOLS.has(event.toolKey)) return fork

          if (!handle) return fork

          const messages = yield* addressed.messages.updateById(
              fork.messages,
              event.toolCallId,
              (msg) => msg.type === 'tool'
                ? { ...msg, presentation: Option.some(presentToolState(handle)) }
                : msg
            )
          return {
            ...fork,
            messages
          }
        }
      }
    }),

    turn_outcome: ({ event, fork, read, addressed }) => Effect.gen(function* () {
      if (fork._currentTurnId !== event.turnId) return fork

      if (event.outcome._tag === 'Completed' && outcomeWillChainContinue(event.outcome)) {
        const nextFork = {
          ...fork,
          _thinkingMessageId: null,
          _activeToolCallIds: [],
          _communicationMessageIdsByStreamId: {}
        }
        return {
          ...nextFork,
          _currentTurnId: null,
          mode: 'idle' as const,
          streamingMessageId: null,
        }
      }

      if (event.outcome._tag === 'ConnectionFailure') {
        const harnessState = read(HarnessStateProjection)
        const statusMsg: StatusIndicatorMessage = {
          type: 'status_indicator' as const,
          id: `turn_outcome:${event.turnId}:connection_failure`,
          message: 'Connection issue: retrying',
          style: 'dim' as const,
          timestamp: event.timestamp,
        }
        const forkWithFlushedThinking = yield* releaseActiveThinking(addressed.messages, fork)
        const withFinalizedTools = yield* finalizeActiveToolMessages(
          addressed.messages,
          forkWithFlushedThinking,
          (toolCallId) => harnessState.handles.handles.get(toolCallId),
        )
        const withMessage = yield* insertMessageIntoFork(addressed.messages, withFinalizedTools, statusMsg)
        const nextFork = {
          ...withMessage,
          _thinkingMessageId: null,
          _activeToolCallIds: [],
          _communicationMessageIdsByStreamId: {}
        }
        return nextFork
      }

      if (event.outcome._tag === 'Overthinking') {
        const harnessState = read(HarnessStateProjection)
        const released = yield* releaseActiveThinking(addressed.messages, fork)
        const withFinalizedTools = yield* finalizeActiveToolMessages(
          addressed.messages,
          released,
          (toolCallId) => harnessState.handles.handles.get(toolCallId),
        )
        const statusMsg: StatusIndicatorMessage = {
          type: 'status_indicator' as const,
          id: `turn_outcome:${event.turnId}:overthinking`,
          message: `Thinking exceeded ${event.outcome.limit} character limit — continuing with feedback`,
          style: 'dim' as const,
          timestamp: event.timestamp,
        }
        const withMessages = yield* insertMessageIntoFork(addressed.messages, withFinalizedTools, statusMsg)
        const nextFork = {
          ...withMessages,
          _activeToolCallIds: [],
          _communicationMessageIdsByStreamId: {}
        }
        return {
          ...nextFork,
          _currentTurnId: null,
          mode: 'idle' as const,
          streamingMessageId: null,
        }
      }

      const harnessState = read(HarnessStateProjection)
      const cleanedState = yield* releaseActiveThinking(addressed.messages, fork)
      const withFinalizedTools = yield* finalizeActiveToolMessages(
        addressed.messages,
        cleanedState,
        (toolCallId) => harnessState.handles.handles.get(toolCallId),
      )

      const endActive: Pick<
        DisplayTimelineState,
        '_currentTurnId' | 'mode' | 'streamingMessageId' | '_thinkingMessageId' | '_activeToolCallIds' | '_communicationMessageIdsByStreamId'
      > = {
        _currentTurnId: null,
        mode: 'idle',
        streamingMessageId: null,
        _thinkingMessageId: null,
        _activeToolCallIds: [],
        _communicationMessageIdsByStreamId: {},
      }

      if (event.outcome._tag === 'Completed') {
        return {
          ...withFinalizedTools,
          ...endActive,
        }
      }

      if (event.outcome._tag === 'Cancelled') {
        // The interrupt handler appends its marker at the tail, so a duplicate
        // for this cancellation can only be the last message.
        const lastId = tailDisplayMessageIds(addressed.messages, withFinalizedTools.messages, 1)[0]
        const lastMessage = lastId === undefined
          ? Option.none<DisplayMessage>()
          : yield* readDisplayMessageById(addressed.messages, withFinalizedTools.messages, lastId)
        const alreadyInterrupted = Option.isSome(lastMessage) &&
          lastMessage.value.type === 'interrupted' &&
          lastMessage.value.timestamp === event.timestamp
        const withInterrupted = alreadyInterrupted
          ? withFinalizedTools
          : yield* appendMessageToFork(addressed.messages, withFinalizedTools, {
              id: `turn_outcome:${event.turnId}:cancelled`,
              type: 'interrupted' as const,
              timestamp: event.timestamp,
              allKilled: Option.none(),
              context: event.forkId === null ? 'root' as const : 'fork' as const,
            })
        return {
          ...withInterrupted,
          ...endActive,
        }
      }

      if (
        event.outcome._tag === 'ContextWindowExceeded'
      ) {
        return {
          ...withFinalizedTools,
          ...endActive,
        }
      }

      const errorMessage = toErrorDisplayMessage(
        `turn_outcome:${event.turnId}:error:${event.outcome._tag}`,
        event.outcome,
        event.timestamp,
      )
      const withError = errorMessage
        ? yield* appendMessageToFork(addressed.messages, withFinalizedTools, errorMessage)
        : withFinalizedTools

      return {
        ...withError,
        ...endActive,
      }
    }),

    compaction_failed: ({ event, fork, addressed }) => {
      if (!event.presentation || event.presentation.surface !== 'inline') return fork

      const errorMsg: ErrorDisplayMessage = {
        id: `compaction_failed:${forkIdKey(event.forkId)}:${event.timestamp}`,
        type: 'error',
        message: event.presentation.message,
        timestamp: event.timestamp,
        cta: event.presentation.cta ? Option.some(event.presentation.cta) : Option.none(),
      }

      return appendMessageToFork(addressed.messages, fork, errorMsg)
    },

    interrupt: ({ event, fork, emit, read, addressed }) => Effect.gen(function* () {
      // Queued messages are by invariant the tail suffix.
      if (fork._queuedUserMessageCount > 0) {
        const queuedMessages = yield* addressed.messages.readWindow(
          addressed.messages.resolveTailWindow(fork.messages, fork._queuedUserMessageCount)
        )
        emit.restoreQueuedMessages({
          forkId: event.forkId,
          messages: queuedMessages.flatMap((m) =>
            m.type === 'queued_user_message'
              ? [{ id: m.id, content: m.content, taskMode: m.taskMode }]
              : []
          )
        })
      }

      const harnessState = read(HarnessStateProjection)
      const stateWithInterruptedTools = yield* finalizeActiveToolMessages(
        addressed.messages,
        fork,
        (toolCallId) => harnessState.handles.handles.get(toolCallId),
      )

      const cleanedState = yield* releaseActiveThinking(addressed.messages, stateWithInterruptedTools)

      // Replace the queued tail suffix with the interrupted marker.
      const total = cleanedState.messages.totalCount
      const messages = yield* addressed.messages.replaceRange(
        cleanedState.messages,
        total - cleanedState._queuedUserMessageCount,
        total,
        [{
          id: `interrupt:${forkIdKey(event.forkId)}:${fork._currentTurnId ?? 'idle'}:${event.timestamp}`,
          type: 'interrupted' as const,
          timestamp: event.timestamp,
          context: event.forkId === null ? 'root' as const : 'fork' as const,
          allKilled: Option.some(false),
        }]
      )
      const withMessages = {
        ...cleanedState,
        messages,
        _queuedUserMessageCount: 0,
        _activeToolCallIds: [],
        _communicationMessageIdsByStreamId: {}
      }

      return {
        ...withMessages,
        _currentTurnId: null,
        mode: 'idle' as const,
        streamingMessageId: null,
      }
    }),

    agent_created: ({ event, fork, addressed }) => {
      if (event.message === null) return fork
      const content = event.message.trim()
      if (!content) return fork
      if (event.forkId === null) return fork

      return Effect.gen(function* () {
        const forkWithFlush = yield* releaseActiveThinking(addressed.messages, fork)
        const commMsg: AgentCommunicationMessage = {
          id: `agent_created:${event.agentId}:message`,
          type: 'agent_communication',
          direction: 'from_agent',
          agentId: event.agentId,
          streamId: Option.none(),
          agentName: Option.some(event.name),
          agentRole: Option.some(event.role),
          forkId: event.forkId,
          content,
          preview: toPreview(content),
          timestamp: event.timestamp,
          status: Option.some('completed' as const),
        }
        return yield* insertMessageIntoFork(addressed.messages, forkWithFlush, commMsg)
      })
    },

    agent_killed: ({ event, fork }) => stopActiveFork(fork, event.timestamp),
    worker_user_killed: ({ event, fork }) => stopActiveFork(fork, event.timestamp),
    worker_idle_closed: ({ event, fork }) => stopActiveFork(fork, event.timestamp),

  },

  signalHandlers: (on) => [
    on(AgentLifecycleProjection.signals.agentCreated, ({ value, state, addressed }) => Effect.gen(function* () {
      const { forkId, parentForkId, name, role } = value
      const parentState = state.forks.get(parentForkId)
      if (!parentState) return state
      const messages = addressed.forFork(parentForkId).messages

      const activityMessageId = `fork_activity:${forkId}:0`
      const msg: ForkActivityMessage = {
        id: activityMessageId,
        type: 'fork_activity',
        forkId,
        name,
        role,
        status: 'running',
        createdAt: value.timestamp,
        activeSince: value.timestamp,
        accumulatedActiveMs: 0,
        resumeCount: Option.some(0),
        completedAt: Option.none(),
        toolCounts: EMPTY_TOOL_COUNTS,
        timestamp: value.timestamp
      }

      const nextParentFork = rememberForkActivityMessage(
        yield* insertMessageIntoFork(messages, parentState, msg),
        forkId,
        activityMessageId
      )
      const forks = new Map(state.forks).set(parentForkId, nextParentFork)
      return { ...state, forks }
    })),

    on(forkToolStepSignal, ({ value, state, read, addressed }) => Effect.gen(function* () {
      const { forkId, toolKey } = value
      if (forkId === null) return state

      const agentState = read(AgentLifecycleProjection)
      const agent = getAgentByForkId(agentState, forkId)
      if (!agent) return state

      const parentState = state.forks.get(agent.parentForkId)
      if (!parentState) return state
      const messagesHandle = addressed.forFork(agent.parentForkId).messages

      const activityMessageId = latestForkActivityMessageId(parentState, forkId)
      if (activityMessageId === undefined) return state

      const index = yield* messagesHandle.updateById(parentState.messages, activityMessageId, (msg) =>
        msg.type === 'fork_activity' && msg.status === 'running'
          ? { ...msg, toolCounts: isToolKey(toolKey) ? incrementToolCount(msg.toolCounts, toolKey) : msg.toolCounts }
          : msg
      )

      return setForkInState(state, agent.parentForkId, { ...parentState, messages: index })
    })),

    on(AgentLifecycleProjection.signals.agentBecameIdle, ({ value, state, read, addressed }) => Effect.gen(function* () {
      const { forkId, parentForkId } = value

      const parentState = state.forks.get(parentForkId)
      if (!parentState) return state
      const messagesHandle = addressed.forFork(parentForkId).messages

      const activityMessageId = latestForkActivityMessageId(parentState, forkId)
      if (activityMessageId === undefined) return state

      const activityMessage = yield* readDisplayMessageById(messagesHandle, parentState.messages, activityMessageId)
      if (
        Option.isNone(activityMessage) ||
        activityMessage.value.type !== 'fork_activity' ||
        activityMessage.value.status !== 'running'
      ) return state
      const msg = activityMessage.value

      const stintMs = Math.max(0, value.timestamp - msg.activeSince)
      const cumulativeTotalTimeMs = msg.accumulatedActiveMs + stintMs
      const index = yield* messagesHandle.updateById(parentState.messages, activityMessageId, (current) =>
        current.type === 'fork_activity'
          ? {
              ...current,
              status: 'completed' as const,
              completedAt: Option.some(value.timestamp),
              accumulatedActiveMs: cumulativeTotalTimeMs,
            }
          : current
      )

      let nextParentState: DisplayTimelineState = { ...parentState, messages: index }

      if (parentForkId === null && value.reason !== 'interrupt') {
        const finishedMsg: WorkerFinishedMessage = {
          id: `worker_finished:${value.agentId}:${value.timestamp}`,
          type: 'worker_finished',
          workerRole: value.role,
          workerId: value.agentId,
          cumulativeTotalTimeMs,
          cumulativeTotalToolsUsed: totalToolsUsed(msg.toolCounts),
          resumed: Option.getOrElse(msg.resumeCount, () => 0) > 0,
          timestamp: value.timestamp,
        }
        nextParentState = yield* insertMessageIntoFork(messagesHandle, nextParentState, finishedMsg)
      }

      const forks = new Map(state.forks).set(parentForkId, nextParentState)
      return { ...state, forks }
    })),

    on(AgentLifecycleProjection.signals.agentBecameWorking, ({ value, state, addressed }) => Effect.gen(function* () {
      const { forkId, parentForkId } = value
      const parentState = state.forks.get(parentForkId)
      if (!parentState) return state
      const messagesHandle = addressed.forFork(parentForkId).messages

      const activityMessageId = latestForkActivityMessageId(parentState, forkId)
      if (activityMessageId === undefined) return state

      const activityMessage = yield* readDisplayMessageById(messagesHandle, parentState.messages, activityMessageId)
      if (Option.isNone(activityMessage) || activityMessage.value.type !== 'fork_activity') return state
      const message = activityMessage.value

      if (Option.isNone(message.completedAt)) {
        const rc = Option.getOrElse(message.resumeCount, () => 0)
        if (rc === 0) {
          const index = yield* messagesHandle.updateById(parentState.messages, activityMessageId, (current) =>
            current.type === 'fork_activity'
              ? { ...current, activeSince: value.timestamp, timestamp: value.timestamp }
              : current
          )
          const nextParentState = { ...parentState, messages: index }

          const forks = new Map(state.forks).set(parentForkId, nextParentState)
          return { ...state, forks }
        }
        return state
      }

      const nextResumeCount = Option.getOrElse(message.resumeCount, () => 0) + 1
      const resumedBlockId = `fork_activity:${forkId}:${nextResumeCount}`
      const resumedBlock: ForkActivityMessage = {
        id: resumedBlockId,
        type: 'fork_activity',
        forkId,
        name: message.name,
        role: message.role,
        status: 'running',
        createdAt: value.timestamp,
        activeSince: value.timestamp,
        accumulatedActiveMs: message.accumulatedActiveMs,
        resumeCount: Option.some(nextResumeCount),
        completedAt: Option.none(),
        toolCounts: message.toolCounts,
        timestamp: value.timestamp
      }

      let nextParentState = rememberForkActivityMessage(
        yield* insertMessageIntoFork(messagesHandle, parentState, resumedBlock),
        forkId,
        resumedBlockId
      )

      if (parentForkId === null) {
        const step: WorkerResumedMessage = {
          id: `worker_resumed:${value.agentId}:${nextResumeCount}:${value.timestamp}`,
          type: 'worker_resumed',
          workerRole: value.role,
          workerId: value.agentId,
          title: message.name,
          timestamp: value.timestamp,
        }
        nextParentState = yield* insertMessageIntoFork(messagesHandle, nextParentState, step)
      }

      const forks = new Map(state.forks).set(parentForkId, nextParentState)
      return { ...state, forks }
    })),

    on(AgentLifecycleProjection.signals.agentKilled, ({ value, state, addressed }) => Effect.gen(function* () {
      const parentState = state.forks.get(value.parentForkId)
      if (!parentState) return state
      const messagesHandle = addressed.forFork(value.parentForkId).messages

      let index = parentState.messages
      for (const activityMessageId of parentState._forkActivityMessageIdsByForkId[value.forkId] ?? []) {
        index = yield* messagesHandle.removeById(index, activityMessageId)
      }
      const withoutActivity = forgetForkActivityMessages(
        { ...parentState, messages: index },
        value.forkId
      )

      const step: WorkerKilledMessage = {
        id: `worker_killed:${value.agentId}:${value.timestamp}`,
        type: 'worker_killed',
        workerRole: value.role,
        workerId: value.agentId,
        title: value.title,
        timestamp: value.timestamp,
      }

      const nextParentState = yield* insertMessageIntoFork(messagesHandle, withoutActivity, step)
      const forks = new Map(state.forks).set(value.parentForkId, nextParentState)
      return { ...state, forks }
    })),

    on(AgentLifecycleProjection.signals.subagentUserKilled, ({ value, state, addressed }) => Effect.gen(function* () {
      const parentState = state.forks.get(value.parentForkId)
      if (!parentState) return state
      const messagesHandle = addressed.forFork(value.parentForkId).messages

      let index = parentState.messages
      for (const activityMessageId of parentState._forkActivityMessageIdsByForkId[value.forkId] ?? []) {
        index = yield* messagesHandle.removeById(index, activityMessageId)
      }
      const withoutActivity = forgetForkActivityMessages(
        { ...parentState, messages: index },
        value.forkId
      )

      const step: WorkerUserKilledMessage = {
        id: `worker_user_killed:${value.agentId}:${value.timestamp}`,
        type: 'worker_user_killed',
        workerRole: value.role,
        workerId: value.agentId,
        title: value.title,
        timestamp: value.timestamp,
      }

      const nextParentState = yield* insertMessageIntoFork(messagesHandle, withoutActivity, step)
      const forks = new Map(state.forks).set(value.parentForkId, nextParentState)
      return { ...state, forks }
    })),

    on(AgentLifecycleProjection.signals.workerIdleClosed, ({ value, state, addressed }) => Effect.gen(function* () {
      const parentState = state.forks.get(value.parentForkId)
      if (!parentState) return state
      const messagesHandle = addressed.forFork(value.parentForkId).messages

      let index = parentState.messages
      for (const activityMessageId of parentState._forkActivityMessageIdsByForkId[value.forkId] ?? []) {
        index = yield* messagesHandle.removeById(index, activityMessageId)
      }
      const nextParentState = forgetForkActivityMessages(
        { ...parentState, messages: index },
        value.forkId
      )
      const forks = new Map(state.forks).set(value.parentForkId, nextParentState)
      return { ...state, forks }
    })),

    on(AgentRoutingProjection.signals.communicationStreamStarted, ({ value, state, read, addressed }) => Effect.gen(function* () {
      const displayFork = state.forks.get(value.targetForkId)
      if (!displayFork) return state

      if (value.direction === 'to_agent') return state
      const messagesHandle = addressed.forFork(value.targetForkId).messages

      const agentState = read(AgentLifecycleProjection)
      const targetAgent = getAgentByForkId(agentState, value.targetForkId)

      const nextFork = yield* startCommunicationStreamMessage(
        messagesHandle,
        displayFork,
        {
          direction: value.direction,
          agentId: value.agentId,
          agentName: targetAgent ? Option.some(targetAgent.name) : Option.none(),
          agentRole: targetAgent ? Option.some(targetAgent.role) : Option.none(),
          forkId: value.targetForkId,
          timestamp: value.timestamp,
        },
        value.streamId,
        value.textDelta
      )

      // Status bar activity handled by messageWorker tool lifecycle, not communication stream
      return setForkInState(state, value.targetForkId, nextFork)
    })),

    on(AgentRoutingProjection.signals.communicationStreamChunk, ({ value, state, read, addressed }) => Effect.gen(function* () {
      const displayFork = state.forks.get(value.targetForkId)
      if (!displayFork) return state

      if (value.direction === 'to_agent') return state
      const messagesHandle = addressed.forFork(value.targetForkId).messages

      const agentState = read(AgentLifecycleProjection)
      const targetAgent = getAgentByForkId(agentState, value.targetForkId)

      const nextFork = yield* upsertCommunicationStreamMessage(
        messagesHandle,
        displayFork,
        {
          direction: value.direction,
          agentId: value.agentId,
          agentName: targetAgent ? Option.some(targetAgent.name) : Option.none(),
          agentRole: targetAgent ? Option.some(targetAgent.role) : Option.none(),
          forkId: value.targetForkId,
          timestamp: value.timestamp,
        },
        value.streamId,
        value.textDelta
      )

      return setForkInState(state, value.targetForkId, nextFork)
    })),

    on(AgentRoutingProjection.signals.communicationStreamCompleted, ({ value, state, addressed }) => Effect.gen(function* () {
      const displayFork = state.forks.get(value.targetForkId)
      if (!displayFork) return state
      if (value.direction === 'to_agent') return state
      const messagesHandle = addressed.forFork(value.targetForkId).messages

      const nextFork = yield* completeCommunicationStreamMessage(messagesHandle, displayFork, value.streamId)

      // Status bar activity handled by messageWorker tool lifecycle, not communication stream
      return setForkInState(state, value.targetForkId, nextFork)
    })),

    on(AgentRoutingProjection.signals.agentMessage, ({ value, state, read }) => {
      const displayFork = state.forks.get(value.targetForkId)
      if (!displayFork) return state
      const agentState = read(AgentLifecycleProjection)
      const turnState = read(TurnProjection)
      const turnFork = turnState.forks.get(value.targetForkId)
      const targetAgent = getAgentByForkId(agentState, value.targetForkId)

      const pending = (turnFork?.pendingInboundCommunications ?? []).map((message): PendingInboundCommunicationDisplay => ({
        ...message,
        agentName: Option.isSome(message.agentName) ? message.agentName : targetAgent ? Option.some(targetAgent.name) : Option.none(),
        agentRole: Option.isSome(message.agentRole) ? message.agentRole : targetAgent ? Option.some(targetAgent.role) : Option.none(),
      }))

      return {
        ...state,
        forks: new Map(state.forks).set(value.targetForkId, {
          ...displayFork,
          _pendingInboundCommunications: pending,
        })
      }
    }),

    on(AgentRoutingProjection.signals.agentResponse, ({ value, state, read }) => {
      const displayFork = state.forks.get(value.targetForkId)
      if (!displayFork) return state
      const agentState = read(AgentLifecycleProjection)
      const turnState = read(TurnProjection)
      const turnFork = turnState.forks.get(value.targetForkId)
      const targetAgent = value.targetForkId ? getAgentByForkId(agentState, value.targetForkId) : undefined

      const pending = (turnFork?.pendingInboundCommunications ?? []).map((message): PendingInboundCommunicationDisplay => ({
        ...message,
        agentName: Option.isSome(message.agentName) ? message.agentName : targetAgent ? Option.some(targetAgent.name) : Option.none(),
        agentRole: Option.isSome(message.agentRole) ? message.agentRole : targetAgent ? Option.some(targetAgent.role) : Option.none(),
      }))

      return {
        ...state,
        forks: new Map(state.forks).set(value.targetForkId, {
          ...displayFork,
          _pendingInboundCommunications: pending,
        })
      }
    }),

    on(TurnProjection.signals.pendingInboundCommunicationsRead, ({ value, state, read, addressed }) => Effect.gen(function* () {
      const displayFork = state.forks.get(value.forkId)
      if (!displayFork) return state

      const agentState = read(AgentLifecycleProjection)
      let nextFork = { ...displayFork }
      const messagesHandle = addressed.forFork(value.forkId).messages

      if (value.forkId !== null) {
        for (const pending of value.messages.filter((message) => message.source === 'agent')) {
          const targetAgent = getAgentByForkId(agentState, value.forkId)
          const commMsg: AgentCommunicationMessage = {
            id: pending.id,
            type: 'agent_communication',
            direction: 'from_agent',
            agentId: pending.agentId,
            streamId: Option.none(),
            agentName: Option.isSome(pending.agentName) ? pending.agentName : targetAgent ? Option.some(targetAgent.name) : Option.none(),
            agentRole: Option.isSome(pending.agentRole) ? pending.agentRole : targetAgent ? Option.some(targetAgent.role) : Option.none(),
            forkId: pending.forkId,
            content: pending.content,
            preview: pending.preview,
            timestamp: pending.timestamp,
            status: Option.some('completed' as const),
          }
          nextFork = yield* insertMessageIntoFork(messagesHandle, nextFork, commMsg)
        }
      }

      const pendingIds = new Set(value.messages.map(m => m.id))
      nextFork = {
        ...nextFork,
        _pendingInboundCommunications: nextFork._pendingInboundCommunications.filter(m => !pendingIds.has(m.id))
      }

      return {
        ...state,
        forks: new Map(state.forks).set(value.forkId, nextFork)
      }
    })),

  ]
})
