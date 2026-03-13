/**
 * DisplayProjection (Forked)
 *
 * UI state with messages and ThinkBlocks, per-fork.
 * Each fork has independent display state for its conversation.
 *
 * Key invariants:
 * - Queued messages always appear at the END of the message list
 * - New content (assistant messages, think blocks) is inserted BEFORE queued messages
 * - Queued messages are promoted to user_message on turn_started
 */

import { Signal, Projection } from '@magnitudedev/event-core'
import type { AppEvent, ToolResult, ToolDisplay } from '../events'
import type { XmlToolResult } from '@magnitudedev/xml-act'
import { getVisualRegistry } from '../visuals/registry'
import { AgentProjection, type AgentState, getAgentByForkId } from './agent'

import { WorkingStateProjection } from './working-state'
import { AgentStatusBridgeProjection } from './agent-status-bridge'

import { getAgentDefinition, type AgentVariant } from '../agents'
import { textOf } from '../content'
import { createId } from '../util/id'

/**
 * Resolve display visibility for a tool call using the agent definition's display policy.
 * Reads the agent role from AgentProjection to determine which agent definition to consult.
 */
/** Map XmlToolResult → ToolResult for display. */
function mapXmlToolResultForDisplay(result: XmlToolResult, display?: ToolDisplay): ToolResult {
  switch (result._tag) {
    case 'Success':
      return { status: 'success', output: result.output, ...(display ? { display } : {}) }
    case 'Error':
      return { status: 'error', message: result.error }
    case 'Rejected': {
      const rej = result.rejection
      const isPerm = rej && typeof rej === 'object' && '_tag' in rej
      if (isPerm) {
        const r = rej as { _tag: string; reason: string }
        if (r._tag === 'UserRejection') {
          return { status: 'rejected', message: 'User rejected the action' }
        }
        return { status: 'rejected', message: 'System rejected', reason: r.reason }
      }
      return { status: 'rejected', message: String(rej) }
    }
    case 'Interrupted':
      return { status: 'interrupted' }
  }
}

function isToolHidden(toolKey: string, forkId: string | null, input: unknown, agentState: AgentState): boolean {
  const variant: AgentVariant = forkId
    ? (getAgentByForkId(agentState, forkId)?.role ?? 'builder') as AgentVariant
    : 'orchestrator'
  const agentDef = getAgentDefinition(variant)
  const result = agentDef.getDisplay(toolKey, input, undefined)
  return result.action === 'hidden'
}

// =============================================================================
// Types
// =============================================================================

export interface UserMessageDisplay {
  readonly id: string
  readonly type: 'user_message'
  readonly content: string
  readonly timestamp: number
  readonly taskMode: boolean
  readonly attachments: readonly { readonly type: 'image'; readonly width: number; readonly height: number; readonly filename: string }[]
}

export interface QueuedUserMessageDisplay {
  readonly id: string
  readonly type: 'queued_user_message'
  readonly content: string
  readonly timestamp: number
  readonly taskMode: boolean
  readonly attachments: readonly { readonly type: 'image'; readonly width: number; readonly height: number; readonly filename: string }[]
}

export interface AssistantMessageDisplay {
  readonly id: string
  readonly type: 'assistant_message'
  readonly content: string
  readonly timestamp: number
}

export interface ThinkBlockStep {
  readonly id: string
  readonly type: 'thinking' | 'tool'
  readonly content?: string  // For thinking
  readonly toolKey?: string  // For tool
  readonly cluster?: string  // Visual cluster key for grouping consecutive steps
  readonly input?: unknown
  readonly result?: ToolResult
  readonly label?: string
  readonly visualState?: unknown  // Managed by visual reducer registry
}

export interface ThinkBlockMessage {
  readonly id: string
  readonly type: 'think_block'
  readonly status: 'active' | 'completed'
  readonly steps: readonly ThinkBlockStep[]
  readonly timestamp: number
  readonly completedAt?: number
}

export interface InterruptedMessage {
  readonly id: string
  readonly type: 'interrupted'
  readonly timestamp: number
  readonly context: 'root' | 'fork'
  readonly allKilled?: boolean
}

export interface UnexpectedErrorMessage {
  readonly id: string
  readonly type: 'unexpected_error'
  readonly tag: string | null
  readonly message: string
  readonly timestamp: number
}

export interface ForkResultMessage {
  readonly id: string
  readonly type: 'fork_result'
  readonly forkId: string
  readonly task: string
  readonly result: unknown
  readonly timestamp: number
}

export interface ForkActivityToolCounts {
  readonly commands: number
  readonly reads: number
  readonly writes: number
  readonly edits: number
  readonly searches: number
  readonly webSearches: number
  readonly artifactWrites: number
  readonly artifactUpdates: number
  readonly clicks: number
  readonly navigations: number
  readonly inputs: number
  readonly evaluations: number
  readonly other: number
}

export interface ForkActivityMessage {
  readonly id: string
  readonly type: 'fork_activity'
  readonly forkId: string
  readonly name: string
  readonly role: string
  readonly status: 'running' | 'completed'
  readonly startedAt: number
  readonly completedAt?: number
  readonly resumeCount?: number
  readonly toolCounts: ForkActivityToolCounts
  readonly artifactNames: readonly string[]
  readonly timestamp: number
}

export interface AgentCommunicationMessage {
  readonly id: string
  readonly type: 'agent_communication'
  readonly direction: 'to_agent' | 'from_agent'
  readonly agentId: string
  readonly agentName?: string
  readonly agentRole?: string
  readonly forkId: string | null
  readonly content: string
  readonly preview: string
  readonly timestamp: number
}

export interface ApprovalRequestMessage {
  readonly id: string
  readonly type: 'approval_request'
  readonly toolCallId: string
  readonly toolKey: string
  readonly input: unknown
  readonly reason: string
  readonly status: 'pending' | 'approved' | 'rejected'
  readonly timestamp: number
  readonly display?: ToolDisplay
}

export type DisplayMessage =
  | UserMessageDisplay
  | QueuedUserMessageDisplay
  | AssistantMessageDisplay
  | ThinkBlockMessage
  | InterruptedMessage
  | UnexpectedErrorMessage
  | ForkResultMessage
  | ForkActivityMessage
  | AgentCommunicationMessage
  | ApprovalRequestMessage

/** Per-fork display state */
export interface DisplayState {
  readonly status: 'idle' | 'streaming'
  readonly messages: readonly DisplayMessage[]
  readonly currentTurnId: string | null  // Tracks active turn for queuing decision
  readonly streamingMessageId: string | null  // Tracks streaming assistant message
  readonly activeThinkBlockId: string | null
  readonly showButton: 'send' | 'stop'
}

// =============================================================================
// Helpers
// =============================================================================

const generateId = () => createId()

/**
 * Find the index where new content should be inserted (before queued messages).
 * Returns the index of the first queued message, or messages.length if none.
 */
function findInsertionIndex(messages: readonly DisplayMessage[]): number {
  const queuedIndex = messages.findIndex(m => m.type === 'queued_user_message')
  return queuedIndex === -1 ? messages.length : queuedIndex
}

/**
 * Insert a message before queued messages (or at end if no queued messages).
 * Returns a new array.
 */
function insertBeforeQueuedMessages(
  messages: readonly DisplayMessage[],
  message: DisplayMessage
): DisplayMessage[] {
  const result = [...messages]
  const insertIndex = findInsertionIndex(result)
  result.splice(insertIndex, 0, message)
  return result
}

const findThinkBlock = (messages: readonly DisplayMessage[], id: string): ThinkBlockMessage | undefined =>
  messages.find((m): m is ThinkBlockMessage => m.type === 'think_block' && m.id === id)

const updateMessageById = <T extends DisplayMessage>(
  messages: readonly DisplayMessage[],
  id: string,
  updater: (msg: T) => T
): DisplayMessage[] =>
  messages.map(m => m.id === id ? updater(m as T) : m)

/**
 * Ensures there's an active ThinkBlock, creating one if needed.
 * New ThinkBlocks are inserted BEFORE queued messages.
 */
function ensureThinkBlock(
  state: DisplayState,
  timestamp: number
): { fork: DisplayState; thinkBlockId: string } {
  if (state.activeThinkBlockId) {
    // Verify it still exists
    const existing = findThinkBlock(state.messages, state.activeThinkBlockId)
    if (existing && existing.status === 'active') {
      return { fork: state, thinkBlockId: state.activeThinkBlockId }
    }
  }

  const thinkBlockId = generateId()
  const thinkBlock: ThinkBlockMessage = {
    id: thinkBlockId,
    type: 'think_block',
    status: 'active',
    steps: [],
    timestamp
  }

  // Insert before queued messages
  const messages = insertBeforeQueuedMessages(state.messages, thinkBlock)

  return {
    fork: {
      ...state,
      messages,
      activeThinkBlockId: thinkBlockId
    },
    thinkBlockId
  }
}

const addStepToThinkBlock = (
  messages: readonly DisplayMessage[],
  thinkBlockId: string,
  step: ThinkBlockStep
): DisplayMessage[] =>
  updateMessageById<ThinkBlockMessage>(messages, thinkBlockId, (block) => ({
    ...block,
    steps: [...block.steps, step]
  }))

const updateStepInThinkBlock = (
  messages: readonly DisplayMessage[],
  thinkBlockId: string,
  stepId: string,
  updater: (step: ThinkBlockStep) => ThinkBlockStep
): DisplayMessage[] =>
  updateMessageById<ThinkBlockMessage>(messages, thinkBlockId, (block) => ({
    ...block,
    steps: block.steps.map(s => s.id === stepId ? updater(s) : s)
  }))

function closeThinkBlock(state: DisplayState, timestamp: number): DisplayState {
  if (!state.activeThinkBlockId) return state

  const block = findThinkBlock(state.messages, state.activeThinkBlockId)
  if (!block) return state

  // Remove empty think blocks
  if (block.steps.length === 0) {
    return {
      ...state,
      messages: state.messages.filter(m => m.id !== state.activeThinkBlockId),
      activeThinkBlockId: null
    }
  }

  return {
    ...state,
    messages: updateMessageById<ThinkBlockMessage>(
      state.messages,
      state.activeThinkBlockId,
      (b) => ({ ...b, status: 'completed', completedAt: timestamp })
    ),
    activeThinkBlockId: null
  }
}

// Standalone signal definition (needed for self-referencing in signalHandlers)
const forkToolStepSignalDef = Signal.create<{ forkId: string | null; toolKey: string }>('Display/forkToolStep')
// Convert to Signal for use in signalHandlers on() calls (which expect Signal, not SignalDef)
const forkToolStepSignal = Signal.fromDef<{ forkId: string | null; toolKey: string }, unknown>(forkToolStepSignalDef, 'Display')

const EMPTY_TOOL_COUNTS: ForkActivityToolCounts = {
  commands: 0, reads: 0, writes: 0, edits: 0, searches: 0, webSearches: 0, artifactWrites: 0, artifactUpdates: 0, clicks: 0, navigations: 0, inputs: 0, evaluations: 0, other: 0
}

function incrementToolCount(counts: ForkActivityToolCounts, toolKey: string): ForkActivityToolCounts {
  switch (toolKey) {
    case 'shell': return { ...counts, commands: counts.commands + 1 }
    case 'fileRead':
    case 'fileTree': return { ...counts, reads: counts.reads + 1 }
    case 'fileWrite': return { ...counts, writes: counts.writes + 1 }
    case 'fileEdit': return { ...counts, edits: counts.edits + 1 }
    case 'fileSearch':
    case 'gather': return { ...counts, searches: counts.searches + 1 }
    case 'webSearch': return { ...counts, webSearches: counts.webSearches + 1 }
    case 'artifactWrite': return { ...counts, artifactWrites: counts.artifactWrites + 1 }
    case 'artifactUpdate': return { ...counts, artifactUpdates: counts.artifactUpdates + 1 }
    case 'click':
    case 'doubleClick':
    case 'rightClick':
    case 'drag': return { ...counts, clicks: counts.clicks + 1 }
    case 'navigate':
    case 'goBack':
    case 'switchTab':
    case 'newTab': return { ...counts, navigations: counts.navigations + 1 }
    case 'type': return { ...counts, inputs: counts.inputs + 1 }
    case 'evaluate': return { ...counts, evaluations: counts.evaluations + 1 }
    default: return { ...counts, other: counts.other + 1 }
  }
}

function findLastIndex<T>(arr: readonly T[], pred: (item: T) => boolean): number {
  for (let i = arr.length - 1; i >= 0; i--) {
    if (pred(arr[i])) return i
  }
  return -1
}

function moveMessageToEndBeforeQueue<T extends DisplayMessage>(
  messages: readonly DisplayMessage[],
  id: string,
  updater?: (message: T) => DisplayMessage
): DisplayMessage[] {
  const index = messages.findIndex(m => m.id === id)
  if (index === -1) return [...messages]
  const target = messages[index] as T
  const updated = updater ? updater(target) : target
  const remaining = [...messages.slice(0, index), ...messages.slice(index + 1)]
  return insertBeforeQueuedMessages(remaining, updated)
}

function toPreview(text: string): string {
  const normalized = text.replace(/\s+/g, ' ').trim()
  if (normalized.length <= 120) return normalized
  return normalized.slice(0, 117) + '...'
}

// =============================================================================
// Projection
// =============================================================================

export const DisplayProjection = Projection.defineForked<AppEvent, DisplayState>()({
  name: 'Display',

  reads: [AgentProjection, WorkingStateProjection] as const,

  initialFork: {
    status: 'idle',
    messages: [],
    currentTurnId: null,
    streamingMessageId: null,
    activeThinkBlockId: null,
    showButton: 'send',

  },

  signals: {
    restoreQueuedMessages: Signal.create<{ forkId: string | null; messages: string[] }>('Display/restoreQueuedMessages'),
    forkToolStep: forkToolStepSignalDef
  },

  eventHandlers: {
    user_message: ({ event, fork }) => {
      const messageId = generateId()

      // If currently in a turn, queue the message
      // Queued messages always go at the END
      if (fork.currentTurnId !== null) {
        return {
          ...fork,
          messages: [
            ...fork.messages,
            {
              id: messageId,
              type: 'queued_user_message' as const,
              content: textOf(event.content),
              timestamp: event.timestamp,
              taskMode: event.taskMode,
              attachments: (event.attachments ?? [])
                .filter((a): a is Extract<typeof a, { type: 'image' }> => a.type === 'image')
                .map(a => ({ type: a.type, width: a.width, height: a.height, filename: a.filename }))
            }
          ]
        }
      }

      // Otherwise add as normal user_message
      return {
        ...fork,
        messages: [
          ...fork.messages,
          {
            id: messageId,
            type: 'user_message' as const,
            content: textOf(event.content),
            timestamp: event.timestamp,
            taskMode: event.taskMode,
            attachments: (event.attachments ?? [])
              .filter((a): a is Extract<typeof a, { type: 'image' }> => a.type === 'image')
              .map(a => ({ type: a.type, width: a.width, height: a.height, filename: a.filename }))
          }
        ]
      }
    },

    turn_started: ({ event, fork }) => {
      // Check if there are queued messages to promote
      const hasQueuedMessages = fork.messages.some(m => m.type === 'queued_user_message')

      // If promoting queued messages, close the current think block first
      // so the new think block appears AFTER the promoted user messages
      const stateBeforePromotion = hasQueuedMessages ? closeThinkBlock(fork, event.timestamp) : fork

      // Promote queued messages to user messages
      const messages = stateBeforePromotion.messages.map(msg => {
        if (msg.type === 'queued_user_message') {
          return {
            ...msg,
            type: 'user_message' as const
          }
        }
        return msg
      })

      // Ensure ThinkBlock exists - reuse existing if there is one, create if not
      const stateWithMessages = { ...stateBeforePromotion, messages }
      const { fork: newState } = ensureThinkBlock(stateWithMessages, event.timestamp)

      return {
        ...newState,
        currentTurnId: event.turnId,  // Track the turn for queuing
        status: 'streaming' as const,
        showButton: 'stop' as const
      }
    },

    message_start: ({ event, fork }) => {
      // Ignore if not for current turn
      if (fork.currentTurnId !== event.turnId) {
        return fork
      }

      if (event.dest !== 'user') {
        return fork
      }

      // Close any active ThinkBlock before starting assistant message
      const closedState = closeThinkBlock(fork, event.timestamp)

      const msgId = generateId()
      const assistantMessage: AssistantMessageDisplay = {
        id: msgId,
        type: 'assistant_message',
        content: '',
        timestamp: event.timestamp
      }

      const messages = insertBeforeQueuedMessages(closedState.messages, assistantMessage)

      return {
        ...closedState,
        streamingMessageId: msgId,
        messages
      }
    },

    message_chunk: ({ event, fork }) => {
      // Ignore if not for current turn
      if (fork.currentTurnId !== event.turnId) {
        return fork
      }

      // Append chunk to current streaming message (if any)
      if (!fork.streamingMessageId) {
        return fork
      }
      return {
        ...fork,
        messages: updateMessageById<AssistantMessageDisplay>(
          fork.messages,
          fork.streamingMessageId,
          (msg) => ({ ...msg, content: msg.content + event.text })
        )
      }
    },

    message_end: ({ event, fork }) => {
      // Ignore if not for current turn
      if (fork.currentTurnId !== event.turnId) {
        return fork
      }

      // Don't create optimistic ThinkBlock if there are queued messages.
      // turn_started will create it at the correct position after promoting them.
      const hasQueuedMessages = fork.messages.some(m => m.type === 'queued_user_message')
      if (hasQueuedMessages) {
        return {
          ...fork,
          streamingMessageId: null  // Message streaming done
        }
      }

      // Create optimistic ThinkBlock for potential follow-up work
      // It will be removed if empty when turn_completed arrives
      const { fork: newState } = ensureThinkBlock(fork, event.timestamp)

      return {
        ...newState,
        streamingMessageId: null  // Message streaming done
      }
    },

    thinking_chunk: ({ event, fork }) => {
      // Ignore if not for current turn
      if (fork.currentTurnId !== event.turnId) {
        return fork
      }

      const { fork: newState, thinkBlockId } = ensureThinkBlock(fork, event.timestamp)
      const block = findThinkBlock(newState.messages, thinkBlockId)

      if (!block) return newState

      // Find existing thinking step or create new one
      const lastStep = block.steps[block.steps.length - 1]
      if (lastStep?.type === 'thinking') {
        return {
          ...newState,
          messages: updateStepInThinkBlock(
            newState.messages,
            thinkBlockId,
            lastStep.id,
            (s) => ({ ...s, content: (s.content ?? '') + event.text })
          )
        }
      }

      // Create new thinking step
      const stepId = generateId()
      return {
        ...newState,
        messages: addStepToThinkBlock(newState.messages, thinkBlockId, {
          id: stepId,
          type: 'thinking',
          content: event.text
        })
      }
    },

    lens_start: ({ event, fork }) => {
      if (fork.currentTurnId !== event.turnId) return fork
      const { fork: newState, thinkBlockId } = ensureThinkBlock(fork, event.timestamp)
      const block = findThinkBlock(newState.messages, thinkBlockId)
      if (block) {
        const lastStep = block.steps[block.steps.length - 1]
        if (lastStep && lastStep.type === 'thinking' && (lastStep.content ?? '').length > 0) {
          // Append space separator before next lens
          return {
            ...newState,
            messages: updateStepInThinkBlock(
              newState.messages,
              thinkBlockId,
              lastStep.id,
              (s) => ({ ...s, content: (s.content ?? '') + ' ' })
            )
          }
        }
      }
      // Create new thinking step if none exists
      return {
        ...newState,
        messages: addStepToThinkBlock(newState.messages, thinkBlockId, {
          id: generateId(),
          type: 'thinking',
          content: ''
        })
      }
    },

    lens_chunk: ({ event, fork }) => {
      if (fork.currentTurnId !== event.turnId) return fork
      const { fork: newState, thinkBlockId } = ensureThinkBlock(fork, event.timestamp)
      const block = findThinkBlock(newState.messages, thinkBlockId)
      if (!block) return newState
      const lastStep = block.steps[block.steps.length - 1]
      if (!lastStep || lastStep.type !== 'thinking') return newState
      return {
        ...newState,
        messages: updateStepInThinkBlock(
          newState.messages,
          thinkBlockId,
          lastStep.id,
          (s) => ({ ...s, content: (s.content ?? '') + event.text })
        )
      }
    },

    lens_end: ({ fork }) => fork,



    tool_event: ({ event, fork, read, emit }) => {
      const inner = event.event
      const registry = getVisualRegistry()
      const visual = registry?.get(event.toolKey)

      switch (inner._tag) {
        case 'ToolInputStarted': {
          // Emit signal for parent fork activity tracking (before any early returns)
          emit.forkToolStep({ forkId: event.forkId, toolKey: event.toolKey })

          // Ignore if not for current turn
          if (fork.currentTurnId !== event.turnId) {
            return fork
          }

          // Consult agent definition's display policy
          const agentState = read(AgentProjection)
          if (isToolHidden(event.toolKey, event.forkId, undefined, agentState)) {
            return fork
          }

          const { fork: newState, thinkBlockId } = ensureThinkBlock(fork, event.timestamp)

          // Initialize visual state from registry, then reduce the ToolInputStarted event
          const initialVisualState = visual
            ? visual.reduce(visual.initial, inner)
            : undefined

          return {
            ...newState,
            messages: addStepToThinkBlock(newState.messages, thinkBlockId, {
              id: event.toolCallId,
              type: 'tool',
              toolKey: event.toolKey,
              cluster: visual?.cluster,
              input: undefined,
              label: event.toolKey + '()',
              visualState: initialVisualState,
            })
          }
        }

        case 'ToolInputReady': {
          if (fork.currentTurnId !== event.turnId) return fork
          if (!fork.activeThinkBlockId) return fork

          // Update the step with the full parsed input and regenerate label
          const label = generateToolLabel(event.toolKey, inner.input)
          return {
            ...fork,
            messages: updateStepInThinkBlock(
              fork.messages,
              fork.activeThinkBlockId,
              event.toolCallId,
              (s) => ({
                ...s,
                input: inner.input,
                label,
                visualState: visual && s.visualState !== undefined
                  ? visual.reduce(s.visualState, inner)
                  : s.visualState,
              })
            )
          }
        }

        case 'ToolExecutionEnded': {
          // Ignore if not for current turn
          if (fork.currentTurnId !== event.turnId) {
            return fork
          }

          // Consult agent definition's display policy
          const agentState = read(AgentProjection)
          if (isToolHidden(event.toolKey, event.forkId, undefined, agentState)) {
            return fork
          }

          if (!fork.activeThinkBlockId) return fork

          // Map XmlToolResult → ToolResult for display (include tool-emitted display data)
          const result = mapXmlToolResultForDisplay(inner.result, event.display)

          return {
            ...fork,
            messages: updateStepInThinkBlock(
              fork.messages,
              fork.activeThinkBlockId,
              event.toolCallId,
              (s) => ({
                ...s,
                result,
                visualState: visual && s.visualState !== undefined
                  ? visual.reduce(s.visualState, inner)
                  : s.visualState,
              })
            )
          }
        }

        default: {
          // All other streaming events: reduce visual state only
          if (fork.currentTurnId !== event.turnId) return fork
          if (!fork.activeThinkBlockId) return fork
          if (!visual) return fork

          return {
            ...fork,
            messages: updateStepInThinkBlock(
              fork.messages,
              fork.activeThinkBlockId,
              event.toolCallId,
              (s) => ({
                ...s,
                visualState: s.visualState !== undefined
                  ? visual.reduce(s.visualState, inner)
                  : s.visualState,
              })
            )
          }
        }
      }
    },

    turn_completed: ({ event, fork }) => {
      // Ignore if not for current turn
      if (fork.currentTurnId !== event.turnId) {
        return fork
      }

      // Determine if we'll continue (same logic as WorkingState)
      let willContinue: boolean
      if (event.result.success) {
        willContinue = event.result.turnDecision === 'continue'
      } else {
        willContinue = !event.result.cancelled
      }

      // Close ThinkBlock only if we're becoming stable (won't continue)
      // If we will continue, keep ThinkBlock open for next turn's steps
      if (!willContinue) {
        const closedState = closeThinkBlock(fork, event.timestamp)
        return {
          ...closedState,
          currentTurnId: null,
          status: 'idle' as const,
          streamingMessageId: null,
          showButton: 'send' as const,
      
        }
      }

      // Keep ThinkBlock open - more work coming
      // Clear currentTurnId so next turn_started can set it
      return {
        ...fork,
        currentTurnId: null,
        status: 'idle' as const,
        streamingMessageId: null,
        showButton: 'send' as const,
    
      }
    },

    turn_unexpected_error: ({ event, fork }) => {
      const closedState = closeThinkBlock(fork, event.timestamp)

      const errorMessage: UnexpectedErrorMessage = {
        id: generateId(),
        type: 'unexpected_error',
        tag: null,
        message: event.message,
        timestamp: event.timestamp
      }

      return {
        ...closedState,
        messages: [...closedState.messages, errorMessage],
        currentTurnId: null,
        status: 'idle' as const,
        streamingMessageId: null,
        showButton: 'send' as const
      }
    },

    interrupt: ({ event, fork, emit }) => {
      // Find queued messages before removing them
      const queuedMessages = fork.messages.filter(
        (m): m is QueuedUserMessageDisplay => m.type === 'queued_user_message'
      )

      // Emit restore signal if there are queued messages
      if (queuedMessages.length > 0) {
        emit.restoreQueuedMessages({
          forkId: event.forkId,
          messages: queuedMessages.map(m => m.content)
        })
      }

      // Close think block and remove queued messages
      const closedState = closeThinkBlock(fork, event.timestamp)
      const messagesWithoutQueued = closedState.messages.filter(
        m => m.type !== 'queued_user_message'
      )

      return {
        ...closedState,
        currentTurnId: null,
        status: 'idle' as const,
        streamingMessageId: null,
        showButton: 'send' as const,
        messages: [
          ...messagesWithoutQueued,
          {
            id: generateId(),
            type: 'interrupted' as const,
            timestamp: event.timestamp,
            context: event.forkId === null ? 'root' as const : 'fork' as const,
            ...(event.allKilled ? { allKilled: true } : {}),
          }
        ]
      }
    },

    agent_dismissed: ({ event, fork }) => {
      // Keep fork state for display/debug history
      return fork
    },


  },

  signalHandlers: (on) => [
    // Insert inline fork activity block in parent's display when agent is created
    on(AgentProjection.signals.agentCreated, ({ value, state }) => {
      const { forkId, parentForkId, name, role } = value
      const parentState = state.forks.get(parentForkId)
      if (!parentState) return state

      const msg: ForkActivityMessage = {
        id: generateId(),
        type: 'fork_activity',
        forkId,
        name,
        role,
        status: 'running',
        startedAt: value.timestamp,
        resumeCount: 0,
        toolCounts: EMPTY_TOOL_COUNTS,
        artifactNames: [],
        timestamp: value.timestamp
      }

      const newMessages = insertBeforeQueuedMessages(parentState.messages, msg)
      return {
        ...state,
        forks: new Map(state.forks).set(parentForkId, { ...parentState, messages: newMessages })
      }
    }),

    // Update tool counts in parent's ForkActivityMessage when a tool step runs
    on(forkToolStepSignal, ({ value, state, read }) => {
      const { forkId, toolKey } = value
      if (forkId === null) return state  // root fork tools, no parent activity to update

      const agentState = read(AgentProjection)
      const agent = getAgentByForkId(agentState, forkId)
      if (!agent) return state

      const parentState = state.forks.get(agent.parentForkId)
      if (!parentState) return state

      const msgIndex = findLastIndex(parentState.messages, (m: DisplayMessage) =>
        m.type === 'fork_activity' && m.forkId === forkId && m.status === 'running')
      if (msgIndex === -1) return state

      const msg = parentState.messages[msgIndex] as ForkActivityMessage
      const newCounts = incrementToolCount(msg.toolCounts, toolKey)
      const newMessages = [...parentState.messages]
      newMessages[msgIndex] = { ...msg, toolCounts: newCounts }

      return {
        ...state,
        forks: new Map(state.forks).set(agent.parentForkId, { ...parentState, messages: newMessages })
      }
    }),

    // Mark fork activity as completed when agent becomes idle
    on(AgentStatusBridgeProjection.signals.agentBecameIdle, ({ value, state }) => {
      const { forkId, parentForkId } = value

      const parentState = state.forks.get(parentForkId)
      if (!parentState) return state

      const msgIndex = findLastIndex(parentState.messages, (m: DisplayMessage) =>
        m.type === 'fork_activity' && m.forkId === forkId && m.status === 'running')
      if (msgIndex === -1) return state

      const msg = parentState.messages[msgIndex] as ForkActivityMessage
      const newMessages = [...parentState.messages]
      newMessages[msgIndex] = { ...msg, status: 'completed', completedAt: value.timestamp }

      return {
        ...state,
        forks: new Map(state.forks).set(parentForkId, { ...parentState, messages: newMessages })
      }
    }),

    on(AgentProjection.signals.agentDismissed, ({ value, state }) => {
      const { forkId, parentForkId } = value
      const parentState = state.forks.get(parentForkId)
      if (!parentState) return state

      const msgIndex = findLastIndex(parentState.messages, (m: DisplayMessage) =>
        m.type === 'fork_activity' && m.forkId === forkId && m.status === 'running')
      if (msgIndex === -1) return state

      const msg = parentState.messages[msgIndex] as ForkActivityMessage
      const newMessages = [...parentState.messages]
      newMessages[msgIndex] = { ...msg, status: 'completed', completedAt: value.timestamp }

      return {
        ...state,
        forks: new Map(state.forks).set(parentForkId, { ...parentState, messages: newMessages })
      }
    }),

    on(AgentStatusBridgeProjection.signals.agentResumed, ({ value, state }) => {
      const { forkId, parentForkId } = value
      const parentState = state.forks.get(parentForkId)
      if (!parentState) return state

      const msgIndex = findLastIndex(parentState.messages, (m: DisplayMessage) =>
        m.type === 'fork_activity' && m.forkId === forkId)
      if (msgIndex === -1) return state

      const message = parentState.messages[msgIndex] as ForkActivityMessage
      if (!message.completedAt) return state  // never went idle — first run, not a resume

      const moved = moveMessageToEndBeforeQueue<ForkActivityMessage>(parentState.messages, message.id, (msg) => ({
        ...msg,
        status: 'running',
        completedAt: undefined,
        resumeCount: (msg.resumeCount ?? 0) + 1,
        timestamp: value.timestamp,
      }))

      return {
        ...state,
        forks: new Map(state.forks).set(parentForkId, { ...parentState, messages: moved })
      }
    }),

    on(AgentProjection.signals.agentMessage, ({ value, state, read }) => {
      const { targetForkId, agentId, message, timestamp } = value
      const content = message.trim()
      if (!content) return state

      const agentState = read(AgentProjection)
      const targetAgent = getAgentByForkId(agentState, targetForkId)
      const parentForkId = targetAgent?.parentForkId ?? null
      let nextState = state

      const parentDisplayFork = state.forks.get(parentForkId)
      if (parentDisplayFork) {
        const communication: AgentCommunicationMessage = {
          id: generateId(),
          type: 'agent_communication',
          direction: 'to_agent',
          agentId,
          agentName: targetAgent?.name,
          agentRole: targetAgent?.role,
          forkId: targetForkId,
          content,
          preview: toPreview(content),
          timestamp
        }

        nextState = {
          ...nextState,
          forks: new Map(nextState.forks).set(parentForkId, {
            ...parentDisplayFork,
            messages: insertBeforeQueuedMessages(parentDisplayFork.messages, communication)
          })
        }
      }

      const childDisplayFork = nextState.forks.get(targetForkId)
      if (!childDisplayFork) return nextState

      const childCommunication: AgentCommunicationMessage = {
        id: generateId(),
        type: 'agent_communication',
        direction: 'from_agent',
        agentId: 'orchestrator',
        agentName: 'Orchestrator',
        agentRole: 'Orchestrator',
        forkId: targetForkId,
        content,
        preview: toPreview(content),
        timestamp
      }

      nextState = {
        ...nextState,
        forks: new Map(nextState.forks).set(targetForkId, {
          ...childDisplayFork,
          messages: insertBeforeQueuedMessages(childDisplayFork.messages, childCommunication)
        })
      }

      return nextState
    }),

    on(AgentProjection.signals.agentResponse, ({ value, state, read }) => {
      const { targetForkId, agentId, message, timestamp } = value
      const content = message.trim()
      if (!content) return state

      const agentState = read(AgentProjection)
      const sourceAgent = agentState.agents.get(agentId)

      let nextState = state

      const displayFork = state.forks.get(targetForkId)
      if (displayFork) {
        const communication: AgentCommunicationMessage = {
          id: generateId(),
          type: 'agent_communication',
          direction: 'from_agent',
          agentId,
          agentName: sourceAgent?.name,
          agentRole: sourceAgent?.role,
          forkId: targetForkId,
          content,
          preview: toPreview(content),
          timestamp
        }

        nextState = {
          ...nextState,
          forks: new Map(nextState.forks).set(targetForkId, {
            ...displayFork,
            messages: insertBeforeQueuedMessages(displayFork.messages, communication)
          })
        }
      }

      if (!sourceAgent) return nextState
      const sourceDisplayFork = nextState.forks.get(sourceAgent.forkId)
      if (!sourceDisplayFork) return nextState

      const childCommunication: AgentCommunicationMessage = {
        id: generateId(),
        type: 'agent_communication',
        direction: 'to_agent',
        agentId: 'orchestrator',
        agentName: 'Orchestrator',
        agentRole: 'Orchestrator',
        forkId: sourceAgent.forkId,
        content,
        preview: toPreview(content),
        timestamp
      }

      nextState = {
        ...nextState,
        forks: new Map(nextState.forks).set(sourceAgent.forkId, {
          ...sourceDisplayFork,
          messages: insertBeforeQueuedMessages(sourceDisplayFork.messages, childCommunication)
        })
      }

      return nextState
    })
  ]
})

// =============================================================================
// Helpers
// =============================================================================

function generateToolLabel(toolKey: string, input: unknown): string {
  // Artifact tools
  if (toolKey === 'artifactRead' && input && typeof input === 'object' && 'id' in input) {
    return `Read artifact "${(input as { id: string }).id}"`
  }
  if (toolKey === 'artifactWrite' && input && typeof input === 'object' && 'id' in input) {
    return `Wrote artifact "${(input as { id: string }).id}"`
  }
  if (toolKey === 'artifactUpdate' && input && typeof input === 'object' && 'id' in input) {
    return `Updated artifact "${(input as { id: string }).id}"`
  }


  // For shell, show the command
  if (toolKey === 'shell' && input && typeof input === 'object' && 'command' in input) {
    const cmd = (input as { command: string }).command
    const shortCmd = cmd.length > 50 ? cmd.slice(0, 47) + '...' : cmd
    return `Running: ${shortCmd}`
  }

  // For webSearch, show the query
  if (toolKey === 'webSearch' && input && typeof input === 'object' && 'query' in input) {
    const query = (input as { query: string }).query
    const shortQuery = query.length > 60 ? query.slice(0, 57) + '...' : query
    return `Searching web for "${shortQuery}"`
  }

  // For fileRead, show the path
  if (toolKey === 'fileRead' && input && typeof input === 'object' && 'path' in input) {
    return `Read ${(input as { path: string }).path}`
  }

  // For fileWrite, show the path
  if (toolKey === 'fileWrite' && input && typeof input === 'object' && 'path' in input) {
    return `Wrote ${(input as { path: string }).path}`
  }

  // For fileEdit, show the path
  if (toolKey === 'fileEdit' && input && typeof input === 'object' && 'path' in input) {
    const edits = 'edits' in input && Array.isArray((input as any).edits) ? (input as any).edits.length : 0
    return 'Edited ' + (input as { path: string }).path + (edits > 0 ? ' (' + edits + (edits === 1 ? ' change' : ' changes') + ')' : '')
  }

  // For fileSearch, show the pattern
  if (toolKey === 'fileSearch' && input && typeof input === 'object' && 'pattern' in input) {
    const pattern = (input as { pattern: string }).pattern
    const shortPattern = pattern.length > 40 ? pattern.slice(0, 37) + '...' : pattern
    return `Searched for "${shortPattern}"`
  }

  // For fileTree, show the path
  if (toolKey === 'fileTree' && input && typeof input === 'object' && 'path' in input) {
    return `Listed ${(input as { path: string }).path}`
  }

  // For agent creation
  if (toolKey === 'agentCreate' && input && typeof input === 'object' && 'name' in input) {
    return `Started agent "${(input as { name: string }).name}"`
  }

  // Default: just the tool key
  return `${toolKey}()`
}
