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
import type { AppEvent, ToolDisplay } from '../events'

import { AgentRoutingProjection } from './agent-routing'
import { AgentStatusProjection, getAgentByForkId } from './agent-status'
import { WorkingStateProjection, type PendingInboundCommunication } from './working-state'


import { textOf } from '../content'
import { createId } from '../util/id'

import { finalizeOpenToolStepsAsInterruptedInSteps } from './display-interrupt'
import { TOOL_DEFINITIONS, type ToolKey } from '../tools/tool-definitions'
import { createToolHandle, type ToolHandle, type ToolState } from '../tools/tool-handle'


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

export interface ThinkingStep {
  readonly id: string
  readonly type: 'thinking'
  readonly content?: string
  readonly label?: string
}

export interface ToolStep {
  readonly id: string
  readonly type: 'tool'
  readonly toolKey: ToolKey
  readonly cluster?: string
  readonly state?: ToolState
}

export interface CommunicationStep {
  readonly id: string
  readonly type: 'communication'
  readonly streamId?: string
  readonly direction: 'to_agent' | 'from_agent'
  readonly agentId: string
  readonly agentName?: string
  readonly agentRole?: string
  readonly forkId: string | null
  readonly content: string
  readonly preview: string
  readonly timestamp: number
  readonly status?: 'streaming' | 'completed'
}

export interface SubagentStartedStep {
  readonly id: string
  readonly type: 'subagent_started'
  readonly subagentType: string
  readonly subagentId: string
  readonly title: string
  readonly resumed: boolean
}

export interface SubagentFinishedStep {
  readonly id: string
  readonly type: 'subagent_finished'
  readonly subagentType: string
  readonly subagentId: string
  readonly cumulativeTotalTimeMs: number
  readonly cumulativeTotalToolsUsed: number
  readonly resumed: boolean
}

export interface SubagentKilledStep {
  readonly id: string
  readonly type: 'subagent_killed'
  readonly subagentType: string
  readonly subagentId: string
  readonly title: string
}

export interface SubagentUserKilledStep {
  readonly id: string
  readonly type: 'subagent_user_killed'
  readonly subagentType: string
  readonly subagentId: string
  readonly title: string
}

export type ThinkBlockStep =
  | ThinkingStep
  | ToolStep
  | CommunicationStep
  | SubagentStartedStep
  | SubagentFinishedStep
  | SubagentKilledStep
  | SubagentUserKilledStep

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
  readonly webFetches: number
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
  readonly createdAt: number
  readonly activeSince: number
  readonly accumulatedActiveMs: number
  readonly completedAt?: number
  readonly resumeCount?: number
  readonly toolCounts: ForkActivityToolCounts
  readonly timestamp: number
}

export interface AgentCommunicationMessage {
  readonly id: string
  readonly type: 'agent_communication'
  readonly streamId?: string
  readonly direction: 'to_agent' | 'from_agent'
  readonly agentId: string
  readonly agentName?: string
  readonly agentRole?: string
  readonly forkId: string | null
  readonly content: string
  readonly preview: string
  readonly timestamp: number
}

export interface BackgroundProcessMessage {
  readonly id: string
  readonly type: 'background_process'
  readonly pid: number
  readonly command: string
  readonly status: 'running' | 'exited' | 'killed'
  readonly stdout: string
  readonly stderr: string
  readonly exitCode: number | null
  readonly signal: string | null
  readonly timestamp: number
  readonly updatedAt: number
}

export interface ApprovalRequestMessage {
  readonly id: string
  readonly type: 'approval_request'
  readonly toolCallId: string
  readonly toolKey: ToolKey
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
  | BackgroundProcessMessage
  | ApprovalRequestMessage

/** Per-fork display state */
export interface PendingInboundCommunicationDisplay extends PendingInboundCommunication {}



export interface DisplayState {
  readonly status: 'idle' | 'streaming'
  readonly messages: readonly DisplayMessage[]
  readonly pendingInboundCommunications: readonly PendingInboundCommunicationDisplay[]
  readonly currentTurnId: string | null  // Tracks active turn for queuing decision
  readonly streamingMessageId: string | null  // Tracks streaming assistant message
  readonly activeThinkBlockId: string | null
  readonly showButton: 'send' | 'stop'
  readonly toolHandles: { readonly [callId: string]: ToolHandle }
}

// =============================================================================
// Helpers
// =============================================================================

const generateId = () => createId()



const getVisualState = (fork: DisplayState, callId: string): ToolState | undefined =>
  fork.toolHandles[callId]?.state

/**
 * Find the index where new content should be inserted (before queued messages).
 * Returns the index of the first queued message, or messages.length if none.
 */
export function findInsertionIndex(messages: readonly DisplayMessage[]): number {
  const queuedIndex = messages.findIndex(m => m.type === 'queued_user_message')
  return queuedIndex === -1 ? messages.length : queuedIndex
}

/**
 * Insert a message before queued messages (or at end if no queued messages).
 * Returns a new array.
 */
export function insertBeforeQueuedMessages(
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
  // NOTE: This generic cast is intentionally retained; TS cannot infer T inside map callback.
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

export function finalizeOpenToolStepsAsInterrupted(state: DisplayState): DisplayState {
  if (!state.activeThinkBlockId) return state
  const block = findThinkBlock(state.messages, state.activeThinkBlockId)
  if (!block) return state

  let updatedHandles = state.toolHandles

  const nextSteps = finalizeOpenToolStepsAsInterruptedInSteps(block.steps, (_toolKey, state, stepId) => {
    if (!stepId) return state

    const handle = updatedHandles[stepId]
    if (!handle) {
      throw new Error(`Display invariant violated: missing tool handle for stepId ${stepId}`)
    }

    const nextHandle = handle.interrupt()
    updatedHandles = {
      ...updatedHandles,
      [stepId]: nextHandle,
    }

    return nextHandle.state
  })

  if (nextSteps === block.steps || nextSteps.every((step, i) => step === block.steps[i])) {
    return state
  }

  return {
    ...state,
    toolHandles: updatedHandles,
    messages: updateMessageById<ThinkBlockMessage>(
      state.messages, state.activeThinkBlockId,
      (b) => ({ ...b, steps: nextSteps })
    )
  }
}

// Standalone signal definition (needed for self-referencing in signalHandlers)
const forkToolStepSignalDef = Signal.create<{ forkId: string | null; toolKey: ToolKey }>('Display/forkToolStep')
// Convert to Signal for use in signalHandlers on() calls (which expect Signal, not SignalDef)
const forkToolStepSignal = Signal.fromDef<{ forkId: string | null; toolKey: ToolKey }, unknown>(forkToolStepSignalDef, 'Display')

const EMPTY_TOOL_COUNTS: ForkActivityToolCounts = {
  commands: 0,
  reads: 0,
  writes: 0,
  edits: 0,
  searches: 0,
  webSearches: 0,
  webFetches: 0,
  artifactWrites: 0,
  artifactUpdates: 0,
  clicks: 0,
  navigations: 0,
  inputs: 0,
  evaluations: 0,
  other: 0
}

function incrementToolCount(counts: ForkActivityToolCounts, toolKey: ToolKey): ForkActivityToolCounts {
  switch (toolKey) {
    case 'shell': return { ...counts, commands: counts.commands + 1 }
    case 'fileRead':
    case 'fileTree': return { ...counts, reads: counts.reads + 1 }
    case 'fileWrite': return { ...counts, writes: counts.writes + 1 }
    case 'fileEdit': return { ...counts, edits: counts.edits + 1 }
    case 'fileSearch': return { ...counts, searches: counts.searches + 1 }
    case 'webSearch': return { ...counts, webSearches: counts.webSearches + 1 }
    case 'webFetch': return { ...counts, webFetches: counts.webFetches + 1 }
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
    case 'agentCreate':
    case 'agentKill':
    case 'skill':
    case 'phase-submit':
    case 'workflow-submit':
    case 'phase-verdict':
    case 'scroll':
    case 'screenshot':
      return { ...counts, other: counts.other + 1 }
  }
}

function findLastIndex<T>(arr: readonly T[], pred: (item: T) => boolean): number {
  for (let i = arr.length - 1; i >= 0; i--) {
    if (pred(arr[i])) return i
  }
  return -1
}

function totalToolsUsed(counts: ForkActivityToolCounts): number {
  return counts.commands
    + counts.reads
    + counts.writes
    + counts.edits
    + counts.searches
    + counts.webSearches
    + counts.webFetches
    + counts.artifactWrites
    + counts.artifactUpdates
    + counts.clicks
    + counts.navigations
    + counts.inputs
    + counts.evaluations
    + counts.other
}

export function moveMessageToEndBeforeQueue<T extends DisplayMessage>(
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

export function toPreview(text: string): string {
  const normalized = text.replace(/\s+/g, ' ').trim()
  if (normalized.length <= 120) return normalized
  return normalized.slice(0, 117) + '...'
}

export function upsertStreamingCommunicationStep(
  fork: DisplayState,
  streamId: string,
  message: Omit<CommunicationStep, 'id' | 'type' | 'preview' | 'streamId' | 'status' | 'content'>,
  textDelta: string
): DisplayState {
  if (message.forkId === null) return fork
  const { fork: stateWithBlock, thinkBlockId } = ensureThinkBlock(fork, message.timestamp)
  const block = findThinkBlock(stateWithBlock.messages, thinkBlockId)
  const last = block?.steps[block.steps.length - 1]

  if (last?.type === 'communication' && last.streamId === streamId) {
    const content = last.content + textDelta
    return {
      ...stateWithBlock,
      messages: updateStepInThinkBlock(
        stateWithBlock.messages,
        thinkBlockId,
        last.id,
        (s) => s.type === 'communication'
          ? { ...s, content, preview: toPreview(content), status: 'streaming' }
          : s
      )
    }
  }

  const content = textDelta
  const step: CommunicationStep = {
    id: generateId(),
    type: 'communication',
    streamId,
    ...message,
    content,
    preview: toPreview(content),
    status: 'streaming',
  }

  return {
    ...stateWithBlock,
    messages: addStepToThinkBlock(stateWithBlock.messages, thinkBlockId, step)
  }
}

function isFileStreamToolKey(toolKey: ToolKey | undefined): toolKey is 'fileWrite' | 'fileEdit' {
  return toolKey === 'fileWrite' || toolKey === 'fileEdit'
}

interface WritePreviewState {
  path: string
  phase: string
  contentSoFar: string
  charCount: number
  lineCount: number
}

interface EditPreviewState {
  path: string
  phase: string
  oldStringSoFar: string
  newStringSoFar: string
  replaceAll: boolean
  childParsePhase: 'idle' | 'streaming_old' | 'streaming_new'
}

function hasFsWritePreview(
  state: unknown,
): state is WritePreviewState {
  if (!state || typeof state !== 'object') return false
  return 'path' in state
    && typeof state.path === 'string'
    && state.path.length > 0
    && 'contentSoFar' in state
    && typeof state.contentSoFar === 'string'
}

function hasEditPreview(
  state: unknown,
): state is EditPreviewState {
  if (!state || typeof state !== 'object') return false
  return 'path' in state
    && typeof state.path === 'string'
    && state.path.length > 0
    && 'oldStringSoFar' in state
    && typeof state.oldStringSoFar === 'string'
}

export interface InProgressFileView {
  filePath: string
  toolCallId: string
  toolKey: 'fileWrite' | 'fileEdit'
  forkId: string | null
  phase: string
  preview: {
    mode: 'write'
    contentSoFar: string
    charCount: number
    lineCount: number
  } | {
    mode: 'edit'
    oldStringSoFar: string
    newStringSoFar: string
    replaceAll: boolean
    childParsePhase: 'idle' | 'streaming_old' | 'streaming_new'
  }
}

function collectFileStreams(
  state: DisplayState,
): InProgressFileView[] {
  const streams: InProgressFileView[] = []

  for (const message of state.messages) {
    if (message.type !== 'think_block') continue

    for (const step of message.steps) {
      if (step.type !== 'tool' || !isFileStreamToolKey(step.toolKey)) continue

      if (step.toolKey === 'fileWrite' && hasFsWritePreview(step.state)) {
        const previewState = step.state
        streams.push({
          filePath: previewState.path,
          toolCallId: step.id,
          toolKey: step.toolKey,
          forkId: null,
          phase: previewState.phase,
          preview: {
            mode: 'write',
            contentSoFar: previewState.contentSoFar,
            charCount: previewState.charCount,
            lineCount: previewState.lineCount,
          },
        })
      }

      if (step.toolKey === 'fileEdit' && hasEditPreview(step.state)) {
        const previewState = step.state
        streams.push({
          filePath: previewState.path,
          toolCallId: step.id,
          toolKey: step.toolKey,
          forkId: null,
          phase: previewState.phase,
          preview: {
            mode: 'edit',
            oldStringSoFar: previewState.oldStringSoFar,
            newStringSoFar: previewState.newStringSoFar,
            replaceAll: previewState.replaceAll,
            childParsePhase: previewState.childParsePhase,
          },
        })
      }
    }
  }

  return streams
}

export function getInProgressFileStreams(
  state: DisplayState,
  resolvedPath?: string
): InProgressFileView[] {
  const streams = collectFileStreams(state)
  if (!resolvedPath) return streams
  return streams.filter(s => s.filePath === resolvedPath)
}


// =============================================================================
// Projection
// =============================================================================

export const DisplayProjection = Projection.defineForked<AppEvent, DisplayState>()({
  name: 'Display',

  reads: [AgentRoutingProjection, AgentStatusProjection, WorkingStateProjection] as const,

  initialFork: {
    status: 'idle',
    messages: [],
    pendingInboundCommunications: [],
    currentTurnId: null,
    streamingMessageId: null,
    activeThinkBlockId: null,
    showButton: 'send',
    toolHandles: {},
  },

  signals: {
    restoreQueuedMessages: Signal.create<{ forkId: string | null; messages: string[] }>('Display/restoreQueuedMessages'),
    forkToolStep: forkToolStepSignalDef
  },

  eventHandlers: {
    oneshot_task: ({ event, fork }) => {
      const messageId = generateId()
      return {
        ...fork,
        messages: [
          ...fork.messages,
          {
            id: messageId,
            type: 'user_message' as const,
            content: event.prompt,
            timestamp: event.timestamp,
            taskMode: true,
            attachments: [],
          }
        ]
      }
    },

    user_message: ({ event, fork }) => {
      const messageId = generateId()
      const content = textOf(event.content)
      const messageType: 'user_message' | 'queued_user_message' =
        fork.currentTurnId !== null ? 'queued_user_message' : 'user_message'

      return {
        ...fork,
        messages: [
          ...fork.messages,
          {
            id: messageId,
            type: messageType,
            content,
            timestamp: event.timestamp,
            taskMode: event.taskMode,
            attachments: (event.attachments ?? [])
              .filter((a): a is Extract<typeof a, { type: 'image' }> => a.type === 'image')
              .map(a => ({ type: a.type, width: a.width, height: a.height, filename: a.filename }))
          }
        ]
      }
    },

    skill_activated: ({ event, fork }) => {
      // Only show as user message when activated by user slash command
      if (event.source !== 'user') return fork

      const messageId = generateId()
      const content = event.message ? `/${event.skillName} ${event.message}` : `/${event.skillName}`
      return {
        ...fork,
        messages: [
          ...fork.messages,
          {
            id: messageId,
            type: 'user_message' as const,
            content,
            timestamp: event.timestamp,
            taskMode: false,
            attachments: [],
          }
        ]
      }
    },

    skill_completed: ({ event, fork }) => {
      const messageId = generateId()
      return {
        ...fork,
        messages: [
          ...fork.messages,
          {
            id: messageId,
            type: 'assistant_message' as const,
            content: `✓ Skill "${event.skillName}" completed`,
            timestamp: event.timestamp,
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
        showButton: 'stop' as const,
        pendingInboundCommunications: [],
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
            (s) => s.type === 'thinking'
              ? { ...s, content: (s.content ?? '') + event.text }
              : s
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
              (s) => s.type === 'thinking'
                ? { ...s, content: (s.content ?? '') + ' ' }
                : s
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
          (s) => s.type === 'thinking'
            ? { ...s, content: (s.content ?? '') + event.text }
            : s
        )
      }
    },

    lens_end: ({ fork }) => fork,



    tool_event: ({ event, fork, read, emit }) => {
      const inner = event.event

      let currentFork = fork
      if (inner._tag === 'ToolInputStarted') {
        const handle = createToolHandle(event.toolKey).process(inner)
        currentFork = {
          ...currentFork,
          toolHandles: {
            ...currentFork.toolHandles,
            [event.toolCallId]: handle,
          },
        }
      } else {
        const handle = currentFork.toolHandles[event.toolCallId]
        if (handle) {
          const nextHandle = handle.process(inner)
          currentFork = {
            ...currentFork,
            toolHandles: {
              ...currentFork.toolHandles,
              [event.toolCallId]: nextHandle,
            },
          }
        }
      }

      switch (inner._tag) {
        case 'ToolInputStarted': {
          // Emit signal for parent fork activity tracking (before any early returns)
          emit.forkToolStep({ forkId: event.forkId, toolKey: event.toolKey })

          // Ignore if not for current turn
          if (currentFork.currentTurnId !== event.turnId) {
            return currentFork
          }

          // Consult agent definition's display policy
          const agentState = read(AgentStatusProjection)
          const toolDef = TOOL_DEFINITIONS[event.toolKey]
          if ('display' in toolDef && toolDef.display === false) {
            return currentFork
          }

          const { fork: newState, thinkBlockId } = ensureThinkBlock(currentFork, event.timestamp)
          return {
            ...newState,
            messages: addStepToThinkBlock(newState.messages, thinkBlockId, {
              id: event.toolCallId,
              type: 'tool',
              toolKey: event.toolKey,
              state: getVisualState(newState, event.toolCallId),
            })
          }
        }

        case 'ToolInputReady': {
          if (currentFork.currentTurnId !== event.turnId) return currentFork
          if (!currentFork.activeThinkBlockId) return currentFork

          return {
            ...currentFork,
            messages: updateStepInThinkBlock(
              currentFork.messages,
              currentFork.activeThinkBlockId,
              event.toolCallId,
              (s) => s.type === 'tool'
                ? {
                    ...s,
                    state: getVisualState(currentFork, event.toolCallId) ?? s.state,
                  }
                : s
            )
          }
        }

        case 'ToolExecutionEnded': {
          // Ignore if not for current turn
          if (currentFork.currentTurnId !== event.turnId) {
            return currentFork
          }

          // Consult agent definition's display policy
          const agentState = read(AgentStatusProjection)
          const toolDef = TOOL_DEFINITIONS[event.toolKey]
          if ('display' in toolDef && toolDef.display === false) {
            return currentFork
          }

          if (!currentFork.activeThinkBlockId) return currentFork

          return {
            ...currentFork,
            messages: updateStepInThinkBlock(
              currentFork.messages,
              currentFork.activeThinkBlockId,
              event.toolCallId,
              (s) => s.type === 'tool'
                ? {
                    ...s,
                    state: getVisualState(currentFork, event.toolCallId) ?? s.state,
                  }
                : s
            )
          }
        }

        case 'ToolObservation':
          return currentFork

        default: {
          if (currentFork.currentTurnId !== event.turnId) return currentFork
          if (!currentFork.activeThinkBlockId) return currentFork

          const vs = getVisualState(currentFork, event.toolCallId)
          if (!vs) return currentFork

          return {
            ...currentFork,
            messages: updateStepInThinkBlock(
              currentFork.messages,
              currentFork.activeThinkBlockId,
              event.toolCallId,
              (s) => s.type === 'tool'
                ? { ...s, state: vs }
                : s
            )
          }
        }
      }
    },

    background_process_registered: ({ event, fork }) => {
      const message: BackgroundProcessMessage = {
        id: `background-process-${event.pid}`,
        type: 'background_process',
        pid: event.pid,
        command: event.command,
        status: 'running',
        stdout: event.initialStdout,
        stderr: event.initialStderr,
        exitCode: null,
        signal: null,
        timestamp: event.startedAt,
        updatedAt: event.timestamp,
      }

      const messages = insertBeforeQueuedMessages(
        fork.messages.filter(m => !(m.type === 'background_process' && m.pid === event.pid)),
        message
      )

      return {
        ...fork,
        messages
      }
    },

    background_process_output: ({ event, fork }) => {
      const existing = fork.messages.find(
        (m): m is BackgroundProcessMessage => m.type === 'background_process' && m.pid === event.pid
      )
      if (!existing) return fork

      const stdout = event.mode === 'tail'
        ? event.stdoutChunk
        : existing.stdout + event.stdoutChunk
      const stderr = event.mode === 'tail'
        ? event.stderrChunk
        : existing.stderr + event.stderrChunk

      const messages = updateMessageById<BackgroundProcessMessage>(
        fork.messages,
        existing.id,
        (msg) => ({
          ...msg,
          stdout,
          stderr,
          updatedAt: event.timestamp,
        })
      )

      return {
        ...fork,
        messages
      }
    },

    background_process_demoted: ({ fork }) => fork,

    background_process_exited: ({ event, fork }) => {
      const existing = fork.messages.find(
        (m): m is BackgroundProcessMessage => m.type === 'background_process' && m.pid === event.pid
      )
      if (!existing) return fork

      const messages = updateMessageById<BackgroundProcessMessage>(
        fork.messages,
        existing.id,
        (msg) => ({
          ...msg,
          status: event.status,
          exitCode: event.exitCode,
          signal: event.signal,
          stdout: msg.stdout,
          stderr: msg.stderr,
          updatedAt: event.timestamp,
        })
      )

      return {
        ...fork,
        messages
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

      // Finalize any still-open tool steps as interrupted before closing think block
      const interruptedState = finalizeOpenToolStepsAsInterrupted(fork)

      // Close think block and remove queued messages
      const closedState = closeThinkBlock(interruptedState, event.timestamp)
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

    agent_created: ({ event, fork }) => {
      const content = event.message.trim()
      if (!content) return fork
      if (event.forkId === null) return fork

      const { fork: newState, thinkBlockId } = ensureThinkBlock(fork, event.timestamp)
      return {
        ...newState,
        messages: addStepToThinkBlock(newState.messages, thinkBlockId, {
          id: generateId(),
          type: 'communication',
          direction: 'from_agent',
          agentId: event.agentId,
          agentName: event.name,
          agentRole: event.role,
          forkId: event.forkId,
          content,
          preview: toPreview(content),
          timestamp: event.timestamp,
          status: 'completed',
        })
      }
    },

    agent_killed: ({ fork }) => fork,
    subagent_user_killed: ({ fork }) => fork,
    subagent_idle_closed: ({ fork }) => fork,


  },

  signalHandlers: (on) => [
    // Insert inline fork activity block in parent's display when agent is created
    on(AgentStatusProjection.signals.agentCreated, ({ value, state }) => {
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
        createdAt: value.timestamp,
        activeSince: value.timestamp,
        accumulatedActiveMs: 0,
        resumeCount: 0,
        toolCounts: EMPTY_TOOL_COUNTS,
        timestamp: value.timestamp
      }

      let nextParentState: DisplayState = {
        ...parentState,
        messages: insertBeforeQueuedMessages(parentState.messages, msg)
      }

      if (parentForkId === null) {
        const withBlock = ensureThinkBlock(nextParentState, value.timestamp)
        const step: SubagentStartedStep = {
          id: generateId(),
          type: 'subagent_started',
          subagentType: value.type,
          subagentId: value.agentId,
          title: value.name,
          resumed: false,
        }
        nextParentState = {
          ...withBlock.fork,
          messages: addStepToThinkBlock(withBlock.fork.messages, withBlock.thinkBlockId, step),
        }
      }

      return {
        ...state,
        forks: new Map(state.forks).set(parentForkId, nextParentState)
      }
    }),

    // Update tool counts in parent's ForkActivityMessage when a tool step runs
    on(forkToolStepSignal, ({ value, state, read }) => {
      const { forkId, toolKey } = value
      if (forkId === null) return state  // root fork tools, no parent activity to update

      const agentState = read(AgentStatusProjection)
      const agent = getAgentByForkId(agentState, forkId)
      if (!agent) return state

      const parentState = state.forks.get(agent.parentForkId)
      if (!parentState) return state

      const msgIndex = findLastIndex(parentState.messages, (m: DisplayMessage) =>
        m.type === 'fork_activity' && m.forkId === forkId && m.status === 'running')
      if (msgIndex === -1) return state

      const msg = parentState.messages[msgIndex]
      if (msg?.type !== 'fork_activity') return state
      const newCounts = incrementToolCount(msg.toolCounts, toolKey)
      const newMessages = [...parentState.messages]
      newMessages[msgIndex] = { ...msg, toolCounts: newCounts }

      return {
        ...state,
        forks: new Map(state.forks).set(agent.parentForkId, { ...parentState, messages: newMessages })
      }
    }),

    // Mark fork activity as completed when agent becomes idle
    on(AgentStatusProjection.signals.agentBecameIdle, ({ value, state }) => {
      const { forkId, parentForkId } = value

      const parentState = state.forks.get(parentForkId)
      if (!parentState) return state

      const msgIndex = findLastIndex(parentState.messages, (m: DisplayMessage) =>
        m.type === 'fork_activity' && m.forkId === forkId && m.status === 'running')
      if (msgIndex === -1) return state

      const msg = parentState.messages[msgIndex]
      if (msg?.type !== 'fork_activity') return state
      const stintMs = Math.max(0, value.timestamp - msg.activeSince)
      const cumulativeTotalTimeMs = msg.accumulatedActiveMs + stintMs
      const newMessages = [...parentState.messages]
      newMessages[msgIndex] = {
        ...msg,
        status: 'completed',
        completedAt: value.timestamp,
        accumulatedActiveMs: cumulativeTotalTimeMs,
      }

      let nextParentState: DisplayState = { ...parentState, messages: newMessages }

      if (parentForkId === null && value.reason !== 'interrupt') {
        const withBlock = ensureThinkBlock(nextParentState, value.timestamp)
        const step: SubagentFinishedStep = {
          id: generateId(),
          type: 'subagent_finished',
          subagentType: value.type,
          subagentId: value.agentId,
          cumulativeTotalTimeMs,
          cumulativeTotalToolsUsed: totalToolsUsed(msg.toolCounts),
          resumed: (msg.resumeCount ?? 0) > 0,
        }
        nextParentState = {
          ...withBlock.fork,
          messages: addStepToThinkBlock(withBlock.fork.messages, withBlock.thinkBlockId, step),
        }
      }

      return {
        ...state,
        forks: new Map(state.forks).set(parentForkId, nextParentState)
      }
    }),

    on(AgentStatusProjection.signals.agentBecameWorking, ({ value, state }) => {
      const { forkId, parentForkId } = value
      const parentState = state.forks.get(parentForkId)
      if (!parentState) return state

      const msgIndex = findLastIndex(parentState.messages, (m: DisplayMessage) =>
        m.type === 'fork_activity' && m.forkId === forkId)
      if (msgIndex === -1) return state

      const message = parentState.messages[msgIndex]
      if (message?.type !== 'fork_activity') return state
      if (!message.completedAt) {
        // First transition into working (post-create): start active stint clock here.
        if ((message.resumeCount ?? 0) === 0) {
          const newMessages = [...parentState.messages]
          newMessages[msgIndex] = {
            ...message,
            activeSince: value.timestamp,
            timestamp: value.timestamp,
          }

          return {
            ...state,
            forks: new Map(state.forks).set(parentForkId, { ...parentState, messages: newMessages })
          }
        }
        return state
      }

      const nextResumeCount = (message.resumeCount ?? 0) + 1
      const moved = moveMessageToEndBeforeQueue<ForkActivityMessage>(parentState.messages, message.id, (msg) => ({
        ...msg,
        status: 'running',
        activeSince: value.timestamp,
        completedAt: undefined,
        resumeCount: nextResumeCount,
        timestamp: value.timestamp,
      }))

      let nextParentState: DisplayState = { ...parentState, messages: moved }

      if (parentForkId === null) {
        const withBlock = ensureThinkBlock(nextParentState, value.timestamp)
        const step: SubagentStartedStep = {
          id: generateId(),
          type: 'subagent_started',
          subagentType: value.type,
          subagentId: value.agentId,
          title: message.name,
          resumed: nextResumeCount > 0,
        }
        nextParentState = {
          ...withBlock.fork,
          messages: addStepToThinkBlock(withBlock.fork.messages, withBlock.thinkBlockId, step),
        }
      }

      return {
        ...state,
        forks: new Map(state.forks).set(parentForkId, nextParentState)
      }
    }),

    on(AgentStatusProjection.signals.agentKilled, ({ value, state }) => {
      const parentState = state.forks.get(value.parentForkId)
      if (!parentState) return state

      const messages = parentState.messages.filter((m) => !(m.type === 'fork_activity' && m.forkId === value.forkId))
      let nextParentState: DisplayState = { ...parentState, messages }

      const withBlock = ensureThinkBlock(nextParentState, value.timestamp)
      const step: SubagentKilledStep = {
        id: generateId(),
        type: 'subagent_killed',
        subagentType: value.type,
        subagentId: value.agentId,
        title: value.title,
      }
      nextParentState = {
        ...withBlock.fork,
        messages: addStepToThinkBlock(withBlock.fork.messages, withBlock.thinkBlockId, step),
      }

      return {
        ...state,
        forks: new Map(state.forks).set(value.parentForkId, nextParentState)
      }
    }),

    on(AgentStatusProjection.signals.subagentUserKilled, ({ value, state }) => {
      const parentState = state.forks.get(value.parentForkId)
      if (!parentState) return state

      const messages = parentState.messages.filter((m) => !(m.type === 'fork_activity' && m.forkId === value.forkId))
      let nextParentState: DisplayState = { ...parentState, messages }

      const withBlock = ensureThinkBlock(nextParentState, value.timestamp)
      const step: SubagentUserKilledStep = {
        id: generateId(),
        type: 'subagent_user_killed',
        subagentType: value.type,
        subagentId: value.agentId,
        title: value.title,
      }
      nextParentState = {
        ...withBlock.fork,
        messages: addStepToThinkBlock(withBlock.fork.messages, withBlock.thinkBlockId, step),
      }

      return {
        ...state,
        forks: new Map(state.forks).set(value.parentForkId, nextParentState)
      }
    }),

    on(AgentStatusProjection.signals.subagentIdleClosed, ({ value, state }) => {
      const parentState = state.forks.get(value.parentForkId)
      if (!parentState) return state

      const messages = parentState.messages.filter((m) => !(m.type === 'fork_activity' && m.forkId === value.forkId))
      return {
        ...state,
        forks: new Map(state.forks).set(value.parentForkId, { ...parentState, messages }),
      }
    }),

    on(AgentRoutingProjection.signals.communicationStreamStarted, ({ value, state, read }) => {
      const displayFork = state.forks.get(value.targetForkId)
      if (!displayFork) return state

      if (value.direction === 'from_agent') return state

      const agentState = read(AgentStatusProjection)
      const targetAgent = getAgentByForkId(agentState, value.targetForkId)
      const nextFork = upsertStreamingCommunicationStep(
        displayFork,
        value.streamId,
        {
          direction: value.direction,
          agentId: value.agentId,
          agentName: targetAgent?.name,
          agentRole: targetAgent?.role,
          forkId: value.targetForkId,
          timestamp: value.timestamp,
        },
        value.textDelta
      )

      return {
        ...state,
        forks: new Map(state.forks).set(value.targetForkId, nextFork)
      }
    }),

    on(AgentRoutingProjection.signals.communicationStreamChunk, ({ value, state, read }) => {
      const displayFork = state.forks.get(value.targetForkId)
      if (!displayFork) return state

      if (value.direction === 'from_agent') return state

      const agentState = read(AgentStatusProjection)
      const targetAgent = getAgentByForkId(agentState, value.targetForkId)
      const nextFork = upsertStreamingCommunicationStep(
        displayFork,
        value.streamId,
        {
          direction: value.direction,
          agentId: value.agentId,
          agentName: targetAgent?.name,
          agentRole: targetAgent?.role,
          forkId: value.targetForkId,
          timestamp: value.timestamp,
        },
        value.textDelta
      )

      return {
        ...state,
        forks: new Map(state.forks).set(value.targetForkId, nextFork)
      }
    }),

    on(AgentRoutingProjection.signals.communicationStreamCompleted, ({ value, state }) => {
      const displayFork = state.forks.get(value.targetForkId)
      if (!displayFork) return state
      if (value.direction === 'from_agent') return state

      return {
        ...state,
        forks: new Map(state.forks).set(
          value.targetForkId,
          finalizeCommunicationStreamInFork(displayFork, value.streamId)
        )
      }
    }),

    on(AgentRoutingProjection.signals.agentMessage, ({ value, state, read }) => {
      const displayFork = state.forks.get(value.targetForkId)
      if (!displayFork) return state
      const agentState = read(AgentStatusProjection)
      const workingState = read(WorkingStateProjection)
      const workingFork = workingState.forks.get(value.targetForkId)
      const targetAgent = getAgentByForkId(agentState, value.targetForkId)

      const pending = (workingFork?.pendingInboundCommunications ?? []).map((message): PendingInboundCommunicationDisplay => ({
        ...message,
        agentName: message.agentName ?? targetAgent?.name,
        agentRole: message.agentRole ?? targetAgent?.role,
      }))

      return {
        ...state,
        forks: new Map(state.forks).set(value.targetForkId, {
          ...displayFork,
          pendingInboundCommunications: pending,
        })
      }
    }),

    on(AgentRoutingProjection.signals.agentResponse, ({ value, state, read }) => {
      const displayFork = state.forks.get(value.targetForkId)
      if (!displayFork) return state
      const agentState = read(AgentStatusProjection)
      const workingState = read(WorkingStateProjection)
      const workingFork = workingState.forks.get(value.targetForkId)
      const targetAgent = value.targetForkId ? getAgentByForkId(agentState, value.targetForkId) : undefined

      const pending = (workingFork?.pendingInboundCommunications ?? []).map((message): PendingInboundCommunicationDisplay => ({
        ...message,
        agentName: message.agentName ?? targetAgent?.name,
        agentRole: message.agentRole ?? targetAgent?.role,
      }))

      return {
        ...state,
        forks: new Map(state.forks).set(value.targetForkId, {
          ...displayFork,
          pendingInboundCommunications: pending,
        })
      }
    }),

    on(WorkingStateProjection.signals.pendingInboundCommunicationsRead, ({ value, state, read }) => {
      const displayFork = state.forks.get(value.forkId)
      if (!displayFork) return state

      const agentState = read(AgentStatusProjection)
      let nextFork = { ...displayFork }

      if (value.forkId !== null) {
        for (const pending of value.messages.filter((message) => message.source === 'agent')) {
          const targetAgent = getAgentByForkId(agentState, value.forkId)
          const withBlock = ensureThinkBlock(nextFork, value.timestamp)
          nextFork = {
            ...withBlock.fork,
            messages: addStepToThinkBlock(withBlock.fork.messages, withBlock.thinkBlockId, {
              id: pending.id,
              type: 'communication',
              direction: 'from_agent',
              agentId: pending.agentId,
              agentName: pending.agentName ?? targetAgent?.name,
              agentRole: pending.agentRole ?? targetAgent?.role,
              forkId: pending.forkId,
              content: pending.content,
              preview: pending.preview,
              timestamp: pending.timestamp,
              status: 'completed',
            })
          }
        }
      }

      const pendingIds = new Set(value.messages.map(m => m.id))
      nextFork = {
        ...nextFork,
        pendingInboundCommunications: nextFork.pendingInboundCommunications.filter(m => !pendingIds.has(m.id))
      }

      return {
        ...state,
        forks: new Map(state.forks).set(value.forkId, nextFork)
      }
    }),


  ]
})

// =============================================================================
// Helpers
// =============================================================================

export function finalizeCommunicationStreamInFork(
  fork: DisplayState,
  streamId: string
): DisplayState {
  if (!fork.activeThinkBlockId) return fork
  const block = findThinkBlock(fork.messages, fork.activeThinkBlockId)
  if (!block) return fork
  const step = block.steps.find((s): s is CommunicationStep => s.type === 'communication' && s.streamId === streamId)
  if (!step) return fork

  return {
    ...fork,
    messages: updateStepInThinkBlock(
      fork.messages,
      fork.activeThinkBlockId,
      step.id,
      (s) => s.type === 'communication'
        ? { ...s, preview: toPreview(s.content), status: 'completed' }
        : s
    )
  }
}


