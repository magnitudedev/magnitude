import { Option } from 'effect'
import type {
  DisplayMessage,
  DisplayTimelineEntry,
  DisplayTimelinePresentation,
  DisplayTimelinePresentationMode,
  DisplayTimelineStatusSlot,
  DisplayTimelineWindowInfo,
  ToolMessage,
  ToolStepPresentation,
} from '@magnitudedev/protocol'
import {
  getToolSummaryFamily,
  HIDDEN_TOOL_KEYS,
  presentToolSummary,
} from './tool-presentation'

export type TimelineScope = 'root' | 'fork'

export interface BuildDisplayTimelinePresentationOptions {
  readonly scope: TimelineScope
  readonly mode: DisplayTimelinePresentationMode
  readonly timelineMode: 'idle' | 'streaming'
  readonly streamingMessageId: string | null
  readonly messages: readonly DisplayMessage[]
  readonly window: DisplayTimelineWindowInfo
}

const DATA_ONLY_MESSAGE_TYPES: ReadonlySet<DisplayMessage['type']> = new Set([
  'fork_activity',
  'fork_result',
  'worker_resumed',
  'worker_finished',
  'worker_killed',
  'worker_user_killed',
])

const ROOT_ONLY_HIDDEN_MESSAGE_TYPES: ReadonlySet<DisplayMessage['type']> = new Set([
  'agent_communication',
])

const DEFAULT_HIDDEN_MESSAGE_TYPES: ReadonlySet<DisplayMessage['type']> = new Set([
  'thinking',
  'status_indicator',
])

function timelineRole(message: DisplayMessage): 'user' | 'assistant' | 'system' | 'agent' {
  switch (message.type) {
    case 'user_message':
    case 'queued_user_message':
    case 'user_bash_command':
      return 'user'
    case 'assistant_message':
    case 'thinking':
      return 'assistant'
    case 'agent_communication':
      return 'agent'
    default:
      return 'system'
  }
}

function isVisibleMessage(
  message: DisplayMessage,
  scope: TimelineScope,
  mode: DisplayTimelinePresentationMode,
): boolean {
  if (DATA_ONLY_MESSAGE_TYPES.has(message.type)) return false
  if (scope === 'root' && ROOT_ONLY_HIDDEN_MESSAGE_TYPES.has(message.type)) return false
  if (mode === 'default' && DEFAULT_HIDDEN_MESSAGE_TYPES.has(message.type)) return false
  if (message.type === 'tool') {
    if (HIDDEN_TOOL_KEYS.has(message.toolKey)) return false
    if (mode === 'default' && message.toolKey === 'spawnWorker') return false
  }
  return true
}

function statusSlotFor(
  messages: readonly DisplayMessage[],
  options: BuildDisplayTimelinePresentationOptions,
): DisplayTimelineStatusSlot {
  if (options.timelineMode !== 'idle') return { kind: 'none' }
  if (options.window.end !== options.window.totalCount) return { kind: 'none' }
  const tail = messages[messages.length - 1]
  if (tail?.type !== 'interrupted') return { kind: 'none' }
  return {
    kind: 'interrupted',
    messageId: tail.id,
    context: tail.context,
    allKilled: Option.isSome(tail.allKilled) && tail.allKilled.value === true,
  }
}

function toolStepFromMessage(message: ToolMessage): ToolStepPresentation | null {
  return message.presentation._tag === 'Some' ? message.presentation.value : null
}

export function buildDisplayTimelinePresentation(
  options: BuildDisplayTimelinePresentationOptions,
): DisplayTimelinePresentation {
  const statusSlot = statusSlotFor(options.messages, options)
  const entries: DisplayTimelineEntry[] = []
  let index = 0

  while (index < options.messages.length) {
    const message = options.messages[index]
    if (!isVisibleMessage(message, options.scope, options.mode)) {
      index++
      continue
    }

    if (statusSlot.kind === 'interrupted' && message.id === statusSlot.messageId) {
      index++
      continue
    }

    if (message.type === 'tool') {
      const family = getToolSummaryFamily(message.toolKey)
      if (family === null) {
        const step = toolStepFromMessage(message)
        if (step === null) {
          index++
          continue
        }
        entries.push({
          kind: 'tool_step',
          id: `tool-step:${message.id}`,
          timestamp: message.timestamp,
          messageId: message.id,
          step,
        })
        index++
        continue
      }

      const toolMessages: ToolMessage[] = [message]
      let nextIndex = index + 1

      while (nextIndex < options.messages.length) {
        const next = options.messages[nextIndex]
        if (!isVisibleMessage(next, options.scope, options.mode)) {
          nextIndex++
          continue
        }
        if (next.type !== 'tool') break
        const nextFamily = getToolSummaryFamily(next.toolKey)
        if (nextFamily !== family) break
        toolMessages.push(next)
        nextIndex++
      }

      const presentedToolMessages = toolMessages.flatMap((toolMessage) => {
        const step = toolStepFromMessage(toolMessage)
        return step === null ? [] : [{ message: toolMessage, step }]
      })
      if (presentedToolMessages.length === 0) {
        index = nextIndex
        continue
      }
      const presentations = presentedToolMessages.map((item) => item.step)

      entries.push({
        kind: 'tool_summary',
        id: `tool-summary:${presentedToolMessages[0]?.message.id ?? index}`,
        timestamp: message.timestamp,
        messageIds: presentedToolMessages.map((item) => item.message.id),
        summary: presentToolSummary(family, presentations),
      })
      index = nextIndex
      continue
    }

    entries.push({
      kind: 'message',
      id: `message:${message.id}`,
      messageId: message.id,
      timestamp: message.timestamp,
      role: timelineRole(message),
      streaming:
        options.timelineMode === 'streaming' &&
        message.type === 'assistant_message' &&
        options.streamingMessageId === message.id,
      interrupted: message.type === 'interrupted',
      nextMessageInterrupted:
        options.messages[index + 1]?.type === 'interrupted' &&
        !(statusSlot.kind === 'interrupted' && options.messages[index + 1]?.id === statusSlot.messageId),
    })
    index++
  }

  return {
    mode: options.mode,
    entries,
    statusSlot,
  }
}
