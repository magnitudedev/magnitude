/**
 * Shared helpers for converting WindowEntry → ai Messages.
 *
 * Used by both `full` (leader) and `autopilot` window-to-prompt mappers.
 */

import type { Message as AiMessage, TerminalMessages, UserPart } from '@magnitudedev/ai'
import { Option } from 'effect'
import type { WindowEntry, CompletedTurn, TurnFeedback } from '../types'
import type { TimelineEntry } from '../inbox/types'
import { renderTimeline } from '../inbox/render'
import { renderFeedbackText } from '../../prompts/feedback-text'

// ---------------------------------------------------------------------------
// system/fork/compacted → UserMessage
// ---------------------------------------------------------------------------

export function systemEntryToMessages(
  entry: Extract<WindowEntry, { type: 'session_context' | 'fork_context' | 'compacted' | 'goal_injection' }>,
): readonly [AiMessage] {
  return [{
    _tag: 'UserMessage',
    parts: entry.content,
  }]
}

// ---------------------------------------------------------------------------
// context → timeline UserMessage
// ---------------------------------------------------------------------------

export function contextEntryToMessages(
  entry: Extract<WindowEntry, { type: 'context' }>,
  timezone: string | null,
): readonly AiMessage[] {
  const parts = renderTimeline({
    timeline: entry.timeline,
    timezone,
  })

  const hasContent = parts.some(p => {
    if (p._tag === 'TextPart') return p.text.trim().length > 0
    if (p._tag === 'ImagePart') return true
    return false
  })

  if (!hasContent) return []

  return [{
    _tag: 'UserMessage',
    parts,
  }]
}

// ---------------------------------------------------------------------------
// assistant_turn → prose-only AssistantMessage
// ---------------------------------------------------------------------------

export function assistantTurnProseOnly(
  entry: Extract<WindowEntry, { type: 'assistant_turn' }>,
): readonly [AiMessage] {
  return [{
    _tag: 'AssistantMessage',
    reasoning: Option.none(),
    reasoningDetails: [],
    text: entry.turn.assistant.text,
    toolCalls: Option.none(),
  }]
}

// ---------------------------------------------------------------------------
// TurnFeedback → UserMessage parts
// ---------------------------------------------------------------------------

export function renderFeedback(feedback: readonly TurnFeedback[]): UserPart[] {
  const text = renderFeedbackText(feedback)
  if (!text) return []
  return [{ _tag: 'TextPart', text }]
}

// ---------------------------------------------------------------------------
// Timeline filtering
// ---------------------------------------------------------------------------

const AUTOPILOT_TIMELINE_KINDS = new Set<TimelineEntry['kind']>([
  'user_message',
  'observation',
  'user_bash_command',
])

export function filteredAutopilotTimeline(
  timeline: readonly TimelineEntry[],
): readonly TimelineEntry[] {
  return timeline.filter(e => AUTOPILOT_TIMELINE_KINDS.has(e.kind))
}

// ---------------------------------------------------------------------------
// Terminal message constraint
// ---------------------------------------------------------------------------

export function ensureTerminalUserMessage(
  messages: AiMessage[],
  placeholder: string = '(continue)',
): TerminalMessages {
  const result = [...messages]
  const last = result[result.length - 1]
  if (!last || last._tag === 'AssistantMessage') {
    result.push({
      _tag: 'UserMessage',
      parts: [{ _tag: 'TextPart', text: placeholder }],
    })
  }
  return result as unknown as TerminalMessages
}
