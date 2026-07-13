/**
 * Full window-to-prompt mapper for the leader agent.
 *
 * Preserves structured assistant turn information (reasoning, tool calls)
 * and converts semantic tool results into native ToolResultMessages for the model API.
 *
 * Composes shared helpers from ./shared.ts and formatters from ./formatters.ts.
 */

import { Prompt, type Message as AiMessage, type TerminalMessages } from '@magnitudedev/ai'
import type { ForkWindowState } from '../types'
import type { ToolResultEntry } from '@magnitudedev/harness'
import type { ToolResultFormatter } from '@magnitudedev/harness'
import { createTruncatingFormatter } from './formatters'
import {
  systemEntryToMessages,
  contextEntryToMessages,
  renderFeedback,
  ensureTerminalUserMessage,
} from './shared'

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

function textMessage(text: string): AiMessage {
  return {
    _tag: 'UserMessage',
    parts: [{ _tag: 'TextPart', text }],
  }
}

/**
 * Coalesce adjacent UserMessages into a single UserMessage.
 * This avoids spurious UserMessage boundaries from consecutive context entries
 * (which each render as their own UserMessage) without mutating cached entries.
 */
function coalesceAdjacentUserMessages(messages: readonly AiMessage[]): AiMessage[] {
  const result: AiMessage[] = []
  for (const msg of messages) {
    const last = result[result.length - 1]
    if (last && last._tag === 'UserMessage' && msg._tag === 'UserMessage') {
      result[result.length - 1] = {
        _tag: 'UserMessage',
        parts: [...last.parts, ...msg.parts],
      }
    } else {
      result.push(msg)
    }
  }
  return result
}

// ---------------------------------------------------------------------------
// ToolResultEntry → ToolResultMessage conversion
// ---------------------------------------------------------------------------

function toolResultEntryToMessage(
  entry: ToolResultEntry,
  formatter: ToolResultFormatter,
  turnId: string,
): AiMessage {
  const parts = formatter(entry)

  return {
    _tag: 'ToolResultMessage',
    toolCallId: entry.toolCallId,
    providerToolCallId: entry.providerToolCallId,
    toolName: entry.toolName,
    parts,
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Convert window state into an ai Prompt.
 *
 * Preserves structured assistant turn information (reasoning, tool calls)
 * and converts semantic tool results into native ToolResultMessages for the model API.
 */
export interface LeaderWindowPromptInput {
  readonly windowState: ForkWindowState
  readonly systemPrompt: string
  readonly timezone: string | null
  readonly formatter: ToolResultFormatter
  readonly autopilotEnabled: boolean
  readonly leaderLastAutopilotKnowledge: boolean | null
}

export function windowToPrompt(input: LeaderWindowPromptInput): Prompt {
  const {
    windowState,
    systemPrompt,
    timezone,
    formatter,
    autopilotEnabled,
    leaderLastAutopilotKnowledge,
  } = input
  const messages: AiMessage[] = []

  for (const msg of windowState.messages) {
    switch (msg.type) {
      case 'observer_turn':
        // Observer turns are only visible to the observer prompt builder,
        // not to the agent's own context window.
        break
      case 'session_context':
      case 'fork_context':
      case 'goal_injection':
      case 'compacted': {
        messages.push(...systemEntryToMessages(msg))
        break
      }

      case 'assistant_turn': {
        const { turn } = msg
        messages.push(turn.assistant)
        const turnFormatter = createTruncatingFormatter(formatter, turn.turnId)
        for (const entry of turn.toolResults) {
          messages.push(toolResultEntryToMessage(entry, turnFormatter, turn.turnId))
        }
        const feedbackParts = renderFeedback(turn.feedback)
        if (feedbackParts.length > 0) {
          messages.push({
            _tag: 'UserMessage',
            parts: feedbackParts,
          })
        }
        break
      }

      case 'advisor_response': {
        // Leader sees advisor responses as tool results within its own turns.
        break
      }

      case 'context': {
        messages.push(...contextEntryToMessages(msg, timezone))
        break
      }
    }
  }

  // TEMPORARILY DISABLED: autopilot context injection.
  // const shouldShowToggle = leaderLastAutopilotKnowledge !== autopilotEnabled
  const shouldShowToggle = false
  if (shouldShowToggle) {
    // Find the last assistant turn and insert the toggle before it.
    let lastAssistantIndex = -1
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i]
      if (msg._tag === 'AssistantMessage') {
        lastAssistantIndex = i
        break
      }
    }

    if (lastAssistantIndex >= 0) {
      messages.splice(lastAssistantIndex, 0, textMessage(`<autopilot_toggled enabled="${autopilotEnabled}" />`))
    } else {
      // No assistant turn found — append before the terminal placeholder.
      messages.push(textMessage(`<autopilot_toggled enabled="${autopilotEnabled}" />`))
    }
  }

  const coalesced = coalesceAdjacentUserMessages(messages)
  const terminal = ensureTerminalUserMessage(coalesced, '(continue)')

  return Prompt.from({
    system: systemPrompt,
    messages: terminal,
  })
}
