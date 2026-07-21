/**
 * Advisor window-to-prompt mapper.
 *
 * Builds a compact, transcript-like prompt for the advisor. The advisor
 * sees user intent, prior turns, and high-level work summaries without raw
 * tool output.
 *
 * User messages are wrapped in <user> tags.
 * Assistant turns ("Magnitude" turns) are wrapped in <magnitude> blocks
 * with <tools> counts and <message> tags.
 * If the turn contains a message_advisor tool call, a <message_advisor>
 * tag is included inside the <magnitude> block.
 * Advisor's own past responses are rendered as native AssistantMessage.
 * Synthetic (autopilot-originated) user messages are rendered as
 * AssistantMessage (the advisor sees its own autopilot output as its own
 * past turns).
 *
 * Autopilot toggle is injected as a transient <autopilot_toggled> tag
 * when the autopilot state differs from what the advisor last knew.
 */

import { Prompt, type Message as AiMessage } from '@magnitudedev/ai'
import { Option } from 'effect'
import type { ForkWindowState, CompletedTurn, WindowEntry } from '../types'
import type { TimelineEntry } from '../inbox/types'
import { renderTimeline } from '../inbox/render'
import { renderContextParts } from '../../content'
import {
  ensureTerminalUserMessage,
  systemEntryToMessages,
} from './shared'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function textMessage(text: string): AiMessage {
  return {
    _tag: 'UserMessage',
    parts: [{ _tag: 'TextPart', text }],
  }
}

// ---------------------------------------------------------------------------
// User message rendering
// ---------------------------------------------------------------------------

function renderUserMessage(text: string): string {
  return `<user>${text}</user>`
}

function renderTimelineEntryText(entry: TimelineEntry): string {
  return renderContextParts(renderTimeline({ timeline: [entry], timezone: null }), {
    includeImageData: false,
  }).filter(part => part._tag === 'TextPart').map(part => part.text).join('')
}

function contextEntryToAdvisorMessages(
  timeline: readonly TimelineEntry[],
): AiMessage[] {
  const userLines: string[] = []
  const autopilotLines: string[] = []

  for (const entry of timeline) {
    if (entry.kind === 'user_message') {
      const text = renderTimelineEntryText(entry)
      if (Option.getOrElse(entry.synthetic, () => false)) {
        autopilotLines.push(text)
      } else {
        userLines.push(renderUserMessage(text))
      }
    }
  }

  const result: AiMessage[] = []
  if (userLines.length > 0) {
    result.push(textMessage(userLines.join('\n')))
  }
  for (const text of autopilotLines) {
    result.push({
      _tag: 'AssistantMessage',
      reasoning: Option.none(),
      text: Option.some(text),
      toolCalls: Option.none(),
    })
  }
  return result
}

// ---------------------------------------------------------------------------
// Tool work summary
// ---------------------------------------------------------------------------

interface ToolWorkSummary {
  toolCounts: Map<string, number>
}

function createToolWorkSummary(): ToolWorkSummary {
  return { toolCounts: new Map() }
}

function hasToolWork(summary: ToolWorkSummary): boolean {
  return summary.toolCounts.size > 0
}

function countTool(summary: ToolWorkSummary, toolName: string): void {
  summary.toolCounts.set(toolName, (summary.toolCounts.get(toolName) ?? 0) + 1)
}

function addTurnWork(summary: ToolWorkSummary, turn: CompletedTurn): void {
  const toolCalls = Option.getOrElse(turn.assistant.toolCalls, () => [])
  const toolResults = turn.toolResults

  const resultCallIds = new Set<string>()
  for (const result of toolResults) {
    resultCallIds.add(result.toolCallId)
  }

  const matchedResultCallIds = new Set<string>()
  for (const call of toolCalls) {
    countTool(summary, call.name)
    if (resultCallIds.has(call.id)) {
      matchedResultCallIds.add(call.id)
    }
  }

  for (const result of toolResults) {
    if (!matchedResultCallIds.has(result.toolCallId)) {
      countTool(summary, result.toolName)
    }
  }
}

function renderToolWorkSummary(summary: ToolWorkSummary): string {
  const attrs = Array.from(summary.toolCounts, ([toolName, count]) => `${toolName}=${count}`)
  return `<tools ${attrs.join(' ')} />`
}

// ---------------------------------------------------------------------------
// Magnitude turn rendering
// ---------------------------------------------------------------------------

/**
 * Get the input text of the `message_advisor` tool call, if present.
 */
function getMessageAdvisorInput(turn: CompletedTurn): string | null {
  const toolCalls = Option.getOrElse(turn.assistant.toolCalls, () => [])
  const call = toolCalls.find(c => c.name === 'message_advisor')
  if (!call) return null
  const input = call.input as Record<string, unknown> | undefined
  if (!input) return null
  const text = input.message
  if (typeof text === 'string') return text.trim()
  return null
}

function buildMagnitudeBlock(turn: CompletedTurn, toolSummary: ToolWorkSummary): string {
  const parts: string[] = ['<magnitude>']

  if (hasToolWork(toolSummary)) {
    parts.push(renderToolWorkSummary(toolSummary))
  }

  const text = Option.getOrElse(turn.assistant.text, () => null)?.trim()
  if (text) {
    parts.push(`<message>${text}</message>`)
  }

  const advisorInput = getMessageAdvisorInput(turn)
  if (advisorInput) {
    parts.push(`<message_advisor>\n${advisorInput}\n</message_advisor>`)
  }

  parts.push('</magnitude>')
  return parts.join('\n')
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface AdvisorWindowPromptInput {
  readonly windowState: ForkWindowState
  readonly systemPrompt: string
  readonly autopilotEnabled: boolean
  readonly advisorLastAutopilotKnowledge: boolean | null
  readonly messageAdvisorText?: string | null
}

/**
 * Build the advisor prompt from the window state.
 *
 * Two-pass approach:
 * 1. Scan all entries to find which assistant_turns have content (text or
 *    message_advisor). These become the terminal messages for tool accumulation.
 * 2. Walk through entries, accumulating tools, and emit a <magnitude> block
 *    when we hit a content turn (with the accumulated tools inside it).
 */
export function advisorWindowToPrompt(input: AdvisorWindowPromptInput): Prompt {
  const { windowState, systemPrompt, autopilotEnabled, advisorLastAutopilotKnowledge } = input
  const messages: AiMessage[] = []
  let pendingWork = createToolWorkSummary()

  // First pass: determine which assistant_turn indices are "content" turns.
  const contentTurnIndices = new Set<number>()
  for (let i = 0; i < windowState.messages.length; i++) {
    const msg = windowState.messages[i]
    if (msg?.type === 'assistant_turn') {
      const turn = msg.turn
      const text = Option.getOrElse(turn.assistant.text, () => null)?.trim()
      const advisorInput = getMessageAdvisorInput(turn)
      if (text || advisorInput) {
        contentTurnIndices.add(i)
      }
    }
  }

  // Second pass: emit messages.
  for (let i = 0; i < windowState.messages.length; i++) {
    const msg = windowState.messages[i]
    switch (msg.type) {
      case 'compacted': {
        // Flush any accumulated tools before the compacted content.
        if (hasToolWork(pendingWork)) {
          messages.push(textMessage(renderToolWorkSummary(pendingWork)))
          pendingWork = createToolWorkSummary()
        }
        messages.push(...systemEntryToMessages(msg, false))
        break
      }

      case 'assistant_turn': {
        const turn = msg.turn
        addTurnWork(pendingWork, turn)

        if (contentTurnIndices.has(i)) {
          // This turn has content — emit a <magnitude> block with the
          // accumulated tools and the turn's content.
          messages.push(textMessage(buildMagnitudeBlock(turn, pendingWork)))
          pendingWork = createToolWorkSummary()
        }
        // If not a content turn, tools accumulate silently.
        break
      }

      case 'context': {
        const rendered = contextEntryToAdvisorMessages(msg.timeline)
        if (rendered.length > 0) {
          // Flush any accumulated tools before user messages.
          if (hasToolWork(pendingWork)) {
            messages.push(textMessage(renderToolWorkSummary(pendingWork)))
            pendingWork = createToolWorkSummary()
          }
          messages.push(...rendered)
        }
        break
      }

      case 'advisor_response': {
        // Flush any accumulated tools before the advisor response.
        if (hasToolWork(pendingWork)) {
          messages.push(textMessage(renderToolWorkSummary(pendingWork)))
          pendingWork = createToolWorkSummary()
        }
        messages.push({
          _tag: 'AssistantMessage',
          reasoning: Option.none(),
          text: Option.some(msg.content),
          toolCalls: Option.none(),
        })
        break
      }

      case 'session_context':
      case 'fork_context':
      case 'goal_injection':
      case 'observer_turn':
      case 'attempt_feedback':
        break
    }
  }

  // Flush any remaining tool work at the end.
  if (hasToolWork(pendingWork)) {
    messages.push(textMessage(renderToolWorkSummary(pendingWork)))
    pendingWork = createToolWorkSummary()
  }

  // TEMPORARILY DISABLED: autopilot context injection.
  // const shouldShowToggle = advisorLastAutopilotKnowledge !== autopilotEnabled
  const shouldShowToggle = false
  if (shouldShowToggle) {
    // Find the last <magnitude> UserMessage and insert the toggle before it.
    let lastMagnitudeIndex = -1
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i]
      if (msg?._tag === 'UserMessage' &&
          msg.parts.length === 1 &&
          msg.parts[0]?._tag === 'TextPart' &&
          msg.parts[0].text.startsWith('<magnitude>')) {
        lastMagnitudeIndex = i
        break
      }
    }

    if (lastMagnitudeIndex >= 0) {
      messages.splice(lastMagnitudeIndex, 0, textMessage(`<autopilot_toggled enabled="${autopilotEnabled}" />`))
    } else {
      // No magnitude block found — append before the terminal placeholder.
      messages.push(textMessage(`<autopilot_toggled enabled="${autopilotEnabled}" />`))
    }
  }

  // Inject the <message_advisor> tag for a manual message_advisor invocation.
  if (input.messageAdvisorText) {
    messages.push(textMessage(`<magnitude>\n<message_advisor>\n${input.messageAdvisorText}\n</message_advisor>\n</magnitude>`))
  }

  return Prompt.from({
    system: systemPrompt,
    messages: ensureTerminalUserMessage(messages),
  })
}
