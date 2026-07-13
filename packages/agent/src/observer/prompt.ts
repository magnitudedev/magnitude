/**
 * Observer prompt builder.
 *
 * Maps a ForkWindowState into the compact prompt the observer LLM sees.
 * The observed agent's turns are rendered as an observer-specific transcript.
 * The observer's own prior turns are rendered as real AssistantMessage + ToolResultMessage
 * pairs so the model sees its own past tool calls as actual tool history.
 */

import { Prompt, type Message as AiMessage, createToolCallId } from '@magnitudedev/ai'
import type { ProviderToolCallId } from '@magnitudedev/ai'
import { Option } from 'effect'
import { ContentBuilder } from '../content'
import { observerPrompt } from '@magnitudedev/roles'
import type { ForkWindowState, WindowEntry, CompletedTurn } from '../window/types'
import type { TimelineEntry } from '../window/inbox/types'
import {
  ensureTerminalUserMessage,
  systemEntryToMessages,
} from '../window/render/shared'
import { createTimeBoundaryEmitter, minuteKey } from '../window/render/time-boundaries'
import { renderTimelineUserMessageParts } from '../window/render/user-message-parts'
import { estimateText, renderXmlBodyValue, renderXmlBodyValues, type JsonValue } from '../truncation'
import { renderFeedbackText } from '../prompts/feedback-text'

import type { ObserverJustification } from './justifications'

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
// Context timeline rendering
// ---------------------------------------------------------------------------

function hasRenderablePart(parts: readonly { readonly _tag: string; readonly text?: string }[]): boolean {
  return parts.some((part) => part._tag !== 'TextPart' || (part.text?.trim().length ?? 0) > 0)
}

function pushParts(builder: ContentBuilder, parts: readonly ReturnType<typeof renderTimelineUserMessageParts>[number][]): void {
  for (const part of parts) {
    if (part._tag === 'TextPart') builder.pushText(part.text)
    else builder.pushPart(part)
  }
}

function isSameMinuteUserMessage(
  entry: TimelineEntry,
  first: Extract<TimelineEntry, { kind: 'user_message' }>,
  timezone: string | null,
): boolean {
  return entry.kind === 'user_message' && minuteKey(entry.timestamp, timezone) === minuteKey(first.timestamp, timezone)
}

function isRenderedNonUserTimelineEntry(entry: TimelineEntry): boolean {
  return entry.kind === 'user_bash_command' || entry.kind === 'escalation'
}

function renderSingleObserverUserMessage(
  builder: ContentBuilder,
  entry: Extract<TimelineEntry, { kind: 'user_message' }>,
): void {
  builder.pushText('\n')
  pushParts(builder, renderTimelineUserMessageParts(entry, {
    open: '<user>\n',
    close: '\n</user>',
    attachmentsInsideWrapper: true,
  }))
}


function renderObserverTimeline(timeline: readonly TimelineEntry[], timezone: string | null): readonly ReturnType<ContentBuilder['build']>[number][] {
  const builder = new ContentBuilder()
  const timeBoundaries = createTimeBoundaryEmitter(timezone)

  const emitTimeBoundary = (timestamp: number) => {
    const marker = timeBoundaries.next(timestamp)
    if (!marker) return
    builder.pushText(`${builder.hasContent() ? '\n\n' : ''}${marker}`)
  }

  for (let i = 0; i < timeline.length; i++) {
    const entry = timeline[i]!

    switch (entry.kind) {
      case 'turn_start': {
        emitTimeBoundary(entry.timestamp)
        break
      }

      case 'user_message': {
        emitTimeBoundary(entry.timestamp)
        const run: Extract<TimelineEntry, { kind: 'user_message' }>[] = [entry]
        let j = i + 1
        let lastConsumed = i
        while (j < timeline.length) {
          const next = timeline[j]!
          if (next.kind === 'user_message') {
            if (isSameMinuteUserMessage(next, entry, timezone)) {
              run.push(next)
              lastConsumed = j
              j++
              continue
            }
            break
          }
          if (isRenderedNonUserTimelineEntry(next)) break
          j++
        }
        if (run.length === 1) renderSingleObserverUserMessage(builder, entry)
        else for (const msg of run) renderSingleObserverUserMessage(builder, msg)
        i = lastConsumed
        break
      }

      case 'user_bash_command': {
        emitTimeBoundary(entry.timestamp)
        builder.pushText(`\n<user_bash_command cwd="${entry.cwd}" exit_code="${entry.exitCode}">\n<command>${entry.command}</command>\n<stdout>${entry.stdout}</stdout>\n<stderr>${entry.stderr}</stderr>\n</user_bash_command>`)
        break
      }

      case 'escalation': {
        // Worker escalation — not rendered in observer context timeline.
        // The observer sees its own past evaluations as assistant+tool result pairs.
        break
      }

      case 'turn_end':
      case 'observation':
      case 'agent_block':
      case 'coordinator_message':
      case 'user_to_agent':
      case 'worker_user_killed':
      case 'lifecycle_hook':
      case 'task_start_hook':
      case 'task_idle_hook':
      case 'task_complete_hook':
      case 'task_tree_dirty':
      case 'task_tree_view':
      case 'task_update':
      case 'task_reassigned':
      case 'detached_process_exited':
        break
    }
  }

  return builder.build()
}

function contextEntryToObserverMessages(timeline: readonly TimelineEntry[], timezone: string | null): AiMessage[] {
  const parts = renderObserverTimeline(timeline, timezone)
  if (!hasRenderablePart(parts)) return []
  return [{ _tag: 'UserMessage', parts }]
}

// ---------------------------------------------------------------------------
// Observed agent turn rendering
// ---------------------------------------------------------------------------

const PARAMS_BUDGET_TOKENS = 300
const RESULT_BUDGET_TOKENS = 500
const TOOLS_BUDGET_TOKENS = 5000

function toXmlTagName(value: string): string {
  const cleaned = value.replace(/[^A-Za-z0-9_.-]/g, '_')
  if (/^[A-Za-z_]/.test(cleaned)) return cleaned || 'value'
  return `_${cleaned}`
}

function renderToolParams(input: JsonValue, budgetTokens: number): string {
  if (typeof input === 'object' && input !== null && !Array.isArray(input)) {
    const entries = Object.entries(input)
    if (entries.length === 0) return '<params></params>'
    const renderedValues = renderXmlBodyValues(entries.map(([, nested]) => nested), budgetTokens)
    const lines = entries.map(([key], index) => {
      const tag = toXmlTagName(key)
      return `<${tag}>${renderedValues[index]}</${tag}>`
    })
    return `<params>\n${lines.join('\n')}\n</params>`
  }

  return `<params>\n<value>${renderXmlBodyValue(input, budgetTokens)}</value>\n</params>`
}

function renderToolOutputValue(value: unknown, budgetTokens: number): string {
  return renderXmlBodyValue(value as JsonValue, budgetTokens)
}

function renderToolResult(result: CompletedTurn['toolResults'][number]['result'] | null, budgetTokens: number): string {
  if (!result || result._tag === 'Interrupted') return '<interrupted/>'

  if (result._tag === 'Success') {
    return `<result>${renderToolOutputValue(result.output, budgetTokens)}</result>`
  }
  if (result._tag === 'Error') {
    return `<error>${renderToolOutputValue(result.error, budgetTokens)}</error>`
  }
  return `<error>${renderToolOutputValue(result, budgetTokens)}</error>`
}

function renderTurnToolsWithBudgets(turn: CompletedTurn, paramsBudget: number, resultBudget: number): string | null {
  const toolCalls = Option.getOrElse(turn.assistant.toolCalls, () => [])
  if (toolCalls.length === 0) return null

  const resultsByCallId = new Map(turn.toolResults.map(result => [result.toolCallId, result.result]))
  const blocks = toolCalls.map((call) => {
    const tag = toXmlTagName(call.name)
    const params = renderToolParams(call.input, paramsBudget)
    const result = renderToolResult(resultsByCallId.get(call.id) ?? null, resultBudget)
    return `<${tag}>\n${params}\n${result}\n</${tag}>`
  })

  return `<tools>\n${blocks.join('\n')}\n</tools>`
}

function renderTurnTools(turn: CompletedTurn): string | null {
  const firstPass = renderTurnToolsWithBudgets(turn, PARAMS_BUDGET_TOKENS, RESULT_BUDGET_TOKENS)
  if (!firstPass) return null

  const estimatedTokens = estimateText(firstPass)
  if (estimatedTokens <= TOOLS_BUDGET_TOKENS) return firstPass

  const scale = Math.max(0.1, TOOLS_BUDGET_TOKENS / estimatedTokens)
  const paramsBudget = Math.max(40, Math.floor(PARAMS_BUDGET_TOKENS * scale))
  const resultBudget = Math.max(80, Math.floor(RESULT_BUDGET_TOKENS * scale))
  return renderTurnToolsWithBudgets(turn, paramsBudget, resultBudget)
}

function assistantTurnToObserverMessage(turn: CompletedTurn): AiMessage | null {
  const parts: string[] = ['<magnitude>']

  const reasoning = Option.getOrElse(turn.assistant.reasoning, () => null)
  if (reasoning && reasoning.length > 0) {
    parts.push(`<thoughts>\n${reasoning}\n</thoughts>`)
  }

  const tools = renderTurnTools(turn)
  if (tools) parts.push(tools)

  const text = Option.getOrElse(turn.assistant.text, () => null)?.trim()
  if (text) {
    parts.push(`<message>\n${text}\n</message>`)
  }

  const feedback = renderFeedbackText(turn.feedback)
  if (feedback) {
    parts.push(`<feedback from="user">\n${feedback}\n</feedback>`)
  }

  parts.push('</magnitude>')
  if (parts.length <= 2) return null
  return textMessage(parts.join('\n'))
}

// ---------------------------------------------------------------------------
// Prior observer turn rendering (real tool call history)
// ---------------------------------------------------------------------------

/**
 * Render a prior observer_turn as a real AssistantMessage + ToolResultMessage pair.
 * The observer sees its own pass or escalate calls as history.
 */
function observerTurnToMessages(entry: Extract<WindowEntry, { type: 'observer_turn' }>): readonly [AiMessage, AiMessage] {
  const toolCallId = createToolCallId()
  const providerToolCallId = toolCallId as unknown as ProviderToolCallId

  if (entry.escalate) {
    const assistantMsg: AiMessage = {
      _tag: 'AssistantMessage',
      reasoning: Option.none(),
      text: Option.none(),
      toolCalls: Option.some([{
        _tag: 'ToolCallPart',
        id: toolCallId,
        providerToolCallId,
        name: 'escalate',
        input: { justification: entry.justification },
      }]),
    }

    const toolResultMsg: AiMessage = {
      _tag: 'ToolResultMessage',
      toolCallId,
      providerToolCallId,
      toolName: 'escalate',
      parts: [{ _tag: 'TextPart', text: '{"status":"ok"}' }],
    }

    return [assistantMsg, toolResultMsg]
  }

  const assistantMsg: AiMessage = {
    _tag: 'AssistantMessage',
    reasoning: Option.none(),
    text: Option.none(),
    toolCalls: Option.some([{
      _tag: 'ToolCallPart',
      id: toolCallId,
      providerToolCallId,
      name: 'pass',
      input: {},
    }]),
  }

  const toolResultMsg: AiMessage = {
    _tag: 'ToolResultMessage',
    toolCallId,
    providerToolCallId,
    toolName: 'pass',
    parts: [{ _tag: 'TextPart', text: '{"status":"ok"}' }],
  }

  return [assistantMsg, toolResultMsg]
}

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

/** Get the observer system prompt. */
export function getObserverSystemPrompt(): string {
  return observerPrompt.render()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface ObserverWindowPromptInput {
  readonly windowState: ForkWindowState
  readonly systemPrompt: string
  readonly observedForkId: string | null
  readonly timezone: string | null
}

export function observerWindowToPrompt(input: ObserverWindowPromptInput): Prompt {
  const { windowState, systemPrompt, timezone } = input
  const messages: AiMessage[] = []

  for (const msg of windowState.messages) {
    switch (msg.type) {
      case 'compacted': {
        messages.push(...systemEntryToMessages(msg))
        break
      }

      case 'observer_turn': {
        const [assistantMsg, toolResultMsg] = observerTurnToMessages(msg)
        messages.push(assistantMsg, toolResultMsg)
        break
      }

      case 'assistant_turn': {
        const rendered = assistantTurnToObserverMessage(msg.turn)
        if (rendered) messages.push(rendered)
        break
      }

      case 'context': {
        const rendered = contextEntryToObserverMessages(msg.timeline, timezone)
        if (rendered.length > 0) {
          messages.push(...rendered)
        }
        break
      }

      case 'session_context':
      case 'fork_context':
        break
    }
  }

  return Prompt.from({
    system: systemPrompt,
    messages: ensureTerminalUserMessage(messages),
  })
}
