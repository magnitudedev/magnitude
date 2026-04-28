import { Effect } from 'effect'
import type { ChatCompletionsRequest, ChatMessage, ChatTool, ChatToolCall } from '@magnitudedev/drivers'
import type { ToolDef } from '../../tools/tool-def'
import type { EncodeOptions } from '../../codec'
import { CodecEncodeError } from '../../codec'
import type {
  MemoryMessage,
  ContentPart,
  TurnPart,
  ResultEntry,
  TurnResultItem,
  TimelineEntry,
} from './memory-types'
import { asMemoryMessage, isToolError, isToolObservation } from './memory-types'

// =============================================================================
// Config
// =============================================================================

export interface EncodeConfig {
  readonly wireModelName:    string
  readonly defaultMaxTokens: number
  readonly supportsReasoning: boolean
  readonly supportsVision:   boolean
}

// =============================================================================
// ContentPart → wire
// =============================================================================

/**
 * Flatten ContentPart[] to either a plain string (text-only) or an array of
 * OpenAI-format content parts (when images are present or explicitly multimodal).
 */
function encodeContent(
  parts: readonly ContentPart[],
  supportsVision: boolean,
): string | readonly unknown[] {
  const hasImage = parts.some(p => p.type === 'image')
  if (!hasImage || !supportsVision) {
    // Concatenate text; drop images if vision not supported
    return parts
      .filter((p): p is Extract<ContentPart, { type: 'text' }> => p.type === 'text')
      .map(p => p.text)
      .join('\n')
  }
  // Multimodal array form
  return parts.map(p => {
    if (p.type === 'text') {
      return { type: 'text', text: p.text }
    }
    return {
      type:      'image_url',
      image_url: {
        url: `data:${p.mediaType};base64,${p.base64}`,
      },
    }
  })
}

// =============================================================================
// ToolDef → ChatTool
// =============================================================================

export function encodeToolDef(td: ToolDef): ChatTool {
  return {
    type:     'function',
    function: {
      name:        td.name,
      description: td.description,
      parameters:  td.parameters,
    },
  }
}

// =============================================================================
// Timeline → user message text
// =============================================================================

function renderTimelineEntry(entry: TimelineEntry): string {
  switch (entry.kind) {
    case 'user_message':
      return `<user_message>\n${entry.text}\n</user_message>`

    case 'parent_message':
      return `<parent_message>\n${entry.text}\n</parent_message>`

    case 'user_bash_command': {
      const parts: string[] = [
        `<bash_command cwd="${entry.cwd}" exit_code="${entry.exitCode}">`,
        `<command>${entry.command}</command>`,
      ]
      if (entry.stdout) parts.push(`<stdout>${entry.stdout}</stdout>`)
      if (entry.stderr) parts.push(`<stderr>${entry.stderr}</stderr>`)
      parts.push('</bash_command>')
      return parts.join('\n')
    }

    case 'agent_block':
      return `<agent_block id="${entry.agentId}" role="${entry.role}"/>`

    case 'observation': {
      // Handled separately as image parts below
      const obsParts = entry.parts as readonly ContentPart[]
      return obsParts
        .filter((p): p is Extract<ContentPart, { type: 'text' }> => p.type === 'text')
        .map(p => p.text)
        .join('\n')
    }

    default:
      // Generic fallback: render kind + any text field
      return `<${entry.kind}/>`
  }
}

/**
 * Collect observation image ContentParts from timeline entries.
 */
function collectObservationImages(entries: readonly TimelineEntry[]): ContentPart[] {
  const images: ContentPart[] = []
  for (const entry of entries) {
    if (entry.kind === 'observation') {
      const obsParts = entry.parts as readonly ContentPart[]
      for (const p of obsParts) {
        if (p.type === 'image') images.push(p)
      }
    }
  }
  return images
}

// =============================================================================
// ResultEntry → ChatMessage[]
// =============================================================================

function encodeResultEntries(
  entries: readonly ResultEntry[],
  systemNotes: string[],
): ChatMessage[] {
  const toolMessages: ChatMessage[] = []

  for (const entry of entries) {
    if (entry.kind === 'turn_results') {
      const items = entry.items as readonly TurnResultItem[]
      for (const item of items) {
        if (isToolObservation(item)) {
          // Tool result — pass ContentPart[] directly as content
          const content = encodeContent(item.content, true /* always try */)
          toolMessages.push({
            role:         'tool',
            tool_call_id: item.toolCallId,
            content,
          })
        } else if (isToolError(item)) {
          const msg = item.message ?? '<error>'
          toolMessages.push({
            role:         'tool',
            tool_call_id: item.toolCallId,
            content:      msg,
          })
        } else {
          // parse errors, message_ack, no_tools_or_messages, etc. → system note
          const k = item.kind as string
          const note = `[${k}]${(item as Record<string,unknown>)['message'] ? ': ' + (item as Record<string,unknown>)['message'] : ''}`
          systemNotes.push(note)
        }
      }
    } else if (entry.kind === 'interrupted') {
      systemNotes.push('[interrupted]')
    } else if (entry.kind === 'error') {
      const msg = (entry as Record<string,unknown>)['message'] as string | undefined
      systemNotes.push(`[error${msg ? ': ' + msg : ''}]`)
    }
    // noop, oneshot_liveness, yield_worker_retrigger — silently dropped
  }

  return toolMessages
}

// =============================================================================
// InboxMessage → ChatMessage[]
// =============================================================================

function encodeInbox(
  inbox: { results: readonly ResultEntry[]; timeline: readonly TimelineEntry[] },
  supportsVision: boolean,
): ChatMessage[] {
  const systemNotes: string[] = []
  const toolMessages = encodeResultEntries(inbox.results, systemNotes)

  // Build the trailing user message from timeline + system notes + images
  const timelineText = inbox.timeline.map(renderTimelineEntry).filter(Boolean).join('\n\n')
  const notesText    = systemNotes.join('\n')

  const textPieces: string[] = []
  if (timelineText) textPieces.push(timelineText)
  if (notesText)    textPieces.push(notesText)

  const trailingText = textPieces.join('\n\n')

  const observationImages = supportsVision ? collectObservationImages(inbox.timeline) : []

  // Only emit the trailing user message if there's content
  if (trailingText || observationImages.length > 0) {
    const userContent: unknown[] = []
    if (trailingText) userContent.push({ type: 'text', text: trailingText })
    for (const img of observationImages) {
      const imgPart = img as Extract<ContentPart, { type: 'image' }>
      userContent.push({
        type:      'image_url',
        image_url: { url: `data:${imgPart.mediaType};base64,${imgPart.base64}` },
      })
    }

    const trailingMessage: ChatMessage = {
      role:    'user',
      content: userContent.length === 1 && (userContent[0] as Record<string,unknown>)['type'] === 'text'
        ? (userContent[0] as Record<string,unknown>)['text'] as string
        : (userContent as readonly unknown[]),
    }
    return [...toolMessages, trailingMessage]
  }

  return toolMessages
}

// =============================================================================
// AssistantTurnMessage → ChatMessage
// =============================================================================

function encodeAssistantTurn(parts: readonly TurnPart[], supportsReasoning: boolean): ChatMessage {
  if (parts.length === 0) {
    return { role: 'assistant', content: '' }
  }

  const thoughtTexts: string[] = []
  const messageTexts: string[] = []
  const toolCalls: ChatToolCall[] = []

  for (const part of parts) {
    if (part.type === 'thought') {
      thoughtTexts.push(part.text)
    } else if (part.type === 'message') {
      messageTexts.push(part.text)
    } else if (part.type === 'tool_call') {
      toolCalls.push({
        id:       part.id,
        type:     'function',
        function: {
          name:      part.toolName,
          arguments: JSON.stringify(part.input),
        },
      })
    }
  }

  const content          = messageTexts.length > 0 ? messageTexts.join('\n\n') : undefined
  const reasoningContent = supportsReasoning && thoughtTexts.length > 0
    ? thoughtTexts.join('\n\n')
    : undefined

  return {
    role:              'assistant',
    ...(content           !== undefined ? { content }                              : {}),
    ...(reasoningContent  !== undefined ? { reasoning_content: reasoningContent }  : {}),
    ...(toolCalls.length  > 0           ? { tool_calls: toolCalls }                : {}),
  } as ChatMessage
}

// =============================================================================
// Single MemoryMessage → ChatMessage[]
// =============================================================================

function encodeMessage(msg: MemoryMessage, config: EncodeConfig): ChatMessage[] {
  switch (msg.type) {
    case 'session_context':
    case 'fork_context':
      return [{
        role:    'system',
        content: encodeContent(msg.content, false) as string,
        // System messages are always text-only (providers don't support images in system)
      }]

    case 'compacted':
      return [{
        role:    'system',
        content: '<compacted>\n' + (encodeContent(msg.content, false) as string) + '\n</compacted>',
      }]

    case 'assistant_turn':
      return [encodeAssistantTurn(msg.parts, config.supportsReasoning)]

    case 'inbox':
      return encodeInbox(msg, config.supportsVision)
  }
}

// =============================================================================
// Top-level encode
// =============================================================================

export function encode(
  memory:  readonly unknown[],
  tools:   readonly ToolDef[],
  options: EncodeOptions,
  config:  EncodeConfig,
): Effect.Effect<ChatCompletionsRequest, CodecEncodeError> {
  return Effect.try({
    try: () => {
      const messages: ChatMessage[] = []

      for (const raw of memory) {
        const msg = asMemoryMessage(raw)
        if (msg === null) {
          // Unknown memory entry — skip silently (forward-compat)
          continue
        }
        const chatMsgs = encodeMessage(msg, config)
        messages.push(...chatMsgs)
      }

      const chatTools = tools.length > 0 ? tools.map(encodeToolDef) : undefined

      const request: ChatCompletionsRequest = {
        model:          config.wireModelName,
        messages,
        ...(chatTools ? { tools: chatTools } : {}),
        max_tokens:     options.maxTokens ?? config.defaultMaxTokens,
        ...(options.stopSequences && options.stopSequences.length > 0
          ? { stop: options.stopSequences as string[] }
          : {}),
        stream:         true,
        stream_options: { include_usage: true },
      }

      return request
    },
    catch: (err) =>
      new CodecEncodeError({
        reason:  String(err),
        context: { memory, tools, options },
      }),
  })
}
