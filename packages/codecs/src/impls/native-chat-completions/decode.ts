import { Stream } from 'effect'
import type { ChatCompletionsStreamChunk, DriverError } from '@magnitudedev/drivers'
import type { ResponseStreamEvent, ResponseUsage } from '../../events/turn-part-event'
import { CodecDecodeError } from '../../codec'
import { newToolCallId } from '../../memory/ids'
import { createStreamingJsonParser } from './jsonish'
import type { StreamingJsonParser, ParsedValue } from './jsonish'

function toResponseUsage(usage: NonNullable<ChatCompletionsStreamChunk['usage']>): ResponseUsage {
  return {
    inputTokens:     usage.prompt_tokens,
    outputTokens:    usage.completion_tokens,
    cacheReadTokens: usage.prompt_tokens_details?.cached_tokens ?? null,
    cacheWriteTokens: null,
  }
}

// =============================================================================
// Field diffing state
// =============================================================================

interface FieldState {
  seenText: string
  complete: boolean
}

interface ToolCallState {
  readonly toolCallId: string
  readonly toolName: string
  readonly parser: StreamingJsonParser
  readonly snapshot: Map<string, FieldState>
}

// =============================================================================
// Decoder state (immutable — rebuilt on each processChunk call)
// =============================================================================

export interface DecoderState {
  readonly ordinal:        number
  readonly thoughtOpen:    boolean
  readonly messageOpen:    boolean
  // Map<index, ToolCallState> — rebuilt (not mutated) per step
  readonly openToolCalls:  ReadonlyMap<number, ToolCallState>
}

export const initialDecoderState: DecoderState = {
  ordinal:       0,
  thoughtOpen:   false,
  messageOpen:   false,
  openToolCalls: new Map(),
}

// =============================================================================
// Helpers
// =============================================================================

/** Attempt JSON.parse; return { _parseError: rawString } on failure. */
export function tryParseJson(s: string): unknown {
  try {
    return JSON.parse(s)
  } catch {
    return { _parseError: s }
  }
}

/** Normalise OpenAI finish_reason to our spec union. */
export function mapReason(
  reason: string | null | undefined,
): 'stop' | 'tool_calls' | 'length' | 'content_filter' | 'other' {
  switch (reason) {
    case 'stop':           return 'stop'
    case 'tool_calls':     return 'tool_calls'
    case 'length':         return 'length'
    case 'content_filter': return 'content_filter'
    default:               return 'other'
  }
}

/**
 * Convert a ParsedValue to a plain JS value.
 */
function parsedValueToJs(node: ParsedValue): unknown {
  switch (node._tag) {
    case 'string':  return node.value
    case 'number':  return Number(node.value)
    case 'boolean': return node.value
    case 'null':    return null
    case 'object':  return Object.fromEntries(node.entries.map(([k, v]) => [k, parsedValueToJs(v)]))
    case 'array':   return node.items.map(parsedValueToJs)
  }
}

/**
 * Walk a ParsedValue tree and diff against the snapshot, emitting field events.
 * Operates in-place: mutates `snapshot` and appends to `events`.
 */
function walkAndDiff(
  node: ParsedValue,
  path: readonly string[],
  toolCallId: string,
  snapshot: Map<string, FieldState>,
  events: ResponseStreamEvent[],
): void {
  const key = path.join('\0')
  let state = snapshot.get(key)

  if (!state) {
    // First time we see this path
    events.push({ type: 'tool_call_field_start', toolCallId, path })
    state = { seenText: '', complete: false }
    snapshot.set(key, state)
  }

  // Recurse into containers
  if (node._tag === 'object') {
    for (const [k, v] of node.entries) {
      walkAndDiff(v, [...path, k], toolCallId, snapshot, events)
    }
  } else if (node._tag === 'array') {
    for (let i = 0; i < node.items.length; i++) {
      walkAndDiff(node.items[i], [...path, String(i)], toolCallId, snapshot, events)
    }
  } else if (node._tag === 'string' || node._tag === 'number') {
    // Leaf text delta
    const newText = node.value
    if (newText.length > state.seenText.length) {
      const delta = newText.slice(state.seenText.length)
      events.push({ type: 'tool_call_field_delta', toolCallId, path, delta })
      state.seenText = newText
    }
  }
  // boolean/null: no deltas (short atoms), only start + end

  // Emit field_end when the node becomes complete
  if (node.state === 'complete' && !state.complete) {
    events.push({ type: 'tool_call_field_end', toolCallId, path, value: parsedValueToJs(node) })
    state.complete = true
  }
}

/**
 * Push a chunk to a tool call's streaming JSON parser, emit field events.
 */
function processToolCallChunk(
  tc: ToolCallState,
  chunk: string,
  events: ResponseStreamEvent[],
): void {
  tc.parser.push(chunk)
  const partial = tc.parser.partial
  if (partial !== undefined) {
    walkAndDiff(partial, [], tc.toolCallId, tc.snapshot, events)
  }
}

/**
 * Finalize a tool call: flush the parser and emit tool_call_end.
 */
function finalizeToolCall(
  tc: ToolCallState,
  events: ResponseStreamEvent[],
): void {
  tc.parser.end()
  const partial = tc.parser.partial
  if (partial !== undefined) {
    walkAndDiff(partial, [], tc.toolCallId, tc.snapshot, events)
  }
  events.push({ type: 'tool_call_end', toolCallId: tc.toolCallId })
}

// =============================================================================
// Pure chunk processor
// =============================================================================

export function processChunk(
  chunk: ChatCompletionsStreamChunk,
  state: DecoderState,
): { state: DecoderState; events: ResponseStreamEvent[] } {
  const events: ResponseStreamEvent[] = []
  let s = state

  const choice = chunk.choices[0]
  if (!choice) {
    return { state: s, events }
  }

  const delta = choice.delta

  // ── 1. reasoning_content ───────────────────────────────────────────────────
  if (delta.reasoning_content) {
    if (!s.thoughtOpen) {
      s = { ...s, thoughtOpen: true }
      events.push({ type: 'thought_start', level: 'medium' })
    }
    events.push({ type: 'thought_delta', text: delta.reasoning_content })
  }

  // ── 2. content ─────────────────────────────────────────────────────────────
  if (delta.content) {
    if (s.thoughtOpen) {
      events.push({ type: 'thought_end' })
      s = { ...s, thoughtOpen: false }
    }
    if (!s.messageOpen) {
      s = { ...s, messageOpen: true }
      events.push({ type: 'message_start' })
    }
    events.push({ type: 'message_delta', text: delta.content })
  }

  // ── 3. tool_calls ──────────────────────────────────────────────────────────
  if (delta.tool_calls && delta.tool_calls.length > 0) {
    if (s.thoughtOpen) {
      events.push({ type: 'thought_end' })
      s = { ...s, thoughtOpen: false }
    }
    if (s.messageOpen) {
      events.push({ type: 'message_end' })
      s = { ...s, messageOpen: false }
    }

    const calls = new Map(s.openToolCalls)
    let ord = s.ordinal

    for (const tc of delta.tool_calls) {
      let entry = calls.get(tc.index)

      if (!entry) {
        ord += 1
        const toolCallId = newToolCallId(ord)
        const toolName   = tc.function?.name ?? ''
        entry = {
          toolCallId,
          toolName,
          parser:   createStreamingJsonParser(),
          snapshot: new Map(),
        }
        calls.set(tc.index, entry)
        events.push({ type: 'tool_call_start', toolCallId, toolName })
      } else if (tc.function?.name && !entry.toolName) {
        entry = { ...entry, toolName: tc.function.name }
        calls.set(tc.index, entry)
      }

      if (tc.function?.arguments) {
        processToolCallChunk(entry, tc.function.arguments, events)
      }
    }

    s = { ...s, openToolCalls: calls, ordinal: ord }
  }

  // ── 4. finish_reason — close everything ───────────────────────────────────
  if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
    if (s.thoughtOpen) {
      events.push({ type: 'thought_end' })
      s = { ...s, thoughtOpen: false }
    }
    if (s.messageOpen) {
      events.push({ type: 'message_end' })
      s = { ...s, messageOpen: false }
    }
    for (const entry of s.openToolCalls.values()) {
      finalizeToolCall(entry, events)
    }
    s = { ...s, openToolCalls: new Map() }

    events.push({
      type: 'response_done',
      reason: mapReason(choice.finish_reason),
      usage: toResponseUsage(chunk.usage ?? {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
      }),
    })
  }

  return { state: s, events }
}

// =============================================================================
// Decode stream
// =============================================================================

export function decode(
  chunks: Stream.Stream<ChatCompletionsStreamChunk, DriverError>,
): Stream.Stream<ResponseStreamEvent, CodecDecodeError | DriverError> {
  return chunks.pipe(
    Stream.mapAccum(
      initialDecoderState,
      (state, chunk) => {
        const result = processChunk(chunk, state)
        return [result.state, result.events] as [DecoderState, ResponseStreamEvent[]]
      },
    ),
    Stream.flatMap(events => Stream.fromIterable(events)),
  )
}
