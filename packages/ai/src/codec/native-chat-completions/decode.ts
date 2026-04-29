import { Stream } from "effect"
import { createStreamingJsonParser } from "../../jsonish/parser"
import type { ParsedValue, StreamingJsonParser } from "../../jsonish/types"
import type { ToolCallId } from "../../prompt/ids"
import type { JsonValue } from "../../prompt/parts"
import type { ResponseStreamEvent } from "../../response/events"
import type { ResponseUsage } from "../../response/usage"
import type { ChatCompletionsStreamChunk } from "../../wire/chat-completions"

interface FieldState {
  seenText: string
  complete: boolean
}

interface ToolCallState {
  readonly toolCallId: ToolCallId
  readonly toolName: string
  readonly parser: StreamingJsonParser
  readonly snapshot: Map<string, FieldState>
}

interface DecoderState {
  readonly nextToolOrdinal: number
  readonly thoughtOpen: boolean
  readonly messageOpen: boolean
  readonly openToolCalls: ReadonlyMap<number, ToolCallState>
}

const initialDecoderState: DecoderState = {
  nextToolOrdinal: 0,
  thoughtOpen: false,
  messageOpen: false,
  openToolCalls: new Map(),
}

function toUsage(
  usage: NonNullable<ChatCompletionsStreamChunk["usage"]>,
): ResponseUsage {
  return {
    inputTokens: usage.prompt_tokens,
    outputTokens: usage.completion_tokens,
    cacheReadTokens: usage.prompt_tokens_details?.cached_tokens ?? 0,
  }
}

function mapReason(reason: string | null | undefined): string {
  switch (reason) {
    case "stop":
    case "tool_calls":
    case "length":
    case "content_filter":
      return reason
    default:
      return "other"
  }
}

function parsedValueToJson(node: ParsedValue): JsonValue {
  switch (node._tag) {
    case "string":
      return node.value
    case "number":
      return Number(node.value)
    case "boolean":
      return node.value
    case "null":
      return null
    case "array":
      return node.items.map(parsedValueToJson)
    case "object":
      return Object.fromEntries(
        node.entries.map(([key, value]) => [key, parsedValueToJson(value)]),
      )
  }
}

function walkAndDiff(
  node: ParsedValue,
  path: readonly string[],
  toolCallId: ToolCallId,
  snapshot: Map<string, FieldState>,
  events: ResponseStreamEvent[],
): void {
  const key = path.join("\0")
  let state = snapshot.get(key)

  if (!state) {
    events.push({
      _tag: "tool_call_field_start",
      toolCallId,
      path,
    })
    state = { seenText: "", complete: false }
    snapshot.set(key, state)
  }

  if (node._tag === "object") {
    for (const [childKey, childValue] of node.entries) {
      walkAndDiff(childValue, [...path, childKey], toolCallId, snapshot, events)
    }
  } else if (node._tag === "array") {
    for (let index = 0; index < node.items.length; index += 1) {
      walkAndDiff(node.items[index], [...path, String(index)], toolCallId, snapshot, events)
    }
  } else if (node._tag === "string" || node._tag === "number") {
    if (node.value.length > state.seenText.length) {
      const delta = node.value.slice(state.seenText.length)
      events.push({
        _tag: "tool_call_field_delta",
        toolCallId,
        path,
        delta,
      })
      state.seenText = node.value
    }
  }

  if (node.state === "complete" && !state.complete) {
    events.push({
      _tag: "tool_call_field_end",
      toolCallId,
      path,
      value: parsedValueToJson(node),
    })
    state.complete = true
  }
}

function processToolCallChunk(
  toolCall: ToolCallState,
  chunk: string,
  events: ResponseStreamEvent[],
): void {
  toolCall.parser.push(chunk)
  const partial = toolCall.parser.partial
  if (partial !== undefined) {
    walkAndDiff(partial, [], toolCall.toolCallId, toolCall.snapshot, events)
  }
}

function finalizeToolCall(
  toolCall: ToolCallState,
  events: ResponseStreamEvent[],
): void {
  toolCall.parser.end()
  const partial = toolCall.parser.partial
  if (partial !== undefined) {
    walkAndDiff(partial, [], toolCall.toolCallId, toolCall.snapshot, events)
  }
  events.push({
    _tag: "tool_call_end",
    toolCallId: toolCall.toolCallId,
  })
}

function processChunk(
  chunk: ChatCompletionsStreamChunk,
  state: DecoderState,
): readonly [DecoderState, readonly ResponseStreamEvent[]] {
  const events: ResponseStreamEvent[] = []
  let nextState = state

  const choice = chunk.choices[0]
  if (!choice) {
    return [nextState, events]
  }

  const delta = choice.delta

  if (delta.reasoning_content) {
    if (!nextState.thoughtOpen) {
      nextState = { ...nextState, thoughtOpen: true }
      events.push({ _tag: "thought_start", level: "medium" })
    }
    events.push({ _tag: "thought_delta", text: delta.reasoning_content })
  }

  if (delta.content) {
    if (nextState.thoughtOpen) {
      events.push({ _tag: "thought_end" })
      nextState = { ...nextState, thoughtOpen: false }
    }
    if (!nextState.messageOpen) {
      nextState = { ...nextState, messageOpen: true }
      events.push({ _tag: "message_start" })
    }
    events.push({ _tag: "message_delta", text: delta.content })
  }

  if (delta.tool_calls && delta.tool_calls.length > 0) {
    if (nextState.thoughtOpen) {
      events.push({ _tag: "thought_end" })
      nextState = { ...nextState, thoughtOpen: false }
    }
    if (nextState.messageOpen) {
      events.push({ _tag: "message_end" })
      nextState = { ...nextState, messageOpen: false }
    }

    const calls = new Map(nextState.openToolCalls)
    let nextToolOrdinal = nextState.nextToolOrdinal

    for (const toolCallDelta of delta.tool_calls) {
      let toolCall = calls.get(toolCallDelta.index)

      if (!toolCall) {
        nextToolOrdinal += 1
        toolCall = {
          toolCallId: (toolCallDelta.id ?? `tool_call_${nextToolOrdinal}`) as ToolCallId,
          toolName: toolCallDelta.function?.name ?? "",
          parser: createStreamingJsonParser(),
          snapshot: new Map(),
        }
        calls.set(toolCallDelta.index, toolCall)
        events.push({
          _tag: "tool_call_start",
          toolCallId: toolCall.toolCallId,
          toolName: toolCall.toolName,
        })
      } else if (toolCallDelta.function?.name && toolCall.toolName.length === 0) {
        toolCall = {
          ...toolCall,
          toolName: toolCallDelta.function.name,
        }
        calls.set(toolCallDelta.index, toolCall)
      }

      if (toolCallDelta.function?.arguments) {
        processToolCallChunk(toolCall, toolCallDelta.function.arguments, events)
      }
    }

    nextState = {
      ...nextState,
      nextToolOrdinal,
      openToolCalls: calls,
    }
  }

  if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
    if (nextState.thoughtOpen) {
      events.push({ _tag: "thought_end" })
      nextState = { ...nextState, thoughtOpen: false }
    }
    if (nextState.messageOpen) {
      events.push({ _tag: "message_end" })
      nextState = { ...nextState, messageOpen: false }
    }
    for (const toolCall of nextState.openToolCalls.values()) {
      finalizeToolCall(toolCall, events)
    }
    nextState = {
      ...nextState,
      openToolCalls: new Map(),
    }
    events.push({
      _tag: "response_done",
      reason: mapReason(choice.finish_reason),
      ...(chunk.usage ? { usage: toUsage(chunk.usage) } : {}),
    })
  }

  return [nextState, events]
}

export function decode<E>(
  chunks: Stream.Stream<ChatCompletionsStreamChunk, E>,
): Stream.Stream<ResponseStreamEvent, E> {
  return Stream.flatMap(
    Stream.mapAccum(
      chunks,
      initialDecoderState,
      (state, chunk): readonly [DecoderState, readonly ResponseStreamEvent[]] => {
        const [nextState, events] = processChunk(chunk, state)
        return [nextState, events]
      },
    ),
    (events) => Stream.fromIterable(events),
  )
}
