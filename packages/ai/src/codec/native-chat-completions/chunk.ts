import { Option } from "effect"
import type { FinishReason, RawInputToken, RawOutputToken } from "../../response/events"
import type { ResponseUsage } from "../../response/usage"
import type { TokenLogprob } from "../../trace"
import type { ChatCompletionsStreamChunk } from "../../wire/chat-completions"

export interface ChatCompletionProviderError {
  readonly message: string
  readonly type: Option.Option<string>
  readonly code: Option.Option<string>
  readonly param: Option.Option<string>
}

export interface ChatCompletionToolCallDelta {
  readonly index: number
  readonly providerToolCallId: Option.Option<string>
  readonly name: Option.Option<string>
  readonly input: Option.Option<string>
}

export interface ChatCompletionDelta {
  readonly text: Option.Option<string>
  readonly thought: Option.Option<string>
  readonly toolCalls: readonly ChatCompletionToolCallDelta[]
  readonly logprobs: readonly TokenLogprob[]
}

export interface NormalizedChatCompletionsStreamChunk {
  readonly delta: Option.Option<ChatCompletionDelta>
  readonly finishReason: Option.Option<FinishReason>
  readonly usage: Option.Option<ResponseUsage>
  readonly error: Option.Option<ChatCompletionProviderError>
  readonly rawInput: Option.Option<ReadonlyArray<RawInputToken>>
  readonly rawOutput: Option.Option<ReadonlyArray<RawOutputToken>>
}

const finishReason = (reason: string): FinishReason => {
  switch (reason) {
    case "stop":
    case "tool_calls":
    case "length":
    case "content_filter":
    case "end_turn":
      return reason
    default:
      return "unknown"
  }
}

const usage = (
  value: NonNullable<ChatCompletionsStreamChunk["usage"]>,
): ResponseUsage => ({
  inputTokens: value.prompt_tokens,
  outputTokens: value.completion_tokens,
  cacheReadTokens: value.prompt_tokens_details?.cached_tokens ?? 0,
  cacheWriteTokens: 0,
  cost: value.cost ?? null,
})

export const normalizeChatCompletionsChunk = (
  chunk: ChatCompletionsStreamChunk,
  thought: Option.Option<string>,
): NormalizedChatCompletionsStreamChunk => {
  const choice = chunk.choices[0]
  const delta = choice === undefined
    ? Option.none<ChatCompletionDelta>()
    : Option.some<ChatCompletionDelta>({
        text: Option.fromNullable(choice.delta.content).pipe(
          Option.filter((text) => text.length > 0),
        ),
        thought: Option.filter(thought, (text) => text.length > 0),
        toolCalls: (choice.delta.tool_calls ?? []).map((toolCall) => ({
          index: toolCall.index,
          providerToolCallId: Option.fromNullable(toolCall.id),
          name: Option.fromNullable(toolCall.function?.name).pipe(
            Option.filter((name) => name.length > 0),
          ),
          input: Option.fromNullable(toolCall.function?.arguments).pipe(
            Option.filter((input) => input.length > 0),
          ),
        })),
        logprobs: (choice.logprobs?.content ?? []).map((token) => ({
          token: token.token,
          logprob: token.logprob,
          topLogprobs: token.top_logprobs.map((candidate) => ({
            token: candidate.token,
            logprob: candidate.logprob,
          })),
        })),
      })

  return {
    delta,
    finishReason: Option.fromNullable(choice?.finish_reason).pipe(
      Option.map(finishReason),
    ),
    usage: Option.fromNullable(chunk.usage).pipe(Option.map(usage)),
    error: Option.fromNullable(chunk.error).pipe(
      Option.map((error) => ({
        message: error.message,
        type: Option.fromNullable(error.type),
        code: Option.fromNullable(error.code),
        param: Option.fromNullable(error.param),
      })),
    ),
    rawInput: Option.fromNullable(chunk.raw_input),
    rawOutput: Option.fromNullable(chunk.raw_output),
  }
}
