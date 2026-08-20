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
  value: Option.Option.Value<ChatCompletionsStreamChunk["usage"]>,
): ResponseUsage => ({
  inputTokens: value.prompt_tokens,
  outputTokens: value.completion_tokens,
  cacheReadTokens: Option.flatMap(
    value.prompt_tokens_details,
    (details) => details.cached_tokens,
  ).pipe(Option.getOrElse(() => 0)),
  cacheWriteTokens: 0,
  cost: Option.getOrNull(value.cost),
})

export const normalizeChatCompletionsChunk = (
  chunk: ChatCompletionsStreamChunk,
  thought: Option.Option<string>,
): NormalizedChatCompletionsStreamChunk => {
  const choice = chunk.choices[0]
  const delta = choice === undefined
    ? Option.none<ChatCompletionDelta>()
    : Option.some<ChatCompletionDelta>({
        text: choice.delta.content.pipe(
          Option.filter((text) => text.length > 0),
        ),
        thought: Option.filter(thought, (text) => text.length > 0),
        toolCalls: Option.getOrElse(choice.delta.tool_calls, () => []).map((toolCall) => ({
          index: toolCall.index,
          providerToolCallId: toolCall.id,
          name: Option.flatMap(toolCall.function, (fn) => fn.name).pipe(
            Option.filter((name) => name.length > 0),
          ),
          input: Option.flatMap(toolCall.function, (fn) => fn.arguments).pipe(
            Option.filter((input) => input.length > 0),
          ),
        })),
        logprobs: Option.flatMap(choice.logprobs, (logprobs) => logprobs.content).pipe(
          Option.getOrElse(() => []),
        ).map((token) => ({
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
    finishReason: choice === undefined ? Option.none() : choice.finish_reason.pipe(
      Option.map(finishReason),
    ),
    usage: chunk.usage.pipe(Option.map(usage)),
    error: chunk.error.pipe(
      Option.map((error) => ({
        message: error.message,
        type: error.type,
        code: error.code,
        param: error.param,
      })),
    ),
    rawInput: chunk.raw_input,
    rawOutput: chunk.raw_output,
  }
}
