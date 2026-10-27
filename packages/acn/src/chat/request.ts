import { type Effect, Option, type ParseResult, Schema } from "effect"
import * as Native from "@magnitudedev/icn-protocol/schemas"
import type { ChatAssistantMessage, ChatMessage, ResponseStarted } from "@magnitudedev/acn-protocol"

// The engine request for a response (spec section 6), from the transcript before it and its
// `response_started`.

const field = <K extends string, A>(key: K, value: Option.Option<A>) =>
  Option.match(value, { onNone: () => ({}), onSome: (present) => ({ [key]: present }) as { readonly [P in K]: A } })

const textOf = (message: ChatAssistantMessage) =>
  message.content.flatMap((part) => (part._tag === "text" ? [part.text] : [])).join("")

export const buildRequest = (
  history: readonly ChatMessage[],
  started: ResponseStarted
): Effect.Effect<Native.ChatCompletionRequest, ParseResult.ParseError> => {
  const { settings } = started
  return Schema.decode(Native.ChatCompletionRequest)({
    model: started.model,
    stream: true,
    timings_per_token: true,
    messages: [
      { role: "system", content: settings.systemPrompt },
      // Reasoning is not sent back, and a response without text is left out.
      ...history.flatMap((message): readonly Native.ChatMessageRequestEncoded[] => {
        if (message._tag === "user") return [{ role: "user", content: message.text }]
        const text = textOf(message)
        return text.length > 0 ? [{ role: "assistant", content: text }] : []
      }),
    ],
    ...field("reasoning_effort", started.reasoningEffort),
    ...field("temperature", settings.temperature),
    ...field("max_tokens", settings.maxOutputTokens),
    ...field("top_p", settings.topP),
    ...field("top_k", settings.topK),
    ...field("min_p", settings.minP),
    ...field("repetition_penalty", settings.repetitionPenalty),
    ...field("presence_penalty", settings.presencePenalty),
    ...field("frequency_penalty", settings.frequencyPenalty),
    ...field("seed", settings.seed),
    ...(settings.stop.length > 0 ? { stop: settings.stop } : {}),
  })
}
