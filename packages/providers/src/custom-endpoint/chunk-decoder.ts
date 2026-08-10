import { Effect, Option, Schema } from "effect"
import {
  ChatChunkDeltaSchema,
  chatCompletionsStreamChunkFields,
  decodeChatCompletionsJson,
  decodeChatCompletionsPayload,
  normalizeChatCompletionsChunk,
  type ChatCompletionChunkDecoder,
} from "@magnitudedev/ai"

const OptionalNullableText = Schema.optionalWith(
  Schema.NullOr(Schema.String),
  { as: "Option", exact: true },
)

const CustomEndpointChunkDeltaSchema = Schema.extend(
  ChatChunkDeltaSchema,
  Schema.Struct({
    reasoning: OptionalNullableText,
    thinking: OptionalNullableText,
    reasoning_details: Schema.optionalWith(
      Schema.NullOr(Schema.Array(Schema.Struct({
        type: Schema.String,
        text: OptionalNullableText,
      }))),
      { as: "Option", exact: true },
    ),
  }),
)

const CustomEndpointChunkSchema = Schema.Struct(
  chatCompletionsStreamChunkFields(CustomEndpointChunkDeltaSchema),
)

type CustomEndpointChunk = Schema.Schema.Type<typeof CustomEndpointChunkSchema>

const nullableText = (
  value: Option.Option<string | null>,
): Option.Option<string> => Option.flatMap(value, Option.fromNullable).pipe(
  Option.filter((text) => text.length > 0),
)

const reasoningDetailsText = (
  details: CustomEndpointChunk["choices"][number]["delta"]["reasoning_details"],
): Option.Option<string> => Option.flatMap(details, (value) => {
  if (value === null) return Option.none()
  const text = value
    .filter((detail) => detail.type === "reasoning.text")
    .flatMap((detail) => Option.toArray(nullableText(detail.text)))
    .join("")
  return Option.liftPredicate(text, (value) => value.length > 0)
})

const thoughtFrom = (
  chunk: CustomEndpointChunk,
): Option.Option<string> => {
  const standard = Option.fromNullable(chunk.choices[0]?.delta.reasoning_content)
  const delta = chunk.choices[0]?.delta
  if (delta === undefined) return standard

  return Option.firstSomeOf([
    standard,
    nullableText(delta.reasoning),
    reasoningDetailsText(delta.reasoning_details),
    nullableText(delta.thinking),
  ])
}

export const customEndpointChunkDecoder: ChatCompletionChunkDecoder = {
  decode: (raw) => decodeChatCompletionsJson(raw).pipe(
    Effect.flatMap((payload) =>
      decodeChatCompletionsPayload(CustomEndpointChunkSchema, payload, raw),
    ),
    Effect.map((chunk) => normalizeChatCompletionsChunk(chunk, thoughtFrom(chunk))),
  ),
}
