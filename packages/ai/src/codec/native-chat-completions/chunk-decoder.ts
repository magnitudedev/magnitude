import { Data, Effect, Option, Schema } from "effect"
import type * as ParseResult from "effect/ParseResult"
import { ChatCompletionsStreamChunk } from "../../wire/chat-completions"
import {
  type NormalizedChatCompletionsStreamChunk,
  normalizeChatCompletionsChunk,
} from "./chunk"

export class ChatPayloadJsonParseError extends Data.TaggedError("ChatPayloadJsonParseError")<{
  readonly message: string
  readonly raw: string
}> {}

export class ChatPayloadSchemaDecodeError extends Data.TaggedError("ChatPayloadSchemaDecodeError")<{
  readonly message: string
  readonly raw: string
}> {}

export type ChatCompletionChunkDecodeError =
  | ChatPayloadJsonParseError
  | ChatPayloadSchemaDecodeError

export interface ChatCompletionChunkDecoder {
  readonly decode: (
    raw: string,
  ) => Effect.Effect<NormalizedChatCompletionsStreamChunk, ChatCompletionChunkDecodeError>
}

export const decodeChatCompletionsJson = (
  raw: string,
): Effect.Effect<unknown, ChatPayloadJsonParseError> =>
  Schema.decodeUnknown(Schema.parseJson(Schema.Unknown))(raw).pipe(
    Effect.mapError(() => new ChatPayloadJsonParseError({
      message: "Invalid JSON chat-completions payload",
      raw,
    })),
  )

export const decodeChatCompletionsPayload = <A, I, R>(
  schema: Schema.Schema<A, I, R>,
  payload: unknown,
  raw: string,
): Effect.Effect<A, ChatPayloadSchemaDecodeError, R> =>
  Schema.decodeUnknown(schema)(payload).pipe(
    Effect.mapError((error: ParseResult.ParseError) => new ChatPayloadSchemaDecodeError({
      message: `Chat-completions chunk decode failed: ${String(error)}`,
      raw,
    })),
  )

export const standardChatCompletionChunkDecoder: ChatCompletionChunkDecoder = {
  decode: (raw) => decodeChatCompletionsJson(raw).pipe(
    Effect.flatMap((payload) =>
      decodeChatCompletionsPayload(ChatCompletionsStreamChunk, payload, raw)),
    Effect.map((chunk) => normalizeChatCompletionsChunk(
      chunk,
      Option.fromNullable(chunk.choices[0]?.delta.reasoning_content),
    )),
  ),
}
