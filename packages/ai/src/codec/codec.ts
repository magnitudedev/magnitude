import type { Effect, Schema, Stream } from "effect"
import type { JsonRecord } from "@magnitudedev/utils/schema"
import type { StreamFailure, StreamFailureContext } from "../errors/failure"
import type { ToolCallId } from "../prompt/ids"
import type { ToolDefinition } from "../tools/tool-definition"
import type { Prompt } from "../prompt/prompt"
import type { ResponseStreamEvent } from "../response/events"
import type { StreamingFieldParser } from "../streaming/field-parser"
import type { TokenLogprob } from "../trace"

export interface Codec<
  TPromptValue,
  TPromptContribution extends JsonRecord,
  TWireChunk,
  TPromptEncodeError,
> {
  readonly id: string
  readonly promptSchema: Schema.Schema<TPromptValue, TPromptContribution, never>
  readonly encodePrompt: (
    model: string,
    prompt: Prompt,
    tools: readonly ToolDefinition[],
  ) => Effect.Effect<TPromptContribution, TPromptEncodeError>
  readonly decode: <E>(
    chunks: Stream.Stream<TWireChunk, E>,
    options: {
      tools?: readonly ToolDefinition[]
      streamContext: StreamFailureContext
      generateToolCallId?: () => ToolCallId
      toStreamFailure: (error: E) => StreamFailure
    },
  ) => {
    readonly events: Stream.Stream<ResponseStreamEvent, never>
    readonly parsers: ReadonlyMap<ToolCallId, StreamingFieldParser>
    readonly logprobs: TokenLogprob[]
  }
}
