import type { Stream } from "effect"
import type { ToolDefinition } from "../tools/tool-definition"
import { Prompt } from "../prompt/prompt"
import type { ResponseStreamEvent } from "../response/events"

export interface Codec<TWireReq, TWireChunk> {
  readonly id: string
  readonly encodePrompt: (
    model: string,
    prompt: Prompt,
    tools: readonly ToolDefinition[],
  ) => Partial<TWireReq>
  readonly decode: <E>(
    chunks: Stream.Stream<TWireChunk, E>,
  ) => Stream.Stream<ResponseStreamEvent, E>
}
