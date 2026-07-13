import type { Effect, Stream } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import { Prompt } from "../prompt/prompt"
import type { BoundModel } from "./bound-model"
import type { ProviderModelCapabilities } from "./capabilities"
import type { ToolCallId } from "../prompt/ids"
import type { ToolDefinition } from "../tools/tool-definition"
import type { ResponseStreamEvent } from "../response/events"
import type { AuthApplicator } from "../auth/auth"
import type { ImagePlaceholderConfig } from "./capabilities"
import type { StreamStartFailure } from "../errors/failure"
import type { StreamingFieldParser } from "../streaming/field-parser"
import type { TokenLogprob } from "../trace"

export type ModelStreamResult = {
  readonly events: Stream.Stream<ResponseStreamEvent, never>
  readonly parsers: ReadonlyMap<ToolCallId, StreamingFieldParser>
  readonly logprobs: TokenLogprob[]
  readonly requestId: string | null
}

export interface ModelSpec<
  TCallOptions,
> {
  readonly modelId: string
  readonly endpoint: string
  readonly capabilities?: ProviderModelCapabilities

  /** Bind this spec with auth and optional default options to create a BoundModel. */
  readonly bind: (args: {
    auth: AuthApplicator,
    defaults?: Partial<TCallOptions>,
    imagePlaceholders?: ImagePlaceholderConfig,
  }) => BoundModel<TCallOptions>

  /** @internal — closed over codec, options, transport config, and start-failure mapper */
  readonly _execute: (
    auth: AuthApplicator,
    prompt: Prompt,
    tools: readonly ToolDefinition[],
    options: TCallOptions,
  ) => Effect.Effect<
    ModelStreamResult,
    StreamStartFailure,
    HttpClient.HttpClient
  >
}
