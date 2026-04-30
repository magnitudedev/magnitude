import type { Effect, Stream } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import { Prompt } from "../prompt/prompt"
import type { BoundModel } from "./bound-model"
import type { ToolDefinition } from "../tools/tool-definition"
import type { ResponseStreamEvent } from "../response/events"
import type { AuthApplicator } from "../auth/auth"
import type { ConnectionError, StreamError } from "../errors/model-error"

export interface ModelSpec<
  TCallOptions,
  TConnectionError = ConnectionError,
  TStreamError = StreamError,
> {
  readonly modelId: string
  readonly endpoint: string
  readonly contextWindow: number
  readonly maxOutputTokens: number

  /** Bind this spec with auth and optional default options to create a BoundModel. */
  readonly bind: (args: {
    auth: AuthApplicator,
    defaults?: Partial<TCallOptions>,
  }) => BoundModel<TCallOptions, TConnectionError, TStreamError>

  /** @internal — closed over codec, options, transport config, classifiers */
  readonly _execute: (
    auth: AuthApplicator,
    prompt: Prompt,
    tools: readonly ToolDefinition[],
    options: TCallOptions,
  ) => Effect.Effect<
    Stream.Stream<ResponseStreamEvent, TStreamError>,
    TConnectionError,
    HttpClient.HttpClient
  >
}
