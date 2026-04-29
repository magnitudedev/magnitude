import type { Effect, Stream } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import { Prompt } from "../prompt/prompt"
import type { ToolDefinition } from "../tools/tool-definition"
import type { ResponseStreamEvent } from "../response/events"
import type { ConnectionError, StreamError } from "../errors/model-error"
import type { ModelSpec } from "./model-spec"

export interface BoundModel<
  TCallOptions,
  TConnectionError = ConnectionError,
  TStreamError = StreamError,
> {
  readonly spec: ModelSpec<TCallOptions, TConnectionError, TStreamError>

  readonly stream: (
    prompt: Prompt,
    tools: readonly ToolDefinition[],
    options?: TCallOptions,
  ) => Effect.Effect<
    Stream.Stream<ResponseStreamEvent, TStreamError>,
    TConnectionError,
    HttpClient.HttpClient
  >
}
