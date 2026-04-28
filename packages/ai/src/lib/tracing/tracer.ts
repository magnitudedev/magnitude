import { Context, Effect, Layer } from "effect"
import type { ModelError } from "../errors/model-error"
import type { ResponseStreamEvent } from "../response/events"
import type { ChatCompletionsRequest } from "../wire/chat-completions"

export class AiTracer extends Context.Tag("@magnitudedev/ai/AiTracer")<
  AiTracer,
  {
    readonly traceRequest: (
      provider: string,
      model: string,
      request: ChatCompletionsRequest,
    ) => Effect.Effect<void>
    readonly traceResponse: (
      provider: string,
      model: string,
      events: readonly ResponseStreamEvent[],
    ) => Effect.Effect<void>
    readonly traceError: (
      provider: string,
      model: string,
      error: ModelError,
    ) => Effect.Effect<void>
  }
>() {}

export const NoopAiTracer: Context.Tag.Service<typeof AiTracer> = {
  traceRequest: () => Effect.void,
  traceResponse: () => Effect.void,
  traceError: () => Effect.void,
}

export const NoopAiTracerLive = Layer.succeed(AiTracer, NoopAiTracer)
