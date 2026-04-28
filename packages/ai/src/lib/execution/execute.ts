import { Effect, Option, Ref, Stream } from "effect"
import type { EncodeOptions } from "../codec/codec"
import type { ModelError } from "../errors/model-error"
import type { Prompt, PromptShape } from "../prompt/prompt"
import type { ResponseStreamEvent } from "../response/events"
import { retryModelStream } from "../retry/retry"
import type { ToolDefinition } from "../tools/tool-definition"
import { AiTracer } from "../tracing/tracer"
import type { ChatCompletionsRequest } from "../wire/chat-completions"
import type { BoundModel } from "./bound-model"

function traceRequest(
  bound: BoundModel,
  request: ChatCompletionsRequest,
): Effect.Effect<void> {
  return Effect.gen(function* () {
    const tracerOption = yield* Effect.serviceOption(AiTracer)
    if (Option.isNone(tracerOption)) return
    yield* tracerOption.value.traceRequest(bound.provider.id, bound.model.id, request).pipe(
      Effect.ignore,
    )
  })
}

function traceResponse(
  bound: BoundModel,
  events: readonly ResponseStreamEvent[],
): Effect.Effect<void> {
  return Effect.gen(function* () {
    const tracerOption = yield* Effect.serviceOption(AiTracer)
    if (Option.isNone(tracerOption)) return
    yield* tracerOption.value.traceResponse(bound.provider.id, bound.model.id, events).pipe(
      Effect.ignore,
    )
  })
}

function traceError(bound: BoundModel, error: ModelError): Effect.Effect<void> {
  return Effect.gen(function* () {
    const tracerOption = yield* Effect.serviceOption(AiTracer)
    if (Option.isNone(tracerOption)) return
    yield* tracerOption.value.traceError(bound.provider.id, bound.model.id, error).pipe(
      Effect.ignore,
    )
  })
}

export function execute(
  bound: BoundModel,
  prompt: Prompt | PromptShape,
  tools: readonly ToolDefinition<any, any>[],
  options: EncodeOptions,
): Stream.Stream<ResponseStreamEvent, ModelError> {
  const acquire = Effect.gen(function* () {
    const request = bound.codec.encode(bound.model.id, prompt, tools, options)

    yield* traceRequest(bound, request)

    const chunks = yield* retryModelStream(
      Effect.sync(() => bound.driver.stream(request, bound.endpoint, bound.authToken)),
    )

    const tracedChunks = chunks.pipe(
      Stream.catchAll((error) =>
        Stream.fromEffect(
          traceError(bound, error).pipe(
            Effect.zipRight(Effect.fail(error)),
          ),
        ),
      ),
    )

    const eventsRef = yield* Ref.make<readonly ResponseStreamEvent[]>([])

    return bound.codec.decode(tracedChunks).pipe(
      Stream.tap((event) => Ref.update(eventsRef, (events) => [...events, event])),
      Stream.ensuring(
        Ref.get(eventsRef).pipe(
          Effect.flatMap((events) => traceResponse(bound, events)),
          Effect.ignore,
        ),
      ),
    )
  })

  return Stream.unwrap(acquire)
}
