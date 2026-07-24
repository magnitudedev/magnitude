import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import {
  PromptBuilder,
  ProviderModelIdSchema,
  type ModelRequestProgress,
} from "@magnitudedev/ai"
import { Effect, Exit, Layer, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { IcnClient } from "../client.js"
import { makeIcnApiClient } from "../generated/client.js"
import { IcnProvider, IcnProviderModelResolver, makeIcnProvider } from "./source.js"

const TEST_BASE_URL = "http://icn.test"

const makeTestLayer = (
  http: HttpClient.HttpClient,
  runtimeModelId?: ReturnType<typeof ProviderModelIdSchema.make>,
) => {
  const httpLayer = Layer.succeed(HttpClient.HttpClient, http)
  const clientLayer = Layer.effect(
    IcnClient,
    makeIcnApiClient({ baseUrl: TEST_BASE_URL }),
  ).pipe(Layer.provide(httpLayer))
  const resolverLayer = Layer.succeed(IcnProviderModelResolver, IcnProviderModelResolver.of({
    resolve: () => Effect.succeed(
      runtimeModelId === undefined
        ? Option.none()
        : Option.some({ runtimeModelId }),
    ),
  }))
  const dependencies = Layer.merge(clientLayer, resolverLayer)

  return makeIcnProvider().pipe(
    Layer.provide(dependencies),
    Layer.merge(dependencies),
    Layer.merge(httpLayer),
  )
}

const jsonResponse = (
  request: Parameters<Parameters<typeof HttpClient.make>[0]>[0],
  body: string,
  status = 200,
) => HttpClientResponse.fromWeb(
  request,
  new Response(body, {
    status,
    headers: { "content-type": "application/json" },
  }),
)

const sseResponse = (
  request: Parameters<Parameters<typeof HttpClient.make>[0]>[0],
  events: readonly object[],
) => HttpClientResponse.fromWeb(
  request,
  new Response(
    `${events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")}data: [DONE]\n\n`,
    { status: 200, headers: { "content-type": "text/event-stream" } },
  ),
)

describe("ICN local provider", () => {
  it("keeps the local provider catalog product-owned", async () => {
    const http = HttpClient.make((request) =>
      Effect.succeed(jsonResponse(request, '{"object":"list","data":[]}')))

    await Effect.runPromise(Effect.gen(function* () {
      const provider = yield* IcnProvider
      expect(yield* provider.catalog.list).toEqual([])
      expect(yield* provider.catalog.refresh).toEqual([])
    }).pipe(Effect.provide(makeTestLayer(http))))
  })

  it("fails before inference when a public model has no native association", async () => {
    let chatRequests = 0
    const http = HttpClient.make((request) => {
      if (request.url.endsWith("/v1/chat/completions")) chatRequests += 1
      return Effect.succeed(jsonResponse(request, '{"object":"list","data":[]}'))
    })
    const modelId = ProviderModelIdSchema.make("mdl_test")

    const result = await Effect.runPromiseExit(Effect.gen(function* () {
      const provider = yield* IcnProvider
      const bound = yield* provider.bindModel(modelId)
      return yield* bound.stream(PromptBuilder.empty().user("hello").build(), [])
    }).pipe(Effect.provide(makeTestLayer(http))))

    expect(Exit.isFailure(result)).toBe(true)
    expect(chatRequests).toBe(0)
  })

  it("reports native request progress without exposing control chunks as model output", async () => {
    const modelId = ProviderModelIdSchema.make("mdl_test")
    const chunk = {
      id: "request-1",
      object: "chat.completion.chunk",
      created: 1,
      model: modelId,
    }
    const http = HttpClient.make((request) => Effect.succeed(sseResponse(request, [
      { ...chunk, choices: [], progress: { _tag: "Queued" } },
      { ...chunk, choices: [], progress: { _tag: "Generating" } },
      {
        ...chunk,
        choices: [{
          index: 0,
          delta: { role: "assistant", content: null },
          finish_reason: null,
        }],
      },
      {
        ...chunk,
        choices: [{
          index: 0,
          delta: { content: "hello" },
          finish_reason: null,
        }],
      },
      {
        ...chunk,
        choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      },
      {
        ...chunk,
        choices: [],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    ])))
    const progress: ModelRequestProgress[] = []

    const output = await Effect.runPromise(Effect.gen(function* () {
      const provider = yield* IcnProvider
      const bound = yield* provider.bindModel(modelId, {
        requestAttribution: {
          key: "test",
          requestStarted: Effect.void,
          requestProgress: (update) => Effect.sync(() => {
            progress.push(update)
          }),
        },
      })
      const result = yield* bound.stream(PromptBuilder.empty().user("hello").build(), [])
      return yield* Stream.runCollect(result.events)
    }).pipe(Effect.provide(makeTestLayer(http, modelId))))

    expect(Array.from(output).map((event) => event._tag)).toEqual(["stream_end"])
    expect(progress).toEqual([
      { phase: "queued", requestId: "request-1" },
      { phase: "generating", requestId: "request-1" },
      { phase: "cleared", requestId: "request-1" },
    ])
  })
})
