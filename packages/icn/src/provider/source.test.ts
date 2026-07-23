import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { PromptBuilder, ProviderModelIdSchema } from "@magnitudedev/ai"
import { Effect, Exit, Layer, Option } from "effect"
import { describe, expect, it } from "vitest"
import { IcnClient } from "../client.js"
import { makeIcnApiClient } from "../generated/client.js"
import { IcnProvider, IcnProviderModelResolver, makeIcnProvider } from "./source.js"

const TEST_BASE_URL = "http://icn.test"

const makeTestLayer = (http: HttpClient.HttpClient) => {
  const httpLayer = Layer.succeed(HttpClient.HttpClient, http)
  const clientLayer = Layer.effect(
    IcnClient,
    makeIcnApiClient({ baseUrl: TEST_BASE_URL }),
  ).pipe(Layer.provide(httpLayer))
  const resolverLayer = Layer.succeed(IcnProviderModelResolver, IcnProviderModelResolver.of({
    resolve: () => Effect.succeed(Option.none()),
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
})
