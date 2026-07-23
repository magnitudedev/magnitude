import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { PromptBuilder, ProviderModelIdSchema } from "@magnitudedev/ai"
import { Effect, Exit, Layer, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { IcnClient } from "../client.js"
import { makeIcnApiClient } from "../generated/client.js"
import { IcnInventory, makeIcnInventory } from "../inventory/index.js"
import { IcnRecipes } from "../recipes/service.js"
import { IcnProvider, makeIcnProvider } from "./source.js"

const TEST_BASE_URL = "http://icn.test"

const makeTestLayer = (http: HttpClient.HttpClient) => {
  const httpLayer = Layer.succeed(HttpClient.HttpClient, http)
  const clientLayer = Layer.effect(
    IcnClient,
    makeIcnApiClient({ baseUrl: TEST_BASE_URL }),
  ).pipe(Layer.provide(httpLayer))
  const inventoryLayer = makeIcnInventory().pipe(Layer.provide(clientLayer))
  const recipesLayer = Layer.succeed(IcnRecipes, IcnRecipes.of({
    get: Effect.succeed({ revision: 0, state: { _tag: "Loading" } }),
    changes: Stream.empty,
    refresh: Effect.void,
    resolve: () => Effect.succeed(Option.none()),
  }))
  const dependencies = Layer.mergeAll(clientLayer, inventoryLayer, recipesLayer)

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
  it("projects its catalog from observed inventory and refreshes through the inventory owner", async () => {
    let inventoryRequests = 0
    const http = HttpClient.make((request) => {
      return Effect.sync(() => {
        inventoryRequests += 1
        return jsonResponse(request, '{"object":"list","data":[]}')
      })
    })

    await Effect.runPromise(Effect.gen(function* () {
      const provider = yield* IcnProvider
      expect(yield* provider.catalog.list).toEqual([])
      expect(inventoryRequests).toBe(1)
      expect(yield* provider.catalog.refresh).toEqual([])
      expect(inventoryRequests).toBe(2)
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
