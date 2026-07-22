import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { PromptBuilder, ProviderModelIdSchema } from "@magnitudedev/ai"
import { Effect, Either, Layer } from "effect"
import { describe, expect, it } from "vitest"
import { IcnClient } from "../client.js"
import { makeIcnApiClient } from "../generated/client.js"
import { IcnInventory, makeIcnInventory } from "../inventory/index.js"
import { IcnProvider, makeIcnProvider } from "./source.js"

const TEST_BASE_URL = "http://icn.test"

const makeTestLayer = (http: HttpClient.HttpClient) => {
  const httpLayer = Layer.succeed(HttpClient.HttpClient, http)
  const clientLayer = Layer.effect(
    IcnClient,
    makeIcnApiClient({ baseUrl: TEST_BASE_URL }),
  ).pipe(Layer.provide(httpLayer))
  const inventoryLayer = makeIcnInventory({
    refreshInterval: "1 hour",
  }).pipe(Layer.provide(clientLayer))
  const dependencies = Layer.merge(clientLayer, inventoryLayer)

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
    const http = HttpClient.make((request) => Effect.sync(() => {
      inventoryRequests += 1
      return jsonResponse(request, '{"object":"list","data":[]}')
    }))

    await Effect.runPromise(Effect.gen(function* () {
      const provider = yield* IcnProvider
      expect(yield* provider.catalog.list).toEqual([])
      expect(inventoryRequests).toBe(1)
      expect(yield* provider.catalog.refresh).toEqual([])
      expect(inventoryRequests).toBe(2)
    }).pipe(Effect.provide(makeTestLayer(http))))
  })

  it("preserves an ICN HTTP rejection as a stream-start provider rejection", async () => {
    const http = HttpClient.make((request) => Effect.succeed(
      request.url.endsWith("/v1/chat/completions")
        ? jsonResponse(
          request,
          JSON.stringify({
            error: {
              message: "assistant messages require content, reasoning_content, or tool_calls",
              type: "invalid_request_error",
              code: "invalid_request",
            },
          }),
          400,
        )
        : jsonResponse(request, '{"object":"list","data":[]}'),
    ))
    const modelId = ProviderModelIdSchema.make("mdl_test")
    const prompt = PromptBuilder.empty().user("hello").build()

    const result = await Effect.runPromise(Effect.gen(function* () {
      const provider = yield* IcnProvider
      const bound = yield* provider.bindModel(modelId)
      return yield* bound.stream(prompt, []).pipe(Effect.either)
    }).pipe(Effect.provide(makeTestLayer(http))))

    expect(Either.isLeft(result)).toBe(true)
    if (Either.isRight(result)) return
    expect(result.left._tag).toBe("StreamStartProviderRejection")
  })
})
