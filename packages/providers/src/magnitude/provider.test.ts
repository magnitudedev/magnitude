import { FetchHttpClient } from "@effect/platform"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { PromptBuilder, ProviderModelIdSchema } from "@magnitudedev/ai"
import { Cause, Chunk, Effect, Fiber, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { createMagnitudeProvider } from "./provider"

describe("Magnitude provider authentication", () => {
  it("Schema-encodes provider-specific bind options into the request", async () => {
    let requestBody: unknown
    const server = Bun.serve({
      port: 0,
      fetch: async (request) => {
        requestBody = await request.json()
        return new Response("data: [DONE]\n\n", {
          headers: { "content-type": "text/event-stream" },
        })
      },
    })

    try {
      const instance = createMagnitudeProvider({
        apiKey: "test-key",
        endpoint: `http://127.0.0.1:${server.port}`,
        sessionId: "session-1",
      })
      const model = await Effect.runPromise(instance.provider.bindModel(
        ProviderModelIdSchema.make("test-model"),
        {
          agentId: "agent-1",
          traits: ["careful"],
          preferProvider: "preferred",
        },
      ))

      await Effect.runPromise(model.stream(
        PromptBuilder.empty().user("hello").build(),
        [],
        {},
      ).pipe(Effect.provide(FetchHttpClient.layer)))

      expect(requestBody).toMatchObject({
        magnitude_additional_options: {
          agent_id: "agent-1",
          traits: ["careful"],
          prefer_provider: "preferred",
          session_id: "session-1",
        },
      })
      expect(requestBody).not.toHaveProperty(
        "magnitude_additional_options.forceTrait",
      )
    } finally {
      server.stop(true)
    }
  })

  it("preserves omission of empty bind metadata", async () => {
    let requestBody: unknown
    const server = Bun.serve({
      port: 0,
      fetch: async (request) => {
        requestBody = await request.json()
        return new Response("data: [DONE]\n\n", {
          headers: { "content-type": "text/event-stream" },
        })
      },
    })

    try {
      const instance = createMagnitudeProvider({
        apiKey: "test-key",
        endpoint: `http://127.0.0.1:${server.port}`,
        sessionId: "",
      })
      const model = await Effect.runPromise(instance.provider.bindModel(
        ProviderModelIdSchema.make("test-model"),
        { agentId: "", preferProvider: "" },
      ))

      await Effect.runPromise(model.stream(
        PromptBuilder.empty().user("hello").build(),
        [],
        {},
      ).pipe(Effect.provide(FetchHttpClient.layer)))

      expect(requestBody).toMatchObject({ magnitude_additional_options: {} })
      expect(requestBody).not.toHaveProperty("magnitude_additional_options.agent_id")
      expect(requestBody).not.toHaveProperty("magnitude_additional_options.prefer_provider")
      expect(requestBody).not.toHaveProperty("magnitude_additional_options.session_id")
    } finally {
      server.stop(true)
    }
  })

  it("represents missing authentication through each operation's typed failure channel", async () => {
    const instance = createMagnitudeProvider({ apiKey: " " })

    expect(instance.authentication._tag).toBe("NotConfigured")

    const [catalog, webSearch, usage] = await Effect.runPromise(Effect.all([
      Effect.either(instance.catalog.list),
      Effect.either(instance.provider.webSearch("query")),
      Effect.either(instance.provider.usage()),
    ]).pipe(Effect.provide(FetchHttpClient.layer)))

    expect(catalog).toMatchObject({ _tag: "Left", left: { _tag: "ModelCatalogError", message: "Magnitude authentication is not configured" } })
    expect(webSearch).toMatchObject({ _tag: "Left", left: { _tag: "WebSearchNotConfigured" } })
    expect(usage).toMatchObject({ _tag: "Left", left: { _tag: "MagnitudeClientError", message: "Magnitude authentication is not configured" } })
  })

  it("does not relabel a configured auth applicator defect as missing authentication", async () => {
    const instance = createMagnitudeProvider({
      auth: () => {
        throw new Error("broken auth applicator")
      },
    })

    const exit = await Effect.runPromise(Effect.exit(
      instance.catalog.list.pipe(Effect.provide(FetchHttpClient.layer)),
    ))

    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure") {
      const defects = Chunk.toReadonlyArray(Cause.defects(exit.cause))
      expect(defects).toHaveLength(1)
      expect(defects[0]).toMatchObject({ message: "broken auth applicator" })
    }
  })

  it("applies the web-search deadline while reading the response body", async () => {
    const stalledHttp = HttpClient.make((request) =>
      Effect.succeed(HttpClientResponse.fromWeb(
        request,
        new Response(new ReadableStream()),
      ))
    )
    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const search = yield* Effect.fork(
          Effect.either(
            createMagnitudeProvider({ apiKey: "cloud_test" }).provider.webSearch("magnitude"),
          ),
        )
        yield* Effect.yieldNow()
        yield* TestClock.adjust("10 seconds")
        return yield* Fiber.join(search)
      }).pipe(
        Effect.provideService(HttpClient.HttpClient, stalledHttp),
        Effect.provide(TestContext.TestContext),
      ),
    )

    expect(result).toMatchObject({
      _tag: "Left",
      left: {
        _tag: "WebSearchTimedOut",
        provider: "magnitude",
        timeoutMs: 10_000,
      },
    })
  })
})
