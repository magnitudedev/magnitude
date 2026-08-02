import { FetchHttpClient } from "@effect/platform"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { Effect, Fiber, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { createExaWebSearch } from "./web-search"

describe("Exa web search", () => {
  it("sends the supported request and maps the response contract", async () => {
    let captured: {
      readonly apiKey: string | null
      readonly body: unknown
    } | null = null
    const server = Bun.serve({
      port: 0,
      fetch: async (request) => {
        captured = {
          apiKey: request.headers.get("x-api-key"),
          body: await request.json(),
        }
        return Response.json({
          requestId: "request-1",
          results: [{
            id: "https://magnitude.dev",
            title: "Magnitude",
            url: "https://magnitude.dev",
            highlights: ["AI coding agents"],
            ignored: "extra fields are allowed",
          }],
          output: {
            content: { answer: 42 },
            grounding: [{
              field: "answer",
              citations: [{
                url: "https://magnitude.dev",
                title: "Magnitude",
              }],
              confidence: "high",
            }],
          },
          costDollars: {
            total: 0.01,
            search: { neural: 0.01 },
          },
          resolvedSearchType: "",
          searchTime: 123.4,
        })
      },
    })

    try {
      const instance = createExaWebSearch({
        apiKey: "exa_test",
        endpoint: `http://127.0.0.1:${server.port}/search`,
      })
      const result = await Effect.runPromise(
        instance.webSearch("magnitude", {
          type: "object",
          properties: { answer: { type: "number" } },
        }).pipe(Effect.provide(FetchHttpClient.layer)),
      )

      expect(captured).toEqual({
        apiKey: "exa_test",
        body: {
          query: "magnitude",
          type: "auto",
          numResults: 10,
          contents: { highlights: true },
          outputSchema: {
            type: "object",
            properties: { answer: { type: "number" } },
          },
        },
      })
      expect(result).toEqual({
        text: "## Magnitude\nAI coding agents",
        sources: [{ title: "Magnitude", url: "https://magnitude.dev" }],
        data: { answer: 42 },
      })
    } finally {
      server.stop(true)
    }
  })

  it("returns an explicit rejected-response error", async () => {
    const server = Bun.serve({
      port: 0,
      fetch: () => Response.json(
        { error: "rate limit exceeded" },
        { status: 429 },
      ),
    })

    try {
      const instance = createExaWebSearch({
        apiKey: "exa_test",
        endpoint: `http://127.0.0.1:${server.port}/search`,
      })
      const result = await Effect.runPromise(
        Effect.either(instance.webSearch("magnitude")).pipe(
          Effect.provide(FetchHttpClient.layer),
        ),
      )

      expect(result).toMatchObject({
        _tag: "Left",
        left: {
          _tag: "WebSearchRejected",
          provider: "exa",
          status: 429,
          message: "rate limit exceeded",
        },
      })
    } finally {
      server.stop(true)
    }
  })

  it("reports missing configuration without a generic cause envelope", async () => {
    const result = await Effect.runPromise(
      Effect.either(createExaWebSearch({ apiKey: " " }).webSearch("magnitude")).pipe(
        Effect.provide(FetchHttpClient.layer),
      ),
    )

    expect(result).toMatchObject({
      _tag: "Left",
      left: expect.objectContaining({ _tag: "WebSearchNotConfigured" }),
    })
    if (result._tag === "Left") {
      expect("cause" in result.left).toBe(false)
    }
  })

  it("applies the request deadline while reading the response body", async () => {
    const stalledHttp = HttpClient.make((request) =>
      Effect.succeed(HttpClientResponse.fromWeb(
        request,
        new Response(new ReadableStream()),
      ))
    )
    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const search = yield* Effect.fork(
          Effect.either(createExaWebSearch({ apiKey: "exa_test" }).webSearch("magnitude")),
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
        provider: "exa",
        timeoutMs: 10_000,
      },
    })
  })
})
