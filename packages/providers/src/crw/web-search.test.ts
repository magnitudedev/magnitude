import { FetchHttpClient } from "@effect/platform"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { Effect, Fiber, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { createCrwWebSearch } from "./web-search"

describe("fastCRW web search", () => {
  it("sends the supported request and maps the response contract", async () => {
    let captured: {
      readonly authorization: string | null
      readonly body: unknown
    } | null = null
    const server = Bun.serve({
      port: 0,
      fetch: async (request) => {
        captured = {
          authorization: request.headers.get("authorization"),
          body: await request.json(),
        }
        return Response.json({
          success: true,
          data: [{
            url: "https://magnitude.dev",
            title: "Magnitude",
            description: "AI coding agents",
            snippet: "AI coding agents",
            position: 1,
            score: 1,
            category: "general",
            ignored: "extra fields are allowed",
          }],
        })
      },
    })

    try {
      const instance = createCrwWebSearch({
        apiKey: "crw_test",
        baseUrl: `http://127.0.0.1:${server.port}`,
      })
      const result = await Effect.runPromise(
        instance.webSearch("magnitude").pipe(Effect.provide(FetchHttpClient.layer)),
      )

      expect(captured).toEqual({
        authorization: "Bearer crw_test",
        body: { query: "magnitude", limit: 10 },
      })
      expect(result).toEqual({
        text: "## Magnitude\nAI coding agents",
        sources: [{ title: "Magnitude", url: "https://magnitude.dev" }],
      })
    } finally {
      server.stop(true)
    }
  })

  it("accepts the self-hosted payload shape and sends no authorization header", async () => {
    let authorization: string | null | undefined
    const server = Bun.serve({
      port: 0,
      fetch: (request) => {
        authorization = request.headers.get("authorization")
        return Response.json({
          success: true,
          data: {
            results: [{ url: "https://example.com", snippet: "self hosted" }],
          },
        })
      },
    })

    try {
      const instance = createCrwWebSearch({
        apiKey: " ",
        baseUrl: `http://127.0.0.1:${server.port}`,
      })
      const result = await Effect.runPromise(
        instance.webSearch("magnitude").pipe(Effect.provide(FetchHttpClient.layer)),
      )

      expect(instance.configured).toBe(true)
      expect(authorization).toBeNull()
      expect(result).toEqual({
        text: "## https://example.com\nself hosted",
        sources: [{ title: "https://example.com", url: "https://example.com" }],
      })
    } finally {
      server.stop(true)
    }
  })

  it("treats a success:false envelope as a rejection", async () => {
    const server = Bun.serve({
      port: 0,
      fetch: () => Response.json({ success: false, error: "search failed upstream" }),
    })

    try {
      const result = await Effect.runPromise(
        Effect.either(
          createCrwWebSearch({
            apiKey: "crw_test",
            baseUrl: `http://127.0.0.1:${server.port}`,
          }).webSearch("magnitude"),
        ).pipe(Effect.provide(FetchHttpClient.layer)),
      )

      expect(result).toMatchObject({
        _tag: "Left",
        left: {
          _tag: "WebSearchRejected",
          provider: "crw",
          message: "search failed upstream",
        },
      })
    } finally {
      server.stop(true)
    }
  })

  it("returns an explicit rejected-response error", async () => {
    const server = Bun.serve({
      port: 0,
      fetch: () => Response.json({ error: "rate limit exceeded" }, { status: 429 }),
    })

    try {
      const result = await Effect.runPromise(
        Effect.either(
          createCrwWebSearch({
            apiKey: "crw_test",
            baseUrl: `http://127.0.0.1:${server.port}`,
          }).webSearch("magnitude"),
        ).pipe(Effect.provide(FetchHttpClient.layer)),
      )

      expect(result).toMatchObject({
        _tag: "Left",
        left: {
          _tag: "WebSearchRejected",
          provider: "crw",
          status: 429,
          message: "rate limit exceeded",
        },
      })
    } finally {
      server.stop(true)
    }
  })

  it("rejects a structured-output request instead of silently dropping the schema", async () => {
    let called = false
    const server = Bun.serve({
      port: 0,
      fetch: () => {
        called = true
        return Response.json({ success: true, data: [] })
      },
    })

    try {
      const result = await Effect.runPromise(
        Effect.either(
          createCrwWebSearch({
            apiKey: "crw_test",
            baseUrl: `http://127.0.0.1:${server.port}`,
          }).webSearch("magnitude", { type: "object" }),
        ).pipe(Effect.provide(FetchHttpClient.layer)),
      )

      expect(called).toBe(false)
      expect(result).toMatchObject({
        _tag: "Left",
        left: { _tag: "WebSearchStructuredOutputUnsupported", provider: "crw" },
      })
    } finally {
      server.stop(true)
    }
  })

  it("reports an empty result set as an empty search, not an error", async () => {
    const server = Bun.serve({
      port: 0,
      fetch: () => Response.json({ success: true, data: [] }),
    })

    try {
      const result = await Effect.runPromise(
        createCrwWebSearch({
          apiKey: "crw_test",
          baseUrl: `http://127.0.0.1:${server.port}`,
        }).webSearch("magnitude").pipe(Effect.provide(FetchHttpClient.layer)),
      )

      expect(result).toEqual({ text: "", sources: [] })
    } finally {
      server.stop(true)
    }
  })

  it("surfaces a response with no results payload instead of reporting no results", async () => {
    const server = Bun.serve({
      port: 0,
      fetch: () => Response.json({ note: "not a search response" }),
    })

    try {
      const result = await Effect.runPromise(
        Effect.either(
          createCrwWebSearch({
            apiKey: "crw_test",
            baseUrl: `http://127.0.0.1:${server.port}`,
          }).webSearch("magnitude"),
        ).pipe(Effect.provide(FetchHttpClient.layer)),
      )

      expect(result).toMatchObject({
        _tag: "Left",
        left: { _tag: "WebSearchInvalidResponse", provider: "crw" },
      })
    } finally {
      server.stop(true)
    }
  })

  it("reports missing configuration without a generic cause envelope", async () => {
    const instance = createCrwWebSearch({ apiKey: " ", baseUrl: " " })
    const result = await Effect.runPromise(
      Effect.either(instance.webSearch("magnitude")).pipe(
        Effect.provide(FetchHttpClient.layer),
      ),
    )

    expect(instance.configured).toBe(false)
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
          Effect.either(
            createCrwWebSearch({ apiKey: "crw_test" }).webSearch("magnitude"),
          ),
        )
        yield* Effect.yieldNow()
        yield* TestClock.adjust("30 seconds")
        return yield* Fiber.join(search)
      }).pipe(
        Effect.provideService(HttpClient.HttpClient, stalledHttp),
        Effect.provide(TestContext.TestContext),
      ),
    )

    expect(result).toMatchObject({
      _tag: "Left",
      left: { _tag: "WebSearchTimedOut", provider: "crw", timeoutMs: 30_000 },
    })
  })
})
