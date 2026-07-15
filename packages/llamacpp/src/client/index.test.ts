import { describe, expect, it } from "vitest"
import { Effect, Option, Secret } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { makeLlamaCppEndpointClient } from "./index"

const responseClient = (respond: (request: globalThis.Request) => Response): HttpClient.HttpClient =>
  HttpClient.make((request) => Effect.succeed(HttpClientResponse.fromWeb(request, respond(new Request(request.url, {
    method: request.method,
    headers: request.headers,
  })))))

describe("LlamaCppEndpointClient", () => {
  it("validates responses, applies auth, and falls back to versioned props", async () => {
    const requested: string[] = []
    const client = responseClient((request) => {
      requested.push(new URL(request.url).pathname)
      expect(request.headers.get("authorization")).toBe("Bearer test-secret")
      switch (new URL(request.url).pathname) {
        case "/health":
          return Response.json({ status: "ok" })
        case "/props":
          return new Response("", { status: 404 })
        case "/v1/props":
          return Response.json({ build_info: "b10011", default_generation_settings: { n_ctx: 8192 } })
        case "/v1/models":
          return Response.json({ data: [{ id: "model", object: "model", meta: { n_ctx: 8192 } }] })
        default:
          return new Response("", { status: 404 })
      }
    })
    const endpoint = makeLlamaCppEndpointClient({
      baseUrl: "http://127.0.0.1:8080/",
      apiKey: Option.some(Secret.fromString("test-secret")),
    })
    const result = await Effect.runPromise(Effect.all([
      endpoint.health,
      endpoint.props,
      endpoint.models,
    ], { concurrency: 1 }).pipe(Effect.provideService(HttpClient.HttpClient, client)))

    expect(result[0]).toEqual({ _tag: "Ready" })
    expect(result[1].default_generation_settings?.n_ctx).toBe(8192)
    expect(result[2][0]?.id).toBe("model")
    expect(requested).toEqual(["/health", "/props", "/v1/props", "/v1/models"])
  })

  it("rejects malformed model responses", async () => {
    const client = responseClient(() => Response.json({ data: [{ object: "model" }] }))
    const result = await Effect.runPromise(
      makeLlamaCppEndpointClient({ baseUrl: "http://localhost:8080", apiKey: Option.none() }).models.pipe(
        Effect.provideService(HttpClient.HttpClient, client),
        Effect.either,
      ),
    )
    expect(result._tag).toBe("Left")
  })
})
