import { describe, expect, it } from "vitest"
import { FetchHttpClient, HttpClient } from "@effect/platform"
import { Effect, Layer } from "effect"
import { icnHttpClient } from "./layer"

describe("icnHttpClient", () => {
  it("sends ICN requests without Bun's five-minute fetch deadline", async () => {
    const inits: RequestInit[] = []
    const fetch = ((_: string | URL | Request, init?: RequestInit) => {
      inits.push(init ?? {})
      return Promise.resolve(new Response("{}"))
    }) as typeof globalThis.fetch
    await Effect.runPromise(
      HttpClient.get("http://127.0.0.1:1/api/v1/candidates/inspect").pipe(
        Effect.provide(icnHttpClient.pipe(Layer.provide(Layer.succeed(FetchHttpClient.Fetch, fetch)))),
      ),
    )
    expect(inits).toHaveLength(1)
    expect(inits[0]).toMatchObject({ timeout: false })
  })
})
