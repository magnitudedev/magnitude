import { afterEach, describe, expect, it } from "vitest"
import { FetchHttpClient } from "@effect/platform"
import { Effect } from "effect"
import { createModelsDevClient } from "./models-dev"

let server: Bun.Server<unknown> | null = null

afterEach(() => {
  server?.stop(true)
  server = null
})

const snapshot = {
  router: {
    id: "router",
    name: "Router",
    models: {
      "org/model:free": {
        id: "org/model:free",
        name: "Model",
        attachment: false,
        reasoning: true,
        tool_call: true,
        structured_output: true,
        open_weights: true,
        modalities: { input: ["text"], output: ["text"] },
        limit: { context: 128000, output: 16000 },
        cost: { input: 1, output: 2 },
      },
    },
  },
}

describe("models.dev client", () => {
  it("uses the last validated snapshot when refresh returns invalid data", async () => {
    let valid = true
    server = Bun.serve({
      port: 0,
      fetch: () => Response.json(valid ? snapshot : { invalid: true }),
    })
    const client = createModelsDevClient({ endpoint: server.url.toString(), ttlMs: 0 })

    const first = await Effect.runPromise(client.refresh.pipe(Effect.provide(FetchHttpClient.layer)))
    valid = false
    const stale = await Effect.runPromise(client.refresh.pipe(Effect.provide(FetchHttpClient.layer)))

    expect(first.router?.models["org/model:free"]?.open_weights).toBe(true)
    expect(stale).toEqual(first)
  })
})
