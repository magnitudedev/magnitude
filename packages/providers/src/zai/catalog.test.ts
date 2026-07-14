import { afterEach, describe, expect, it } from "vitest"
import { FetchHttpClient } from "@effect/platform"
import { Effect } from "effect"
import type { ModelsDevClient } from "../catalog/models-dev"
import { createZaiCatalog } from "./catalog"

const servers: Bun.Server<unknown>[] = []

afterEach(() => {
  for (const server of servers.splice(0)) server.stop(true)
})

describe("Z.AI catalog", () => {
  it("exposes the semantic GLM-5.2 reasoning controls, including disabled", async () => {
    const server = Bun.serve({
      port: 0,
      fetch: () => Response.json({ data: [{ id: "glm-5.2" }] }),
    })
    servers.push(server)

    const provider = {
        id: "zai",
        name: "Z.AI",
        api: "https://api.z.ai/api/paas/v4",
        models: {
          "glm-5.2": {
            id: "glm-5.2",
            name: "GLM-5.2",
            attachment: false,
            reasoning: true,
            tool_call: true,
            structured_output: true,
            open_weights: true,
            reasoning_options: [{ type: "effort", values: ["high", "max"] }],
            modalities: { input: ["text"], output: ["text"] },
            limit: { context: 204_800, output: 131_072 },
            cost: { input: 1, output: 1 },
          },
        },
      } as const
    const modelsDev: ModelsDevClient = {
      getProvider: () => Effect.succeed(provider),
      refresh: Effect.succeed({ zai: provider }),
    }

    const catalog = createZaiCatalog({
      endpoint: server.url.toString().replace(/\/$/, ""),
      auth: () => {},
      modelsDev,
    })
    const result = await Effect.runPromise(catalog.list.pipe(Effect.provide(FetchHttpClient.layer)))

    expect(result[0]?.reasoningEfforts).toEqual(["none", "high", "max"])
  })
})
