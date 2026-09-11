import { FetchHttpClient } from "@effect/platform"
import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { createProviderClient } from "./provider-client"

describe("provider client web-search routing", () => {
  it.each([
    { cloud: " ", exa: "exa-key", expected: "exa" },
    { cloud: " ", exa: " ", expected: "unavailable" },
  ] as const)(
    "selects $expected for cloud=$cloud and exa=$exa",
    async ({ cloud, exa, expected }) => {
      const client = createProviderClient({
        apiKey: cloud,
        exaApiKey: exa,
      })

      await expect(Effect.runPromise(client.webSearchSource)).resolves.toBe(expected)
    },
  )
})

describe("provider client built-in providers", () => {
  it("registers MiniMax with its current catalog", async () => {
    const client = createProviderClient({ miniMax: { apiKey: "test-key" } })
    const [providers, models] = await Effect.runPromise(Effect.all([
      client.listProviders,
      client.catalog.list,
    ]).pipe(Effect.provide(FetchHttpClient.layer)))

    expect(providers).toContainEqual(expect.objectContaining({
      id: "minimax",
      displayName: "MiniMax",
      kind: "Hosted",
      authStatus: { _tag: "authenticated" },
    }))
    expect(models.map((model) => model.providerModelId)).toEqual([
      "MiniMax-M3",
      "MiniMax-M2.7",
    ])
  })
})
