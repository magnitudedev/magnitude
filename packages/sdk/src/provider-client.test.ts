import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { createProviderClient } from "./provider-client"

describe("provider client web-search routing", () => {
  it.each([
    { cloud: "cloud-key", exa: "exa-key", expected: "magnitude" },
    { cloud: "cloud-key", exa: " ", expected: "magnitude" },
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
