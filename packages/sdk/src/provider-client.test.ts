import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { createProviderClient } from "./provider-client"

describe("provider client web-search routing", () => {
  // Every case pins all sources explicitly: each provider falls back to its own
  // environment variable, so an unpinned field would make the result depend on
  // the ambient environment.
  it.each([
    { cloud: " ", exa: "exa-key", crwKey: " ", crwUrl: " ", expected: "exa" },
    { cloud: " ", exa: " ", crwKey: " ", crwUrl: " ", expected: "unavailable" },
    { cloud: " ", exa: " ", crwKey: "crw-key", crwUrl: " ", expected: "crw" },
    { cloud: " ", exa: " ", crwKey: " ", crwUrl: "http://localhost:3000", expected: "crw" },
    { cloud: " ", exa: "exa-key", crwKey: "crw-key", crwUrl: " ", expected: "exa" },
  ] as const)(
    "selects $expected for exa=$exa crwKey=$crwKey crwUrl=$crwUrl",
    async ({ cloud, exa, crwKey, crwUrl, expected }) => {
      const client = createProviderClient({
        apiKey: cloud,
        exaApiKey: exa,
        crwApiKey: crwKey,
        crwBaseUrl: crwUrl,
      })

      await expect(Effect.runPromise(client.webSearchSource)).resolves.toBe(expected)
    },
  )
})
