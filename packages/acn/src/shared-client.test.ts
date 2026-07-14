import { describe, expect, it } from "vitest"
import { Effect } from "effect"
import type { MagnitudeStorageShape } from "@magnitudedev/storage"
import type { ProviderAuth } from "@magnitudedev/protocol"
import { resolveEndpointProviderAuthFromStorage } from "./shared-client"

function storageWithAuth(initial?: ProviderAuth) {
  const entries: Record<string, ProviderAuth> = initial ? { local: initial } : {}
  const storage = {
    auth: {
      get: (providerId: string) => Effect.succeed(entries[providerId]),
      set: (providerId: string, auth: ProviderAuth) => Effect.sync(() => {
        entries[providerId] = auth
      }),
    },
  } as unknown as MagnitudeStorageShape
  return { entries, storage }
}

describe("endpoint provider auth resolution", () => {
  it("persists and returns a missing provider's default endpoint", async () => {
    const { entries, storage } = storageWithAuth()

    const resolved = await Effect.runPromise(
      resolveEndpointProviderAuthFromStorage(storage, "local", {
        endpoint: "http://127.0.0.1:8080",
      }),
    )

    expect(resolved).toEqual({ endpoint: "http://127.0.0.1:8080" })
    expect(entries.local).toEqual({
      type: "endpoint",
      endpoint: "http://127.0.0.1:8080",
    })
  })

  it("uses an explicit endpoint instead of the default", async () => {
    const { entries, storage } = storageWithAuth({
      type: "endpoint",
      endpoint: "http://127.0.0.1:9090",
    })

    const resolved = await Effect.runPromise(
      resolveEndpointProviderAuthFromStorage(storage, "local", {
        endpoint: "http://127.0.0.1:8080",
      }),
    )

    expect(resolved).toEqual({ endpoint: "http://127.0.0.1:9090" })
    expect(entries.local).toEqual({
      type: "endpoint",
      endpoint: "http://127.0.0.1:9090",
    })
  })
})
