import { describe, expect, it } from "vitest"
import { Effect } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { ModelCatalogError } from "@magnitudedev/ai"
import { makeProviderRegistry } from "./registry"

const unavailableCatalog = {
  list: Effect.fail(new ModelCatalogError({ message: "invalid provider key" })),
  refresh: Effect.fail(new ModelCatalogError({ message: "invalid provider key" })),
  get: () => Effect.fail(new ModelCatalogError({ message: "invalid provider key" })),
}

describe("provider registry status", () => {
  it("reports a configured provider whose catalog cannot be read as unhealthy", async () => {
    const registry = makeProviderRegistry({
      magnitude: null,
      configuredProviders: [{
        provider: {
          id: "cloud",
          displayName: "Cloud",
          catalog: unavailableCatalog,
          bindModel: () => Effect.die("not used"),
        },
        authStatus: { _tag: "authenticated" },
        authKind: "api",
        authSource: "file",
      }],
    })

    const providers = await Effect.runPromise(
      registry.listProviders.pipe(Effect.provide(FetchHttpClient.layer)),
    )

    expect(providers).toEqual([{
      id: "cloud",
      displayName: "Cloud",
      authStatus: { _tag: "authenticated" },
      authKind: "api",
      authSource: "file",
      status: "error",
      message: "invalid provider key",
    }])
  })
})
