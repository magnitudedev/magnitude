import { FetchHttpClient } from "@effect/platform"
import { Cause, Chunk, Effect } from "effect"
import { describe, expect, it } from "vitest"
import { createMagnitudeProvider } from "./provider"

describe("Magnitude provider authentication", () => {
  it("represents missing authentication through each operation's typed failure channel", async () => {
    const instance = createMagnitudeProvider({ apiKey: " " })

    expect(instance.authentication._tag).toBe("NotConfigured")

    const [catalog, webSearch, usage] = await Effect.runPromise(Effect.all([
      Effect.either(instance.catalog.list),
      Effect.either(instance.provider.webSearch("query")),
      Effect.either(instance.provider.usage()),
    ]).pipe(Effect.provide(FetchHttpClient.layer)))

    expect(catalog).toMatchObject({ _tag: "Left", left: { _tag: "ModelCatalogError", message: "Magnitude authentication is not configured" } })
    expect(webSearch).toMatchObject({ _tag: "Left", left: { _tag: "WebSearchError", message: "Magnitude authentication is not configured" } })
    expect(usage).toMatchObject({ _tag: "Left", left: { _tag: "MagnitudeClientError", message: "Magnitude authentication is not configured" } })
  })

  it("does not relabel a configured auth applicator defect as missing authentication", async () => {
    const instance = createMagnitudeProvider({
      auth: () => {
        throw new Error("broken auth applicator")
      },
    })

    const exit = await Effect.runPromise(Effect.exit(
      instance.catalog.list.pipe(Effect.provide(FetchHttpClient.layer)),
    ))

    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure") {
      const defects = Chunk.toReadonlyArray(Cause.defects(exit.cause))
      expect(defects).toHaveLength(1)
      expect(defects[0]).toMatchObject({ message: "broken auth applicator" })
    }
  })
})
