import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { canonicalModelId } from "./connections-runtime"
import { CliModelsConnection, type CliModelsClient } from "./inference-client"
import { catalogState, makeInstalledCatalogModel } from "./fixtures/inference"

const alias = ProviderModelIdSchema.make("hf:unsloth/Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-UD-Q4_K_XL.gguf")

const resolve = (connect: Effect.Effect<CliModelsClient, unknown>, modelId = alias) =>
  Effect.runPromise(canonicalModelId(modelId).pipe(Effect.provideService(CliModelsConnection, { connect })))

const serving = (aliases: readonly string[]) => Effect.succeed({ models: {
  getCatalog: () => Effect.succeed(catalogState([makeInstalledCatalogModel({
    catalogData: { ...makeInstalledCatalogModel().catalogData, aliases: aliases as never },
  })])),
} } as unknown as CliModelsClient)

describe("connections connect --set-model with an ID from before an upgrade", () => {
  it("selects the catalog model an old hf: ID is now an alias of", async () => {
    expect(await resolve(serving([alias]))).toBe(makeInstalledCatalogModel().modelId)
  })

  it("keeps an ID that isn't an alias as typed", async () => {
    expect(await resolve(serving([]))).toBe(alias)
  })

  it("keeps the ID as typed when the service can't be asked", async () => {
    expect(await resolve(Effect.fail("no service"))).toBe(alias)
  })
})
