import { Effect } from "effect"
import type { ProviderModel } from "../../lib/model/provider-model"
import { getAllProviders } from "../../providers/registry"
import type { CatalogueSource } from "../types"

export const staticCatalogueSource: CatalogueSource = {
  id: "static",
  fetch: () =>
    Effect.sync(() => {
      const result = new Map<string, readonly ProviderModel[]>()

      for (const provider of getAllProviders()) {
        result.set(
          provider.id,
          provider.models.map((model) => ({ ...model })),
        )
      }

      return result
    }),
}
