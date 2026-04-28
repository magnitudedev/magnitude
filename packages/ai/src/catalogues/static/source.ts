import { Effect } from "effect"
import { getAllProviders } from "../../providers/registry"
import type { CatalogueSource } from "../types"

export const staticCatalogueSource: CatalogueSource = {
  id: "static",
  fetch: Effect.sync(() =>
    getAllProviders().flatMap((provider) => provider.models.map((model) => ({ ...model }))),
  ),
}
