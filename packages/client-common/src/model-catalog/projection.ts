import {
  ProviderModelCatalogDegraded,
  ProviderModelCatalogLoading,
  ProviderModelCatalogReady,
  ProviderModelCatalogRefreshing,
  type LocalModelsState,
  type ModelCatalogState,
  type ProviderModelCatalogState,
} from "@magnitudedev/sdk"

export const localModelsFromCatalog = (catalog: ModelCatalogState): LocalModelsState =>
  catalog._tag === "Initializing"
    ? {
        preparation: {
          discovery: { complete: false, modelsFound: 0 },
          assessment: { complete: false, settledModels: 0, totalModels: 0 },
        },
        models: [],
      }
    : {
        preparation: catalog.localModelPreparation,
        models: catalog.models.flatMap((entry) => entry._tag === "Local" ? [entry.product] : []),
      }

export const providerCatalogFromCatalog = (
  catalog: ModelCatalogState,
): ProviderModelCatalogState => {
  if (catalog._tag === "Initializing") return new ProviderModelCatalogLoading({})
  const models = catalog.models.flatMap((entry) => entry._tag === "Remote"
    ? [entry.offering]
    : entry.offering._tag === "Some" ? [entry.offering.value] : [])
  switch (catalog._tag) {
    case "Ready": return new ProviderModelCatalogReady({ providers: catalog.providers, models })
    case "Refreshing": return new ProviderModelCatalogRefreshing({
      providers: catalog.providers,
      models,
      failures: catalog.failures,
    })
    case "Degraded": return new ProviderModelCatalogDegraded({
      providers: catalog.providers,
      models,
      failures: catalog.failures,
    })
  }
}
