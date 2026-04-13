import { Layer } from 'effect'

import { CatalogSourceRegistry } from './contracts'
import { localCatalogSource } from './local-catalog-source'
import { modelsDevCatalogSource } from './models-dev-catalog-source'
import { openRouterCatalogSource } from './openrouter-catalog-source'
import { staticCatalogSource } from './static-catalog-source'

export const CatalogSourceRegistryLive = Layer.succeed(CatalogSourceRegistry, {
  list: () => [
    staticCatalogSource,
    modelsDevCatalogSource,
    localCatalogSource,
    openRouterCatalogSource,
  ],
})
