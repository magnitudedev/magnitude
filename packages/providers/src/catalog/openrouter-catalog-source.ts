import { Effect } from 'effect'
import { OPENROUTER_TTL_MS } from '@magnitudedev/storage'

import type { CatalogSource } from './contracts'
import { fetchOpenRouterModels, normalizeOpenRouterModels } from './openrouter-source'
import { resolveCachedSource } from './source-cache'

export const openRouterCatalogSource: CatalogSource = {
  id: 'openrouter-api',
  priority: 300,
  supports: (provider) => provider.id === 'openrouter',
  refresh: () =>
    resolveCachedSource(
      'openrouter',
      OPENROUTER_TTL_MS,
      fetchOpenRouterModels,
    ).pipe(
      Effect.map((data) => (data ? normalizeOpenRouterModels('openrouter', 'OpenRouter', data) : [])),
    ),
}
