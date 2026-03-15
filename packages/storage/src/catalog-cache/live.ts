import { Effect, Layer } from 'effect'

import { CatalogCache } from './contracts'
import {
  isCatalogCacheStale,
  isCatalogCacheValid,
  loadCatalogCache,
  saveCatalogCache,
} from './storage'
import { GlobalStorage } from '../services'

export const CatalogCacheLive = Layer.effect(
  CatalogCache,
  Effect.gen(function* () {
    const globalStorage = yield* GlobalStorage

    return CatalogCache.of({
      load: <T>(sourceId: string) =>
        Effect.promise(() => loadCatalogCache<T>(globalStorage, sourceId)),
      save: (sourceId: string, data: unknown, ttlMs: number) =>
        Effect.promise(() => saveCatalogCache(globalStorage, sourceId, data, ttlMs)),
      isValid: isCatalogCacheValid,
      isStale: isCatalogCacheStale,
    })
  })
)