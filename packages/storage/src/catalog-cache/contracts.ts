import { Context, Effect } from 'effect'

export interface CachedCatalogSourceData<T> {
  readonly _cachedAt: number
  readonly ttlMs: number
  readonly data: T
}

export interface CatalogCacheService {
  readonly load: <T>(sourceId: string) => Effect.Effect<CachedCatalogSourceData<T> | null>
  readonly save: (
    sourceId: string,
    data: unknown,
    ttlMs: number
  ) => Effect.Effect<void>
  readonly isValid: (cached: CachedCatalogSourceData<unknown>) => boolean
  readonly isStale: (cached: CachedCatalogSourceData<unknown>) => boolean
}

export class CatalogCache extends Context.Tag('CatalogCache')<
  CatalogCache,
  CatalogCacheService
>() {}