import { Context, Effect } from "effect"

export interface CachedData<T> {
  readonly _cachedAt: number
  readonly ttlMs: number
  readonly data: T
}

export class CatalogueCache extends Context.Tag("@magnitudedev/ai/CatalogueCache")<
  CatalogueCache,
  {
    readonly load: <T>(sourceId: string) => Effect.Effect<CachedData<T> | null>
    readonly save: (sourceId: string, data: unknown, ttlMs: number) => Effect.Effect<void>
    readonly isValid: (cached: CachedData<unknown>) => boolean
    readonly isStale: (cached: CachedData<unknown>) => boolean
  }
>() {}
