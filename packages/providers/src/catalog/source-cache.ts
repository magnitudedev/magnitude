import { CatalogCache } from '@magnitudedev/storage'
import { Effect, Option } from 'effect'

export function resolveCachedSource<T>(
  sourceId: string,
  ttlMs: number,
  fetchEffect: Effect.Effect<T, Error>,
): Effect.Effect<T | null, never, CatalogCache> {
  return Effect.gen(function* () {
    const cache = yield* CatalogCache

    const cached = yield* cache.load<T>(sourceId).pipe(
      Effect.catchAll(() => Effect.succeed(null)),
    )
    if (cached && cache.isValid(cached)) return cached.data

    const fresh = yield* fetchEffect.pipe(Effect.option)
    if (Option.isSome(fresh)) {
      yield* cache.save(sourceId, fresh.value, ttlMs).pipe(
        Effect.catchAll(() => Effect.void),
      )
      return fresh.value
    }

    if (cached && cache.isStale(cached)) return cached.data
    return null
  })
}
