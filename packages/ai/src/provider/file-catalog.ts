import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { BunFileSystem, BunPath } from "@effect/platform-bun"
import { Effect, Layer, Schema } from "effect"

import { ModelCatalogError, type ModelCatalog } from "./catalog"
import type { ProviderModel } from "./model"

// =============================================================================
// File cache format
// =============================================================================

const ModelCacheFileSchema = Schema.Struct({
  fetchedAt: Schema.Number,
  ttlMs: Schema.Number,
  models: Schema.Array(Schema.Any),
})

interface ModelCacheFile {
  readonly fetchedAt: number
  readonly ttlMs: number
  readonly models: readonly ProviderModel[]
}

// Layer providing both FileSystem and Path with no further requirements.
const PlatformFileLayer: Layer.Layer<FileSystem.FileSystem | Path.Path> = Layer.merge(
  BunFileSystem.layer,
  BunPath.layer,
)

// =============================================================================
// Factory
// =============================================================================

/**
 * Wraps an inner `ModelCatalog` with a two-layer cache: in-memory (fast,
 * per-process, short TTL) and file-backed (shared across processes/restarts,
 * longer TTL). Uses closure-captured variables — never `this` in Effect.gen.
 *
 * Cache hierarchy on `list`:
 * 1. In-memory (if fresh) → return immediately
 * 2. File cache (if fresh) → populate in-memory, return
 * 3. Fetch from provider via `inner.refresh` → write both, return
 *
 * `refresh` always bypasses both caches, fetches from the provider, and writes
 * to both.
 *
 * File writes are atomic (write-to-temp-then-rename) to prevent partial reads
 * from concurrent processes.
 *
 * The `FileSystem` and `Path` services are provided internally via
 * `@effect/platform-bun`, so the returned catalog has the same context
 * requirement (`HttpClient`) as the inner catalog.
 */
export function makeFileBackedModelCatalog<T extends ProviderModel>(
  inner: ModelCatalog<T>,
  cachePath: string,
  fileTtlMs: number = 10 * 60 * 1000,
  inMemoryTtlMs: number = 5 * 60 * 1000,
  options?: {
    /** Keep stale entries for still-configured providers omitted by a partial refresh. */
    readonly preserveProviderIds?: readonly string[]
  },
): ModelCatalog<T> {
  let inMemory: readonly T[] | null = null
  let inMemoryFetchedAt = 0

  const readFileCache: Effect.Effect<{ fetchedAt: number; ttlMs: number; models: readonly T[] } | null, never, never> =
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const raw = yield* fs.readFileString(cachePath).pipe(
        Effect.catchAll(() => Effect.succeed(null)),
      )
      if (raw === null) return null
      const parsed = yield* Schema.decodeUnknown(Schema.parseJson(ModelCacheFileSchema))(raw).pipe(
        Effect.catchAll(() => Effect.succeed(null)),
      )
      if (parsed === null) return null
      return {
        fetchedAt: parsed.fetchedAt,
        ttlMs: parsed.ttlMs,
        models: parsed.models as readonly T[],
      }
    }).pipe(Effect.provide(PlatformFileLayer))

  const writeFileCache = (
    models: readonly T[],
  ): Effect.Effect<void, never, never> =>
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const dir = path.dirname(cachePath)
      yield* fs.makeDirectory(dir, { recursive: true }).pipe(
        Effect.catchAll(() => Effect.void),
      )
      const tmpPath = `${cachePath}.${process.pid}.${Math.random().toString(36).slice(2)}.tmp`
      const payload: ModelCacheFile = {
        fetchedAt: Date.now(),
        ttlMs: fileTtlMs,
        models,
      }
      const content = yield* Schema.encodeUnknown(Schema.parseJson({ space: 2 }))(payload).pipe(
        Effect.catchAll(() => Effect.succeed("{}")),
      )
      yield* fs.writeFileString(tmpPath, content).pipe(
        Effect.catchAll(() => Effect.void),
      )
      yield* fs.rename(tmpPath, cachePath).pipe(
        Effect.catchAll(() =>
          fs.remove(tmpPath, { force: true }).pipe(
            Effect.catchAll(() => Effect.void),
            Effect.andThen(Effect.void),
          ),
        ),
      )
    }).pipe(Effect.provide(PlatformFileLayer))

  const refresh: ModelCatalog<T>["refresh"] = Effect.gen(function* () {
    const previous = inMemory ?? (yield* readFileCache)?.models ?? []
    const refreshed: readonly T[] = yield* inner.refresh
    const refreshedProviderIds = new Set(refreshed.map((model) => model.providerId))
    const preserveProviderIds = new Set(options?.preserveProviderIds ?? [])
    const preserved = previous.filter((model) =>
      preserveProviderIds.has(model.providerId)
      && !refreshedProviderIds.has(model.providerId)
    )
    const models = [...refreshed, ...preserved]
    inMemory = models
    inMemoryFetchedAt = Date.now()
    yield* writeFileCache(models)
    return models
  })

  const list: ModelCatalog<T>["list"] = Effect.gen(function* () {
    // 1. Check in-memory
    if (inMemory && Date.now() - inMemoryFetchedAt < inMemoryTtlMs) {
      return inMemory
    }
    // 2. Check file cache
    const fileCache = yield* readFileCache
    if (fileCache && Date.now() - fileCache.fetchedAt < fileTtlMs) {
      inMemory = fileCache.models
      inMemoryFetchedAt = Date.now()
      return fileCache.models
    }
    // 3. Fetch from provider
    return yield* refresh
  })

  const get: ModelCatalog<T>["get"] = (providerId, providerModelId) =>
    Effect.gen(function* () {
      const models = yield* list
      const model = models.find(
        (m) => m.providerId === providerId && m.providerModelId === providerModelId,
      )
      if (!model) {
        return yield* new ModelCatalogError({
          message: `Model not found: ${providerId}/${providerModelId}`,
        })
      }
      return model
    })

  return { list, get, refresh }
}
