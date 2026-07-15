import { Context, Effect, Layer, Stream } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import * as HttpClient from "@effect/platform/HttpClient"
import { BunFileSystem, BunPath } from "@effect/platform-bun"
import { FetchHttpClient } from "@effect/platform"
import {
  LlamaCppModelDownloadFailed,
  LlamaCppGatedModelAccessDenied,
  LlamaCppHfTokenMissing,
} from "../errors"
import { hfCacheDir, scanHfCache } from "./hf-cache"
import { scanDirectory } from "./scan"
import type {
  LocalModelInfo,
  DiscoverOptions,
  DownloadModelParams,
  DownloadModelResult,
  DownloadProgress,
  DownloadState,
  RepoGgufFile,
} from "./types"
import {
  makeDownloadRegistry,
  type DownloadRegistry,
} from "./download-registry"
import {
  downloadModelStream,
  cancelModelDownload,
  listRepoGgufFiles,
} from "./download"

// ── Service Tag ──

export interface LlamaCppModelStoreApi {
  /**
   * Discover all locally available models on disk.
   * Scans HF cache and user-configured directories.
   * Fresh scan on every call (metadata reading is stat-cached).
   */
  readonly discover: (options?: DiscoverOptions) => Effect.Effect<
    readonly LocalModelInfo[],
    never
  >

  /** Get a single model by ID. Returns null if not found. */
  readonly get: (modelId: string) => Effect.Effect<LocalModelInfo | null, never>

  /**
   * Delete a model from disk. Removes the GGUF file(s) and any paired mmproj.
   * No-op if the model is not found. Does not stop running instances.
   */
  readonly deleteModel: (modelId: string) => Effect.Effect<void, never>

  /** List available GGUF files in a HuggingFace repo (for browsing quants). */
  readonly listRepoFiles: (repo: string) => Effect.Effect<
    readonly RepoGgufFile[],
    LlamaCppModelDownloadFailed | LlamaCppGatedModelAccessDenied | LlamaCppHfTokenMissing
  >

  /**
   * Download a GGUF file from HF. Returns a Stream of progress events.
   * The final event carries the DownloadModelResult.
   * Resumable: if a .incomplete partial file exists, resumes from that offset.
   * Idempotent: if already fully cached and valid, stream emits a single completion event.
   */
  readonly download: (params: DownloadModelParams) => Stream.Stream<
    DownloadProgress | DownloadModelResult,
    LlamaCppModelDownloadFailed | LlamaCppGatedModelAccessDenied | LlamaCppHfTokenMissing
  >

  /** Cancel an active or paused download. Deletes the .incomplete partial file. */
  readonly cancelDownload: (params: DownloadModelParams) => Effect.Effect<void, never>

  /** List all active, paused, completed, and failed downloads. */
  readonly listDownloads: () => Effect.Effect<readonly DownloadState[], never>

  /** Get the state of a specific download by ID. */
  readonly getDownloadState: (downloadId: string) => Effect.Effect<DownloadState | null, never>
}

export class LlamaCppModelStore extends Context.Tag("LlamaCppModelStore")<
  LlamaCppModelStore,
  LlamaCppModelStoreApi
>() {}

// ── Platform layer (baked in) ──

const PlatformLayer = Layer.provideMerge(
  Layer.mergeAll(BunPath.layer, FetchHttpClient.layer),
  BunFileSystem.layer,
)

// ── Factory ──

export interface LlamaCppModelStoreDeps {
  /** User-configured extra directories to scan for models. */
  readonly extraDirs?: readonly string[]
  /** Stored HF token from MagnitudeConfig. */
  readonly hfToken?: string
}

export function makeLlamaCppModelStore(
  deps: LlamaCppModelStoreDeps = {},
): LlamaCppModelStoreApi {
  const registry: DownloadRegistry = makeDownloadRegistry()

  const discover: LlamaCppModelStoreApi["discover"] = (options) =>
    discoverModels({ ...deps, extraDirs: options?.extraDirs ?? deps.extraDirs }).pipe(
      Effect.provide(PlatformLayer),
    )

  const get: LlamaCppModelStoreApi["get"] = (modelId) =>
    Effect.gen(function* () {
      const models = yield* discover()
      return models.find((m) => m.id === modelId) ?? null
    })

  const deleteModel: LlamaCppModelStoreApi["deleteModel"] = (modelId) =>
    deleteModelFromDisk(modelId).pipe(
      Effect.provide(PlatformLayer),
    )

  const listRepoFiles: LlamaCppModelStoreApi["listRepoFiles"] = (repo) =>
    listRepoGgufFiles(repo).pipe(Effect.provide(PlatformLayer))

  const download: LlamaCppModelStoreApi["download"] = (params) =>
    downloadModelStream(params, registry).pipe(
      Stream.provideLayer(PlatformLayer) as <A, E, R>(s: Stream.Stream<A, E, R>) => Stream.Stream<A, E, never>,
    )

  const cancelDownload: LlamaCppModelStoreApi["cancelDownload"] = (params) =>
    cancelModelDownload(params, registry).pipe(
      Effect.provide(PlatformLayer),
    )

  const listDownloads: LlamaCppModelStoreApi["listDownloads"] = () => registry.list()

  const getDownloadState: LlamaCppModelStoreApi["getDownloadState"] = (downloadId) =>
    registry.get(downloadId)

  return { discover, get, deleteModel, listRepoFiles, download, cancelDownload, listDownloads, getDownloadState }
}

// ── Discovery implementation ──

function discoverModels(
  deps: LlamaCppModelStoreDeps,
): Effect.Effect<
  readonly LocalModelInfo[],
  never,
  FileSystem.FileSystem | Path.Path | HttpClient.HttpClient
> {
  return Effect.gen(function* () {
    // Scan function for a directory — produces user-dir source models
    const scanDir = (dir: string) =>
      scanDirectory(dir, { _tag: "user-dir" as const, dir })

    // 1. Scan HF Hub cache
    const cacheDir = hfCacheDir()
    const hfModels = yield* scanHfCache(cacheDir, scanDir)

    // 2. Scan user-configured directories
    const userModels: LocalModelInfo[] = []
    if (deps.extraDirs) {
      for (const dir of deps.extraDirs) {
        const models = yield* scanDirectory(dir, { _tag: "user-dir" as const, dir })
        userModels.push(...models)
      }
    }

    // 3. Merge & deduplicate by ID (HF cache takes precedence)
    const seen = new Set<string>()
    const merged: LocalModelInfo[] = []

    for (const model of hfModels) {
      if (!seen.has(model.id)) {
        seen.add(model.id)
        merged.push(model)
      }
    }

    for (const model of userModels) {
      if (!seen.has(model.id)) {
        seen.add(model.id)
        merged.push(model)
      }
    }

    return merged
  })
}

// ── Delete implementation ──

function deleteModelFromDisk(
  modelId: string,
): Effect.Effect<void, never, FileSystem.FileSystem | Path.Path | HttpClient.HttpClient> {
  return Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem

    // Find the model by scanning
    const models = yield* discoverModels({})
    const model = models.find((m) => m.id === modelId)
    if (!model) return

    // Delete all shards
    const filesToDelete = model.shardPaths ?? [model.filePath]
    for (const file of filesToDelete) {
      const exists = yield* fs.exists(file).pipe(Effect.catchAll(() => Effect.succeed(false)))
      if (exists) {
        yield* fs.remove(file).pipe(Effect.ignore)
      }
    }

    // Delete paired mmproj
    if (model.mmprojPath) {
      const mmprojExists = yield* fs.exists(model.mmprojPath).pipe(
        Effect.catchAll(() => Effect.succeed(false)),
      )
      if (mmprojExists) {
        yield* fs.remove(model.mmprojPath).pipe(Effect.ignore)
      }
    }
  })
}
