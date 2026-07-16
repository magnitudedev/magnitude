import { randomUUID } from "node:crypto"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Effect, Exit, Fiber, Layer, Option, Schema, Stream } from "effect"
import {
  ModelArtifactKey,
  makeModelFileId,
} from "../model-files"
import { sha256File } from "../model-files/platform"
import { makeHuggingFaceArtifactId } from "./artifact-identity"
import { blobPath, isWithin, snapshotPath } from "./cache-paths"
import {
  DownloadCheckingSpace,
  type DownloadProgress,
  DownloadReady,
  DownloadVerifying,
  DownloadingProgress,
  HuggingFaceArtifact,
  type HuggingFaceConnectionOptions,
  HuggingFaceDownload,
  type HuggingFaceDownloadApi,
  type HuggingFaceManagedStoreOptions,
  StorageCapacity,
} from "./contracts"
import {
  HuggingFaceArtifactInvalidError,
  HuggingFaceCacheError,
  type HuggingFaceDownloadError,
  HuggingFaceDigestMismatchError,
  HuggingFaceInvalidRequestError,
  HuggingFaceInsufficientSpaceError,
  HuggingFaceManifestPublicationError,
  HuggingFaceSizeMismatchError,
} from "./errors"
import { HuggingFaceFilePath } from "./identity"
import { HuggingFaceInstallationManifestJson, type HuggingFaceCachedFile as CachedFile } from "./installation-schema"
import { mapHuggingFaceHubError } from "./hub"
import { makeHuggingFaceUpstream, type HuggingFaceUpstreamApi } from "./upstream"

export interface HuggingFaceDownloadOptions {
  readonly store: HuggingFaceManagedStoreOptions
  readonly reserveBytes: number
  readonly progressIntervalMillis: number
  readonly connection: HuggingFaceConnectionOptions
}

export interface HuggingFaceDownloadFromUpstreamOptions extends Omit<HuggingFaceDownloadOptions, "connection"> {
  readonly upstream: HuggingFaceUpstreamApi
}

const safeDiagnostic = (error: unknown): string => error instanceof Error ? error.name.slice(0, 128) : "UnknownError"

export const makeHuggingFaceDownloadFromUpstream = (options: HuggingFaceDownloadFromUpstreamOptions): Effect.Effect<HuggingFaceDownloadApi, never, FileSystem.FileSystem | Path.Path | StorageCapacity> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  const capacity = yield* StorageCapacity
  const cacheRoot = path.resolve(options.store.cacheRoot)
  const installationRoot = path.resolve(options.store.installationRoot)
  const writerLock = yield* Effect.makeSemaphore(1)

  const emitProgress = (emit: { readonly single: (value: DownloadProgress) => boolean }, value: DownloadProgress) => Effect.sync(() => { emit.single(value) })

  const run = (input: Schema.Schema.Type<typeof HuggingFaceArtifact>, emit: { readonly single: (value: DownloadProgress) => boolean }): Effect.Effect<void, HuggingFaceDownloadError> => writerLock.withPermits(1)(Effect.gen(function* () {
    const artifact = yield* Schema.validate(HuggingFaceArtifact)(input).pipe(
      Effect.mapError(() => new HuggingFaceInvalidRequestError({ operation: "download", diagnostic: "InvalidArtifact" })),
    )
    if (makeHuggingFaceArtifactId(artifact) !== artifact.id) return yield* new HuggingFaceArtifactInvalidError({ repository: artifact.repository, reason: "artifact-id-mismatch", path: Option.none() })
    yield* fs.makeDirectory(cacheRoot, { recursive: true }).pipe(
      Effect.mapError((error) => new HuggingFaceCacheError({ operation: "prepare", path: cacheRoot, diagnostic: safeDiagnostic(error) })),
    )
    yield* fs.makeDirectory(installationRoot, { recursive: true }).pipe(
      Effect.mapError((error) => new HuggingFaceCacheError({ operation: "prepare", path: installationRoot, diagnostic: safeDiagnostic(error) })),
    )

    const cached = new Map<string, string>()
    for (const file of artifact.files) {
      const pointer = snapshotPath(path, cacheRoot, artifact.repository, artifact.commit, file)
      const info = yield* fs.stat(pointer).pipe(Effect.option)
      if (Option.isSome(info) && info.value.type === "File" && Number(info.value.size) === file.sizeBytes) cached.set(file.path, pointer)
    }
    let completedBytes = artifact.files.filter((file) => cached.has(file.path)).reduce((sum, file) => sum + file.sizeBytes, 0)
    const requiredBytes = artifact.totalBytes - completedBytes + options.reserveBytes
    const availableBytes = yield* capacity.availableBytes(cacheRoot)
    yield* emitProgress(emit, new DownloadCheckingSpace({ artifactId: artifact.id, requiredBytes, availableBytes, aggregate: { completedBytes, totalBytes: artifact.totalBytes } }))
    if (availableBytes < requiredBytes) return yield* new HuggingFaceInsufficientSpaceError({ requiredBytes, availableBytes })

    const installedFiles: CachedFile[] = []
    for (const file of artifact.files) {
      let pointer = cached.get(file.path)
      if (!pointer) {
        const incomplete = `${blobPath(path, cacheRoot, artifact.repository, file)}.incomplete`
        yield* fs.remove(incomplete, { force: true }).pipe(Effect.ignore)
        let observedFileBytes = 0
        yield* emitProgress(emit, new DownloadingProgress({
          artifactId: artifact.id,
          file: { path: file.path, completedBytes: 0, totalBytes: file.sizeBytes },
          aggregate: { completedBytes, totalBytes: artifact.totalBytes },
        }))
        const poll = Effect.gen(function* () {
          const info = yield* fs.stat(incomplete).pipe(Effect.option)
          if (Option.isSome(info) && info.value.type === "File") {
            const current = Math.min(file.sizeBytes, Number(info.value.size))
            if (current > observedFileBytes) {
              observedFileBytes = current
              yield* emitProgress(emit, new DownloadingProgress({
                artifactId: artifact.id,
                file: { path: file.path, completedBytes: current, totalBytes: file.sizeBytes },
                aggregate: { completedBytes: completedBytes + current, totalBytes: artifact.totalBytes },
              }))
            }
          }
          yield* Effect.sleep(`${Math.max(100, options.progressIntervalMillis)} millis`)
        }).pipe(Effect.forever)
        const pollFiber = yield* Effect.fork(poll)
        const downloadedPointer = yield* options.upstream.downloadToCache({ repository: artifact.repository, commit: artifact.commit, path: file.path, cacheDir: cacheRoot }).pipe(
          Effect.mapError((error) => mapHuggingFaceHubError(error, { operation: "download", repository: Option.some(artifact.repository), revision: Option.some(artifact.requestedRevision), path: Option.some(file.path) })),
          Effect.ensuring(Fiber.interrupt(pollFiber)),
          Effect.onExit((exit) => Exit.isFailure(exit) ? fs.remove(incomplete, { force: true }).pipe(Effect.ignore) : Effect.void),
        )
        pointer = downloadedPointer
        if (!isWithin(path, cacheRoot, path.resolve(downloadedPointer))) return yield* new HuggingFaceCacheError({ operation: "inspect", path: downloadedPointer, diagnostic: "UpstreamPathEscapesCacheRoot" })
        const info = yield* fs.stat(downloadedPointer).pipe(
          Effect.mapError((error) => new HuggingFaceCacheError({ operation: "inspect", path: downloadedPointer, diagnostic: safeDiagnostic(error) })),
        )
        if (info.type !== "File" || Number(info.size) !== file.sizeBytes) return yield* new HuggingFaceSizeMismatchError({ path: file.path, expectedBytes: file.sizeBytes, actualBytes: Number(info.size) })
        if (observedFileBytes !== file.sizeBytes) yield* emitProgress(emit, new DownloadingProgress({
          artifactId: artifact.id,
          file: { path: file.path, completedBytes: file.sizeBytes, totalBytes: file.sizeBytes },
          aggregate: { completedBytes: completedBytes + file.sizeBytes, totalBytes: artifact.totalBytes },
        }))
        completedBytes += file.sizeBytes
      }

      yield* emitProgress(emit, new DownloadVerifying({ artifactId: artifact.id, path: file.path, aggregate: { completedBytes, totalBytes: artifact.totalBytes } }))
      const finalInfo = yield* fs.stat(pointer).pipe(
        Effect.mapError((error) => new HuggingFaceCacheError({ operation: "inspect", path: pointer, diagnostic: safeDiagnostic(error) })),
      )
      if (finalInfo.type !== "File" || Number(finalInfo.size) !== file.sizeBytes) return yield* new HuggingFaceSizeMismatchError({ path: file.path, expectedBytes: file.sizeBytes, actualBytes: Number(finalInfo.size) })
      if (file.content._tag === "LfsSha256") {
        const digest = yield* sha256File(pointer).pipe(
          Effect.provideService(FileSystem.FileSystem, fs),
          Effect.mapError((error) => new HuggingFaceCacheError({ operation: "inspect", path: pointer, diagnostic: safeDiagnostic(error) })),
        )
        if (digest !== file.content.sha256) return yield* new HuggingFaceDigestMismatchError({ path: file.path })
      }
      const relative = path.relative(cacheRoot, pointer)
      const snapshotRelativePath = yield* Schema.decodeUnknown(HuggingFaceFilePath)(relative).pipe(
        Effect.mapError(() => new HuggingFaceCacheError({ operation: "inspect", path: pointer, diagnostic: "UnsafeSnapshotPath" })),
      )
      installedFiles.push({ ...file, snapshotRelativePath })
    }

    const manifestPath = path.join(installationRoot, `${artifact.id}.json`)
    const temporary = path.join(installationRoot, `.${artifact.id}.${randomUUID()}.tmp`)
    const encoded = yield* Schema.encode(HuggingFaceInstallationManifestJson)({
      version: 1,
      artifact: { ...artifact, files: [...artifact.files], relationships: [...artifact.relationships] },
      files: installedFiles,
      installedAt: new Date(),
    }).pipe(Effect.mapError((error) => new HuggingFaceManifestPublicationError({ path: manifestPath, diagnostic: safeDiagnostic(error) })))
    yield* fs.writeFileString(temporary, encoded, { mode: 0o600 }).pipe(
      Effect.mapError((error) => new HuggingFaceManifestPublicationError({ path: temporary, diagnostic: safeDiagnostic(error) })),
    )
    yield* fs.rename(temporary, manifestPath).pipe(
      Effect.mapError((error) => new HuggingFaceManifestPublicationError({ path: manifestPath, diagnostic: safeDiagnostic(error) })),
      Effect.ensuring(fs.remove(temporary, { force: true }).pipe(Effect.ignore)),
    )
    const modelFileId = makeModelFileId(options.store.sourceId, ModelArtifactKey.make(artifact.id))
    yield* emitProgress(emit, new DownloadReady({ artifactId: artifact.id, modelFileId, aggregate: { completedBytes: artifact.totalBytes, totalBytes: artifact.totalBytes } }))
  }))

  return {
    download: (artifact) => Stream.asyncPush<DownloadProgress, HuggingFaceDownloadError>((emit) => run(artifact, emit).pipe(
      Effect.matchEffect({
        onFailure: (error) => Effect.sync(() => { emit.fail(error) }),
        onSuccess: () => Effect.sync(() => { emit.end() }),
      }),
      Effect.forkScoped,
    ), { bufferSize: "unbounded" }),
  }
})

export const makeHuggingFaceDownload = (options: HuggingFaceDownloadOptions): Effect.Effect<HuggingFaceDownloadApi, never, FileSystem.FileSystem | Path.Path | StorageCapacity> =>
  makeHuggingFaceDownloadFromUpstream({ ...options, upstream: makeHuggingFaceUpstream(options.connection) })

export const HuggingFaceDownloadLive = (options: HuggingFaceDownloadOptions): Layer.Layer<HuggingFaceDownload, never, FileSystem.FileSystem | Path.Path | StorageCapacity> =>
  Layer.effect(HuggingFaceDownload, makeHuggingFaceDownload(options))
