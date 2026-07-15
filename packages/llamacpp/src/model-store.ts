import { createHash } from "node:crypto"
import { Context, Effect, Either, Layer, Option, Schema, Secret, Stream } from "effect"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as Path from "@effect/platform/Path"
import type {
  ArtifactDownloadFile,
  ArtifactDownloadPlan,
  LlamaCppModelStoreConfig,
  ModelArtifactMetadata,
  ModelArtifactSummary,
  ModelDownloadEvent,
  ModelStoreSnapshot,
  ResolvedModelArtifact,
} from "./contracts"
import { LlamaCppModelStoreError } from "./errors"
import { readGgufMetadata } from "./models/gguf"
import { hfCacheDir, scanHfCache } from "./models/hf-cache"
import { scanDirectory } from "./models/scan"
import type { LocalModelInfo } from "./models/types"

type ModelStorePlatform =
  | FileSystem.FileSystem
  | Path.Path
  | HttpClient.HttpClient
  | CommandExecutor.CommandExecutor

export interface LlamaCppModelStoreApi {
  readonly inspect: Effect.Effect<ModelStoreSnapshot, LlamaCppModelStoreError>
  readonly resolve: (modelId: string) => Effect.Effect<ResolvedModelArtifact, LlamaCppModelStoreError>
  readonly download: (plan: ArtifactDownloadPlan) => Stream.Stream<ModelDownloadEvent, LlamaCppModelStoreError>
  readonly deleteOwned: (modelId: string) => Effect.Effect<void, LlamaCppModelStoreError>
}

export class LlamaCppModelStore extends Context.Tag("LlamaCppModelStore")<
  LlamaCppModelStore,
  LlamaCppModelStoreApi
>() {}

const DownloadFileSchema = Schema.Struct({
  path: Schema.String,
  sizeBytes: Schema.Number,
  sha256: Schema.String,
})
const OwnedManifestSchema = Schema.Struct({
  artifactId: Schema.String,
  repo: Schema.String,
  revision: Schema.String,
  files: Schema.Array(DownloadFileSchema),
})
const OwnedManifestJson = Schema.parseJson(OwnedManifestSchema)

type OwnedManifest = Schema.Schema.Type<typeof OwnedManifestSchema>

const storeError = (
  operation: LlamaCppModelStoreError["operation"],
  code: LlamaCppModelStoreError["code"],
  reason: string,
  options?: { readonly modelId?: string; readonly cause?: unknown },
): LlamaCppModelStoreError => new LlamaCppModelStoreError({
  operation,
  code,
  reason,
  ...(options?.modelId === undefined ? {} : { modelId: options.modelId }),
  ...(options?.cause === undefined ? {} : { cause: options.cause }),
})

const safeRelativePath = (value: string): boolean =>
  value.length > 0
  && !value.startsWith("/")
  && !value.startsWith("\\")
  && !value.split(/[\\/]+/).includes("..")

const validatePlan = (plan: ArtifactDownloadPlan): Effect.Effect<void, LlamaCppModelStoreError> =>
  Effect.gen(function* () {
    if (!plan.artifactId.trim() || !plan.repo.trim() || !plan.revision.trim() || plan.files.length === 0) {
      return yield* storeError("download", "invalid_plan", "Artifact plan is incomplete")
    }
    if (!/^[a-f0-9]{40}$/i.test(plan.revision)) {
      return yield* storeError("download", "invalid_plan", "Artifact revision must be an immutable 40-character commit SHA")
    }
    const seen = new Set<string>()
    for (const file of plan.files) {
      if (
        !safeRelativePath(file.path)
        || !Number.isSafeInteger(file.sizeBytes) || file.sizeBytes <= 0
        || !/^[a-f0-9]{64}$/i.test(file.sha256)
        || seen.has(file.path)
      ) {
        return yield* storeError("download", "invalid_plan", `Invalid artifact file plan: ${file.path}`)
      }
      seen.add(file.path)
    }
    if (!Number.isSafeInteger(plan.safetyReserveBytes) || plan.safetyReserveBytes < 0) {
      return yield* storeError("download", "invalid_plan", "Artifact safety reserve is invalid")
    }
  })

const sha256File = (
  file: string,
): Effect.Effect<string, LlamaCppModelStoreError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const hash = createHash("sha256")
    yield* fs.stream(file).pipe(
      Stream.runForEach((chunk) => Effect.sync(() => hash.update(chunk))),
      Effect.mapError((cause) => storeError("resolve", "storage_failed", `Could not hash ${file}`, { cause })),
    )
    return hash.digest("hex")
  })

const metadataFrom = (model: LocalModelInfo): ModelArtifactMetadata => ({
  displayName: model.displayName,
  architecture: model.architecture ?? null,
  quantization: model.quantization ?? null,
  contextLength: model.contextLength ?? null,
  parameterCount: model.parameterCount ?? null,
  layerCount: model.layerCount ?? null,
  tokenizerModel: model.tokenizerModel ?? null,
  tokenizerPre: model.tokenizerPre ?? null,
  baseModelNames: model.baseModelNames ?? [],
})

const stableId = (value: string): string => createHash("sha256").update(value).digest("hex")

const resolvedFromScanned = (
  model: LocalModelInfo,
  source: ResolvedModelArtifact["source"],
  modelId: string,
): ResolvedModelArtifact => ({
  modelId,
  source,
  sizeBytes: model.fileSizeBytes,
  metadata: metadataFrom(model),
  hasVisionProjector: model.mmprojPath !== undefined,
  primaryPath: model.filePath,
  shardPaths: model.shardPaths ?? [model.filePath],
  projectorPath: model.mmprojPath ?? null,
})

const readOwnedArtifact = (
  config: LlamaCppModelStoreConfig,
  artifactId: string,
  verify: boolean,
): Effect.Effect<ResolvedModelArtifact, LlamaCppModelStoreError, FileSystem.FileSystem | Path.Path> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const root = path.join(config.ownedRoot, artifactId)
    const manifestFile = path.join(root, "manifest.json")
    const manifest = yield* fs.readFileString(manifestFile).pipe(
      Effect.flatMap(Schema.decodeUnknown(OwnedManifestJson)),
      Effect.mapError((cause) => storeError("resolve", "integrity_failed", `Owned artifact ${artifactId} has an invalid manifest`, { modelId: artifactId, cause })),
    )
    if (manifest.artifactId !== artifactId) {
      return yield* storeError("resolve", "integrity_failed", "Owned artifact directory and manifest ID differ", { modelId: artifactId })
    }

    for (const file of manifest.files) {
      if (!safeRelativePath(file.path)) {
        return yield* storeError("resolve", "integrity_failed", "Owned artifact manifest contains an unsafe path", { modelId: artifactId })
      }
      const absolute = path.join(root, file.path)
      const stat = yield* fs.stat(absolute).pipe(Effect.orElseSucceed(() => null))
      if (stat?.type !== "File" || Number(stat.size) !== file.sizeBytes) {
        return yield* storeError("resolve", "integrity_failed", `Owned artifact file is missing or has the wrong size: ${file.path}`, { modelId: artifactId })
      }
      if (verify) {
        const digest = yield* sha256File(absolute)
        if (digest !== file.sha256.toLowerCase()) {
          return yield* storeError("resolve", "integrity_failed", `Owned artifact SHA-256 mismatch: ${file.path}`, { modelId: artifactId })
        }
      }
    }

    const modelFiles = manifest.files.filter((file) => file.path.toLowerCase().endsWith(".gguf") && !file.path.toLowerCase().includes("mmproj"))
    const primary = modelFiles[0]
    if (!primary) {
      return yield* storeError("resolve", "integrity_failed", "Owned artifact contains no model GGUF", { modelId: artifactId })
    }
    const primaryPath = path.join(root, primary.path)
    const metadata = yield* readGgufMetadata(primaryPath)
    const projector = manifest.files.find((file) => file.path.toLowerCase().includes("mmproj") && file.path.toLowerCase().endsWith(".gguf"))
    const displayName = metadata?.generalName ?? metadata?.generalBasename ?? artifactId
    return {
      modelId: artifactId,
      source: { _tag: "MagnitudeOwned", manifestId: artifactId },
      // Runtime fit and identity checks concern the model weights. The projector
      // is a companion artifact with its own memory requirements, not a model
      // shard, so it must not be folded into the served model's byte identity.
      sizeBytes: modelFiles.reduce((sum, file) => sum + file.sizeBytes, 0),
      metadata: {
        displayName,
        architecture: metadata?.architecture ?? null,
        quantization: metadata?.quantization ?? null,
        contextLength: metadata?.contextLength ?? null,
        parameterCount: metadata?.parameterCount ?? null,
        layerCount: metadata?.layerCount ?? null,
        tokenizerModel: metadata?.tokenizerModel ?? null,
        tokenizerPre: metadata?.tokenizerPre ?? null,
        baseModelNames: metadata?.baseModelNames ?? [],
      },
      hasVisionProjector: projector !== undefined,
      primaryPath,
      shardPaths: modelFiles.map((file) => path.join(root, file.path)),
      projectorPath: projector ? path.join(root, projector.path) : null,
    }
  })

const discoverOwned = (
  config: LlamaCppModelStoreConfig,
): Effect.Effect<{
  readonly artifacts: readonly ResolvedModelArtifact[]
  readonly warnings: readonly { readonly code: string; readonly message: string }[]
}, never, FileSystem.FileSystem | Path.Path> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const entries = yield* fs.readDirectory(config.ownedRoot).pipe(Effect.orElseSucceed(() => []))
    const results = yield* Effect.all(entries
      .filter((entry) => !entry.startsWith("."))
      .map((entry) => readOwnedArtifact(config, entry, false).pipe(Effect.either)), {
        concurrency: 4,
      })
    return {
      artifacts: results.flatMap((result) => Either.isRight(result) ? [result.right] : []),
      warnings: results.flatMap((result) => Either.isLeft(result)
        ? [{ code: "invalid_owned_artifact", message: result.left.reason }]
        : []),
    }
  })

const discoverExternal = (
  config: LlamaCppModelStoreConfig,
): Effect.Effect<readonly ResolvedModelArtifact[], never, FileSystem.FileSystem | Path.Path> =>
  Effect.gen(function* () {
    const hfRoot = config.huggingFaceCacheRoot ?? hfCacheDir()
    const hf = yield* scanHfCache(hfRoot, (directory) =>
      scanDirectory(directory, { _tag: "user-dir", dir: directory }))
    const hfArtifacts = hf.map((model) => resolvedFromScanned(
      model,
      {
        _tag: "HuggingFaceCache",
        repo: model.repoId ?? "unknown",
        revision: model.commit ?? "unknown",
      },
      stableId(`hf:${model.repoId ?? "unknown"}:${model.commit ?? "unknown"}:${model.filePath}`),
    ))

    const userArtifacts: ResolvedModelArtifact[] = []
    for (const directory of config.userDirectories ?? []) {
      const models = yield* scanDirectory(directory.path, { _tag: "user-dir", dir: directory.path })
      for (const model of models) {
        userArtifacts.push(resolvedFromScanned(
          model,
          { _tag: "UserDirectory", directoryId: directory.directoryId },
          stableId(`user:${directory.directoryId}:${model.filePath}`),
        ))
      }
    }
    return [...hfArtifacts, ...userArtifacts]
  })

const discoverAll = (
  config: LlamaCppModelStoreConfig,
): Effect.Effect<readonly ResolvedModelArtifact[], LlamaCppModelStoreError, FileSystem.FileSystem | Path.Path> =>
  Effect.all([discoverOwned(config), discoverExternal(config)], { concurrency: 2 }).pipe(
    Effect.map(([owned, external]) => [...owned.artifacts, ...external]),
    Effect.mapError((cause) => storeError("inspect", "storage_failed", "Could not inspect model stores", { cause })),
  )

const summary = ({ primaryPath: _, shardPaths: __, projectorPath: ___, ...artifact }: ResolvedModelArtifact): ModelArtifactSummary => artifact

const availableBytes = (
  root: string,
): Effect.Effect<number, LlamaCppModelStoreError, CommandExecutor.CommandExecutor> =>
  Command.string(Command.make("df", "-Pk", root)).pipe(
    Effect.flatMap((output) => {
      const lines = output.trim().split("\n")
      const columns = lines.at(-1)?.trim().split(/\s+/)
      const availableKiB = Number(columns?.at(-3))
      return Number.isFinite(availableKiB)
        ? Effect.succeed(availableKiB * 1024)
        : Effect.fail(storeError("download", "storage_failed", "Could not parse destination free space"))
    }),
    Effect.mapError((cause) => cause instanceof LlamaCppModelStoreError
      ? cause
      : storeError("download", "storage_failed", "Could not inspect destination free space", { cause })),
  )

const encodePath = (value: string): string => value.split("/").map(encodeURIComponent).join("/")
const downloadEvent = (event: ModelDownloadEvent): ModelDownloadEvent => event

const downloadArtifact = (
  config: LlamaCppModelStoreConfig,
  plan: ArtifactDownloadPlan,
): Stream.Stream<ModelDownloadEvent, LlamaCppModelStoreError, ModelStorePlatform> =>
  Stream.unwrapScoped(Effect.gen(function* () {
    yield* validatePlan(plan)
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const client = yield* HttpClient.HttpClient
    yield* fs.makeDirectory(config.ownedRoot, { recursive: true }).pipe(
      Effect.mapError((cause) => storeError("download", "storage_failed", "Could not create owned model store", { cause })),
    )
    const finalRoot = path.join(config.ownedRoot, plan.artifactId)
    const finalExists = yield* fs.exists(finalRoot).pipe(Effect.orElseSucceed(() => false))
    if (finalExists) {
      const existing = yield* readOwnedArtifact(config, plan.artifactId, true)
      return Stream.make(downloadEvent({ _tag: "Ready", artifact: summary(existing) }))
    }

    const partialRoot = path.join(config.ownedRoot, `.partial-${plan.artifactId}`)
    const planFile = path.join(partialRoot, "plan.json")
    const encodedPlan = yield* Schema.encode(OwnedManifestJson)({
      artifactId: plan.artifactId,
      repo: plan.repo,
      revision: plan.revision,
      files: plan.files,
    }).pipe(Effect.mapError((cause) => storeError("download", "invalid_plan", "Could not encode artifact plan", { cause })))
    const priorPlan = yield* fs.readFileString(planFile).pipe(Effect.option)
    if (Option.isSome(priorPlan) && priorPlan.value !== encodedPlan) {
      yield* fs.remove(partialRoot, { recursive: true, force: true }).pipe(
        Effect.mapError((cause) => storeError("download", "storage_failed", "Could not clear incompatible partial download", { cause })),
      )
    }
    yield* fs.makeDirectory(partialRoot, { recursive: true }).pipe(
      Effect.mapError((cause) => storeError("download", "storage_failed", "Could not create partial download directory", { cause })),
    )
    yield* fs.writeFileString(planFile, encodedPlan).pipe(
      Effect.mapError((cause) => storeError("download", "storage_failed", "Could not persist resume fingerprint", { cause })),
    )

    let completedBeforeStart = 0
    for (const file of plan.files) {
      const partialFile = path.join(partialRoot, `${file.path}.incomplete`)
      const destination = path.join(partialRoot, file.path)
      const destinationStat = yield* fs.stat(destination).pipe(Effect.orElseSucceed(() => null))
      if (destinationStat?.type === "File" && Number(destinationStat.size) === file.sizeBytes) {
        const digest = yield* sha256File(destination).pipe(Effect.provideService(FileSystem.FileSystem, fs))
        if (digest === file.sha256.toLowerCase()) {
          completedBeforeStart += file.sizeBytes
          yield* fs.remove(partialFile, { force: true }).pipe(Effect.ignore)
          continue
        }
      }
      if (destinationStat !== null) {
        yield* fs.remove(destination, { force: true }).pipe(
          Effect.mapError((cause) => storeError("download", "storage_failed", `Could not clear invalid finalized file ${file.path}`, { cause })),
        )
      }
      const partialStat = yield* fs.stat(partialFile).pipe(Effect.orElseSucceed(() => null))
      if (partialStat?.type === "File" && Number(partialStat.size) <= file.sizeBytes) {
        completedBeforeStart += Number(partialStat.size)
      } else if (partialStat !== null) {
        yield* fs.remove(partialFile, { force: true }).pipe(
          Effect.mapError((cause) => storeError("download", "storage_failed", `Could not clear invalid partial file ${file.path}`, { cause })),
        )
      }
    }
    const totalBytes = plan.files.reduce((sum, file) => sum + file.sizeBytes, 0)
    const requiredBytes = totalBytes - completedBeforeStart + plan.safetyReserveBytes
    const freeBytes = yield* availableBytes(config.ownedRoot)
    if (freeBytes < requiredBytes) {
      return Stream.fail(storeError("download", "insufficient_space", `Need ${requiredBytes} bytes but only ${freeBytes} bytes are available`))
    }

    let completedBytes = completedBeforeStart
    const downloadFile = (filePlan: ArtifactDownloadFile): Stream.Stream<ModelDownloadEvent, LlamaCppModelStoreError> =>
      Stream.unwrapScoped(Effect.gen(function* () {
        const partialFile = path.join(partialRoot, `${filePlan.path}.incomplete`)
        const destination = path.join(partialRoot, filePlan.path)
        yield* fs.makeDirectory(path.dirname(partialFile), { recursive: true }).pipe(
          Effect.mapError((cause) => storeError("download", "storage_failed", `Could not create directory for ${filePlan.path}`, { cause })),
        )
        const destinationStat = yield* fs.stat(destination).pipe(Effect.orElseSucceed(() => null))
        if (destinationStat?.type === "File" && Number(destinationStat.size) === filePlan.sizeBytes) {
          return Stream.empty
        }
        const partialStat = yield* fs.stat(partialFile).pipe(Effect.orElseSucceed(() => null))
        const requestedOffset = partialStat?.type === "File" && Number(partialStat.size) < filePlan.sizeBytes
          ? Number(partialStat.size)
          : 0
        if (partialStat?.type === "File" && Number(partialStat.size) >= filePlan.sizeBytes) {
          yield* fs.remove(partialFile, { force: true }).pipe(Effect.ignore)
        }
        const url = `https://huggingface.co/${encodePath(plan.repo)}/resolve/${encodeURIComponent(plan.revision)}/${encodePath(filePlan.path)}`
        let request = HttpClientRequest.get(url)
        if (requestedOffset > 0) request = HttpClientRequest.setHeader(request, "Range", `bytes=${requestedOffset}-`)
        if (config.huggingFaceToken) {
          request = HttpClientRequest.setHeader(request, "Authorization", `Bearer ${Secret.value(config.huggingFaceToken)}`)
        }
        const response = yield* client.execute(request).pipe(
          Effect.mapError((cause) => storeError("download", "download_failed", `Could not download ${filePlan.path}`, { cause })),
        )
        if (response.status !== 200 && response.status !== 206) {
          return Stream.fail(storeError("download", "download_failed", `Hugging Face returned HTTP ${response.status} for ${filePlan.path}`))
        }
        const offset = response.status === 206 ? requestedOffset : 0
        if (offset === 0 && requestedOffset > 0) {
          completedBytes -= requestedOffset
          yield* fs.remove(partialFile, { force: true }).pipe(Effect.ignore)
        }
        const handle = yield* fs.open(partialFile, { flag: offset > 0 ? "a" : "w" }).pipe(
          Effect.mapError((cause) => storeError("download", "storage_failed", `Could not open ${filePlan.path}`, { cause })),
        )
        const body = response.stream.pipe(
          Stream.mapError((cause) => storeError("download", "download_failed", `Download stream failed for ${filePlan.path}`, { cause })),
          Stream.mapEffect((chunk) => handle.write(chunk).pipe(
            Effect.tap(() => Effect.sync(() => { completedBytes += chunk.byteLength })),
            Effect.map((): ModelDownloadEvent => ({
              _tag: "Downloading",
              artifactId: plan.artifactId,
              file: filePlan.path,
              completedBytes,
              totalBytes,
            })),
            Effect.mapError((cause) => storeError("download", "storage_failed", `Could not write ${filePlan.path}`, { cause })),
          )),
        )
        const verify = Effect.gen(function* () {
          const stat = yield* fs.stat(partialFile).pipe(
            Effect.mapError((cause) => storeError("download", "storage_failed", `Could not inspect ${filePlan.path}`, { cause })),
          )
          if (stat.type !== "File" || Number(stat.size) !== filePlan.sizeBytes) {
            return yield* storeError("download", "integrity_failed", `Size mismatch for ${filePlan.path}`)
          }
          const digest = yield* sha256File(partialFile).pipe(
            Effect.provideService(FileSystem.FileSystem, fs),
            Effect.mapError((cause) => storeError("download", "integrity_failed", cause.reason, { cause })),
          )
          if (digest !== filePlan.sha256.toLowerCase()) {
            return yield* storeError("download", "integrity_failed", `SHA-256 mismatch for ${filePlan.path}`)
          }
          yield* fs.rename(partialFile, destination).pipe(
            Effect.mapError((cause) => storeError("download", "storage_failed", `Could not finalize ${filePlan.path}`, { cause })),
          )
        })
        return body.pipe(
          Stream.concat(Stream.make(downloadEvent({ _tag: "Verifying", artifactId: plan.artifactId, file: filePlan.path }))),
          Stream.concat(Stream.fromEffect(verify).pipe(Stream.drain)),
        )
      }))

    const files = Stream.fromIterable(plan.files).pipe(Stream.flatMap(downloadFile, { concurrency: 1 }))
    const publish = Effect.gen(function* () {
      yield* fs.remove(planFile, { force: true }).pipe(Effect.ignore)
      const manifest: OwnedManifest = {
        artifactId: plan.artifactId,
        repo: plan.repo,
        revision: plan.revision,
        files: [...plan.files],
      }
      const encoded = yield* Schema.encode(OwnedManifestJson)(manifest).pipe(
        Effect.mapError((cause) => storeError("download", "storage_failed", "Could not encode ownership manifest", { cause })),
      )
      yield* fs.writeFileString(path.join(partialRoot, "manifest.json"), encoded).pipe(
        Effect.mapError((cause) => storeError("download", "storage_failed", "Could not write ownership manifest", { cause })),
      )
      yield* fs.rename(partialRoot, finalRoot).pipe(
        Effect.mapError((cause) => storeError("download", "storage_failed", "Could not publish owned artifact", { cause })),
      )
      return yield* readOwnedArtifact(config, plan.artifactId, true)
    })
    return Stream.make(downloadEvent({ _tag: "Resolving", artifactId: plan.artifactId })).pipe(
      Stream.concat(Stream.make(downloadEvent({ _tag: "CheckingSpace", artifactId: plan.artifactId, requiredBytes, availableBytes: freeBytes }))),
      Stream.concat(files),
      Stream.concat(Stream.make(downloadEvent({ _tag: "Publishing", artifactId: plan.artifactId }))),
      Stream.concat(Stream.fromEffect(Effect.uninterruptible(publish)).pipe(
        Stream.map((artifact): ModelDownloadEvent => ({ _tag: "Ready", artifact: summary(artifact) })),
      )),
    )
  }))

const deleteOwnedArtifact = (
  config: LlamaCppModelStoreConfig,
  modelId: string,
): Effect.Effect<void, LlamaCppModelStoreError, FileSystem.FileSystem | Path.Path> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const artifacts = yield* discoverAll(config)
    const artifact = artifacts.find((candidate) => candidate.modelId === modelId)
    if (!artifact) return yield* storeError("delete", "artifact_not_found", `Unknown model ${modelId}`, { modelId })
    if (artifact.source._tag !== "MagnitudeOwned") {
      return yield* storeError("delete", "artifact_not_owned", "Only Magnitude-owned models can be deleted", { modelId })
    }
    yield* fs.remove(path.join(config.ownedRoot, artifact.source.manifestId), { recursive: true, force: true }).pipe(
      Effect.mapError((cause) => storeError("delete", "storage_failed", `Could not delete model ${modelId}`, { modelId, cause })),
    )
  })

export const LlamaCppModelStoreLive = (
  config: LlamaCppModelStoreConfig,
): Layer.Layer<LlamaCppModelStore, never, ModelStorePlatform> => Layer.effect(
  LlamaCppModelStore,
  Effect.gen(function* () {
    const context = yield* Effect.context<ModelStorePlatform>()
    const all = discoverAll(config).pipe(Effect.provide(context))
    const lockRegistry = yield* Effect.makeSemaphore(1)
    const artifactLocks = new Map<string, Effect.Semaphore>()
    const lockFor = (artifactId: string): Effect.Effect<Effect.Semaphore> =>
      lockRegistry.withPermits(1)(Effect.gen(function* () {
        const existing = artifactLocks.get(artifactId)
        if (existing) return existing
        const created = yield* Effect.makeSemaphore(1)
        artifactLocks.set(artifactId, created)
        return created
      }))

    const withArtifactStreamLock = (
      artifactId: string,
      stream: Stream.Stream<ModelDownloadEvent, LlamaCppModelStoreError>,
    ): Stream.Stream<ModelDownloadEvent, LlamaCppModelStoreError> => Stream.unwrap(
      lockFor(artifactId).pipe(Effect.map((lock) => Stream.acquireRelease(
        lock.take(1),
        () => lock.release(1),
      ).pipe(Stream.flatMap(() => stream)))),
    )

    return LlamaCppModelStore.of({
      inspect: Effect.all([discoverOwned(config), discoverExternal(config)], { concurrency: 2 }).pipe(
        Effect.map(([owned, external]) => ({
          artifacts: [...owned.artifacts, ...external].map(summary),
          warnings: owned.warnings,
        })),
        Effect.mapError((cause) => storeError("inspect", "storage_failed", "Could not inspect model stores", { cause })),
        Effect.provide(context),
      ),
      resolve: (modelId) => Effect.gen(function* () {
        const artifacts = yield* all
        const artifact = artifacts.find((candidate) => candidate.modelId === modelId)
        if (!artifact) return yield* storeError("resolve", "artifact_not_found", `Unknown model ${modelId}`, { modelId })
        return artifact.source._tag === "MagnitudeOwned"
          ? yield* readOwnedArtifact(config, modelId, true).pipe(Effect.provide(context))
          : artifact
      }),
      download: (plan) => withArtifactStreamLock(
        plan.artifactId,
        downloadArtifact(config, plan).pipe(Stream.provideContext(context)),
      ),
      deleteOwned: (modelId) => lockFor(modelId).pipe(
        Effect.flatMap((lock) => lock.withPermits(1)(
          deleteOwnedArtifact(config, modelId).pipe(Effect.provide(context)),
        )),
      ),
    })
  }),
)
