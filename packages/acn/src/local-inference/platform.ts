import { randomUUID } from "node:crypto"
import { homedir, platform } from "node:os"
import { createServer } from "node:net"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import {
  Context,
  Effect,
  Exit,
  Layer,
  Option,
  PubSub,
  Redacted,
  Ref,
  Scope,
  Stream,
} from "effect"
import {
  Hardware,
  HuggingFace,
  LlamaCpp,
  LocalModelIndexSchema,
  type LocalModelIndexStoreApi,
  makeLocalModelIndexStore,
  ModelFiles,
} from "@magnitudedev/local-inference"
import { readStructuredFile, writeStructuredFileAtomic } from "@magnitudedev/storage"

interface ActiveInstances {
  readonly scope: Scope.CloseableScope
  readonly registry: LlamaCpp.LlamaInstanceRegistryApi
}

export interface LocalInferencePlatformApi {
  readonly modelIndex: LocalModelIndexStoreApi
  readonly files: ModelFiles.ModelFileRegistryApi
  readonly hardware: Hardware.HostHardwareApi
  readonly installations: LlamaCpp.LlamaCppInstallationRegistryApi
  readonly hub: HuggingFace.HuggingFaceHubApi
  readonly downloads: HuggingFace.HuggingFaceDownloadApi
  readonly cli: Effect.Effect<LlamaCpp.LlamaCli, LlamaCpp.LlamaCppInstallationUnavailable>
  readonly instances: Effect.Effect<LlamaCpp.LlamaInstanceRegistryApi>
  readonly instanceChanges: Stream.Stream<LlamaCpp.LlamaInstanceRegistryApi>
  readonly serverClient: (
    origin: URL,
    authorization: Option.Option<Redacted.Redacted<string>>,
  ) => Effect.Effect<LlamaCpp.LlamaServerClient>
}

export class LocalInferencePlatform extends Context.Tag("LocalInferencePlatform")<
  LocalInferencePlatform,
  LocalInferencePlatformApi
>() {}

export interface LocalInferencePlatformOptions {
  readonly root: string
  readonly modelsRoot: string
  readonly indexPath: string
}

const freePort = (): Effect.Effect<number> => Effect.async<number>((resume) => {
  const server = createServer()
  server.once("error", () => resume(Effect.succeed(0)))
  server.listen(0, "127.0.0.1", () => {
    const address = server.address()
    const port = typeof address === "object" && address ? address.port : 0
    server.close(() => resume(Effect.succeed(port)))
  })
})

export const LocalInferencePlatformLive = (
  options: LocalInferencePlatformOptions,
): Layer.Layer<
  LocalInferencePlatform,
  never,
  FileSystem.FileSystem | Path.Path | HttpClient.HttpClient | CommandExecutor.CommandExecutor
> => Layer.scoped(LocalInferencePlatform, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  const http = yield* HttpClient.HttpClient
  const executor = yield* CommandExecutor.CommandExecutor
  const hardware = yield* Hardware.makeHostHardware()
  const host = yield* hardware.inspect.pipe(Effect.orDie)
  const store = {
    cacheRoot: options.modelsRoot,
    installationRoot: path.join(options.modelsRoot, ".manifests"),
    sourceId: ModelFiles.ModelFileSourceId.make("huggingface-managed"),
  }
  const managedSource = yield* HuggingFace.makeHuggingFaceCacheSource({ store, label: Option.some("Magnitude models") })
  const existingMagnitudeModelsSource = yield* ModelFiles.makeDirectoryModelSource({
    id: ModelFiles.ModelFileSourceId.make("existing-magnitude-models"),
    label: Option.some("Existing Magnitude models"),
    root: path.join(options.root, "..", "llamacpp", "models"),
    recursive: true,
    followSymlinks: false,
    maxDepth: 8,
    ignore: Option.none(),
  })
  const huggingFaceRoot = process.env.HF_HOME?.trim()
    ? path.join(process.env.HF_HOME.trim(), "hub")
    : path.join(homedir(), ".cache", "huggingface", "hub")
  const externalCacheSource = yield* ModelFiles.makeDirectoryModelSource({
    id: ModelFiles.ModelFileSourceId.make("huggingface-user-cache"),
    label: Option.some("Hugging Face cache"),
    root: huggingFaceRoot,
    recursive: true,
    followSymlinks: true,
    maxDepth: 10,
    ignore: Option.some((relative) => relative.includes("/.locks/") || relative.endsWith(".lock")),
  })
  const gguf = yield* ModelFiles.makeGgufFormat()
  const persistedIndex = yield* readStructuredFile(options.indexPath, LocalModelIndexSchema).pipe(
    Effect.option,
    Effect.map(Option.flatMap((result) => result._tag === "Present"
      ? Option.some(result.value)
      : Option.none())),
  )
  const modelIndex = yield* makeLocalModelIndexStore({
    initialIndex: persistedIndex,
    persist: (index) => writeStructuredFileAtomic(options.indexPath, LocalModelIndexSchema, index).pipe(
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.catchAll((cause) => Effect.logWarning("Failed to persist local model index").pipe(
        Effect.annotateLogs({ cause: String(cause) }),
      )),
    ),
  })
  const registry = yield* ModelFiles.makeModelFileRegistry({
    sources: [
      ModelFiles.ModelFileSourceRegistration.Deletable({ source: managedSource }),
      ModelFiles.ModelFileSourceRegistration.ReadOnly({ source: existingMagnitudeModelsSource }),
      ModelFiles.ModelFileSourceRegistration.ReadOnly({ source: externalCacheSource }),
    ],
    formats: [gguf],
    initialIndex: Option.map(persistedIndex, (index) => index.artifacts),
  })
  const persistArtifacts = registry.artifactIndex.pipe(Effect.flatMap(modelIndex.replaceArtifacts))
  const files: ModelFiles.ModelFileRegistryApi = {
    ...registry,
    inspect: (refresh) => registry.inspect(refresh).pipe(
      Effect.tap(() => refresh === "cached" ? Effect.void : persistArtifacts),
    ),
    remove: (id) => registry.remove(id).pipe(
      Effect.zipRight(persistArtifacts),
    ),
  }
  const managedVariant = host.platform === "darwin"
    ? host.nativeArchitecture === "arm64" ? Option.some(LlamaCpp.LlamaDistributionVariantId.make("macos-arm64-metal")) : Option.some(LlamaCpp.LlamaDistributionVariantId.make("macos-x64-cpu"))
    : host.platform === "linux"
      ? host.nativeArchitecture === "arm64" ? Option.some(LlamaCpp.LlamaDistributionVariantId.make("linux-arm64-cpu")) : Option.some(LlamaCpp.LlamaDistributionVariantId.make("linux-x64-cpu"))
      : Option.none<LlamaCpp.LlamaDistributionVariantId>()
  const installations = yield* LlamaCpp.makeLlamaCppInstallationRegistry({
    configuredServerExecutable: Option.none(),
    managedRoot: path.join(options.root, "llamacpp", "distribution"),
    searchPath: [],
    managedVariant,
    manifest: LlamaCpp.DEFAULT_LLAMA_DISTRIBUTION_MANIFEST,
    platform: platform(),
    nativeArchitecture: host.nativeArchitecture,
  })
  const connection = {
    hubUrl: Option.none<URL>(),
    token: Option.none<Redacted.Redacted<string>>(),
    fetch: Option.none<typeof fetch>(),
  }
  const hub = HuggingFace.makeHuggingFaceHub(connection)
  const capacity = yield* HuggingFace.makeStorageCapacity()
  const downloads = yield* HuggingFace.makeHuggingFaceDownload({
    store,
    reserveBytes: 2 * 1024 ** 3,
    progressIntervalMillis: 250,
    connection,
  }).pipe(Effect.provideService(HuggingFace.StorageCapacity, capacity))

  const active = yield* Ref.make<Option.Option<ActiveInstances>>(Option.none())
  const instanceChanges = yield* PubSub.unbounded<LlamaCpp.LlamaInstanceRegistryApi>()
  const lock = yield* Effect.makeSemaphore(1)
  const selectedCli = installations.selected.pipe(Effect.flatMap((installation) => LlamaCpp.makeLlamaCli(installation).pipe(
    Effect.provideService(CommandExecutor.CommandExecutor, executor),
  )))
  const managedCli = selectedCli.pipe(
    Effect.map(Option.some),
    Effect.catchTag("LlamaCppInstallationUnavailable", () => Effect.succeed(Option.none<LlamaCpp.LlamaCli>())),
  )

  const resolveActive = lock.withPermits(1)(Effect.gen(function* () {
    const current = yield* Ref.get(active)
    if (Option.isSome(current)) return current.value
    const scope = yield* Scope.make()
    const port = yield* freePort()
    if (port <= 0) return yield* Effect.die("Unable to reserve a loopback port for llama.cpp")
    const registry = yield* LlamaCpp.makeLlamaInstanceRegistry({
      managedCli,
      modelFiles: files,
      presetPath: path.join(options.root, "llamacpp", "runtime", "models.ini"),
      host: "127.0.0.1",
      port,
      apiKey: Redacted.make(randomUUID()),
      modelsMax: 1,
      external: [],
    }).pipe(
      Effect.provideService(Scope.Scope, scope),
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.provideService(Path.Path, path),
      Effect.provideService(HttpClient.HttpClient, http),
    )
    const next = { scope, registry }
    yield* Ref.set(active, Option.some(next))
    yield* PubSub.publish(instanceChanges, registry)
    return next
  }))

  yield* Effect.addFinalizer(() => Ref.get(active).pipe(Effect.flatMap(Option.match({
    onNone: () => Effect.void,
    onSome: (value) => Scope.close(value.scope, Exit.void),
  }))))
  yield* installations.changes.pipe(
    Stream.runForEach(() => Ref.get(active).pipe(Effect.flatMap(Option.match({
      onNone: () => Effect.void,
      onSome: (value) => value.registry.reconcileManagedInstallation,
    })))),
    Effect.forkScoped,
  )

  return {
    modelIndex,
    files,
    hardware,
    installations,
    hub,
    downloads,
    cli: selectedCli,
    instances: resolveActive.pipe(Effect.map((value) => value.registry)),
    instanceChanges: Stream.fromPubSub(instanceChanges),
    serverClient: (origin, authorization) => LlamaCpp.makeLlamaServerClient({
      origin,
      authorization,
      timeout: Option.some("30 seconds"),
    }).pipe(Effect.provideService(HttpClient.HttpClient, http)),
  }
}))
