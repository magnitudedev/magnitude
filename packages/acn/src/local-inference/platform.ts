import { delimiter } from "node:path"
import { createHash, randomUUID } from "node:crypto"
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
  Redacted,
  Ref,
  Scope,
} from "effect"
import {
  Hardware,
  HuggingFace,
  LlamaCpp,
  ModelFiles,
} from "@magnitudedev/local-inference"
import { readStructuredFile, writeStructuredFileAtomic } from "@magnitudedev/storage"

export interface ExternalLlamaConfiguration {
  readonly id: string
  readonly origin: URL
  readonly apiKey: Option.Option<Redacted.Redacted<string>>
}

interface ActiveInstances {
  readonly key: string
  readonly externalKey: string
  readonly scope: Scope.CloseableScope
  readonly cli: Option.Option<LlamaCpp.LlamaCli>
  readonly registry: LlamaCpp.LlamaInstanceRegistryApi
}

export interface LocalInferencePlatformApi {
  readonly files: ModelFiles.ModelFileRegistryApi
  readonly hardware: Hardware.HostHardwareApi
  readonly distribution: LlamaCpp.LlamaDistributionApi
  readonly hub: HuggingFace.HuggingFaceHubApi
  readonly downloads: HuggingFace.HuggingFaceDownloadApi
  readonly cli: Effect.Effect<LlamaCpp.LlamaCli, LlamaCpp.LlamaDistributionError>
  readonly instances: Effect.Effect<LlamaCpp.LlamaInstanceRegistryApi, LlamaCpp.LlamaDistributionError>
}

export class LocalInferencePlatform extends Context.Tag("LocalInferencePlatform")<
  LocalInferencePlatform,
  LocalInferencePlatformApi
>() {}

export interface LocalInferencePlatformOptions {
  readonly root: string
  readonly indexPath: string
  readonly configuredExecutable: Option.Option<string>
  readonly external: Effect.Effect<readonly ExternalLlamaConfiguration[]>
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
    cacheRoot: path.join(options.root, "huggingface", "cache"),
    installationRoot: path.join(options.root, "huggingface", "installations"),
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
  const persistedIndex = yield* readStructuredFile(options.indexPath, ModelFiles.LocalModelFileIndexSchema).pipe(
    Effect.map((result) => result._tag === "Present" ? result.value : undefined),
    Effect.catchAll(() => Effect.succeed(undefined)),
  )
  const registry = yield* ModelFiles.makeModelFileRegistry({
    sources: [
      ModelFiles.ModelFileSourceRegistration.Deletable({ source: managedSource }),
      ModelFiles.ModelFileSourceRegistration.ReadOnly({ source: existingMagnitudeModelsSource }),
      ModelFiles.ModelFileSourceRegistration.ReadOnly({ source: externalCacheSource }),
    ],
    formats: [gguf],
    initialIndex: persistedIndex,
  })
  const persistIndex = registry.index.pipe(
    Effect.flatMap((index) => writeStructuredFileAtomic(options.indexPath, ModelFiles.LocalModelFileIndexSchema, index)),
    Effect.provideService(FileSystem.FileSystem, fs),
    Effect.catchAll((cause) => Effect.logWarning("Failed to persist local model index").pipe(
      Effect.annotateLogs({ cause: String(cause) }),
    )),
  )
  const files: ModelFiles.ModelFileRegistryApi = {
    ...registry,
    inspect: (refresh) => registry.inspect(refresh).pipe(
      Effect.tap(() => refresh === "cached" ? Effect.void : persistIndex),
    ),
  }
  const distribution = yield* LlamaCpp.makeLlamaDistribution({
    configuredExecutable: options.configuredExecutable,
    managedRoot: path.join(options.root, "llamacpp", "distribution"),
    manifest: LlamaCpp.DEFAULT_LLAMA_DISTRIBUTION_MANIFEST,
    platform: platform(),
    nativeArchitecture: host.nativeArchitecture,
    searchPath: (process.env.PATH ?? "").split(delimiter).filter(Boolean),
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
  const lock = yield* Effect.makeSemaphore(1)

  const resolveActive = lock.withPermits(1)(Effect.gen(function* () {
    const external = yield* options.external
    const externalKey = external.map((item) => `${item.id}:${item.origin}:${Option.match(item.apiKey, {
      onNone: () => "none",
      onSome: (value) => createHash("sha256").update(Redacted.value(value)).digest("hex"),
    })}`).join("\0")
    const current = yield* Ref.get(active)
    if (Option.isSome(current) && current.value.externalKey === externalKey && Option.isSome(current.value.cli)) return current.value
    const binary = yield* distribution.resolve.pipe(Effect.option)
    const key = [
      Option.match(binary, { onNone: () => "no-binary", onSome: (value) => value.fingerprint }),
      externalKey,
    ].join("\0")
    if (Option.isSome(current) && current.value.key === key) return current.value
    if (Option.isSome(current)) yield* Scope.close(current.value.scope, Exit.void)
    const cli = yield* Option.match(binary, {
      onNone: () => Effect.succeed(Option.none<LlamaCpp.LlamaCli>()),
      onSome: (value) => LlamaCpp.makeLlamaCli().pipe(
        Effect.provideService(LlamaCpp.LlamaBinary, value),
        Effect.provideService(CommandExecutor.CommandExecutor, executor),
        Effect.map(Option.some),
      ),
    })
    const scope = yield* Scope.make()
    const port = yield* freePort()
    if (port <= 0) return yield* Effect.die("Unable to reserve a loopback port for llama.cpp")
    const registry = yield* LlamaCpp.makeLlamaInstanceRegistry({
      cli,
      modelFiles: files,
      presetPath: path.join(options.root, "llamacpp", "runtime", "models.ini"),
      host: "127.0.0.1",
      port,
      apiKey: Redacted.make(randomUUID()),
      modelsMax: 1,
      external: external.map((item) => ({
        id: LlamaCpp.ExternalServerConfigId.make(item.id),
        origin: item.origin,
        authorization: item.apiKey,
        label: Option.some(item.id),
      })),
    }).pipe(
      Effect.provideService(Scope.Scope, scope),
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.provideService(Path.Path, path),
      Effect.provideService(HttpClient.HttpClient, http),
    )
    const next = { key, externalKey, scope, cli, registry }
    yield* Ref.set(active, Option.some(next))
    return next
  }))

  yield* Effect.addFinalizer(() => Ref.get(active).pipe(Effect.flatMap(Option.match({
    onNone: () => Effect.void,
    onSome: (value) => Scope.close(value.scope, Exit.void),
  }))))

  return {
    files,
    hardware,
    distribution,
    hub,
    downloads,
    cli: resolveActive.pipe(Effect.flatMap((value) => Option.match(value.cli, {
      onNone: () => distribution.resolve.pipe(Effect.flatMap((binary) => LlamaCpp.makeLlamaCli().pipe(
        Effect.provideService(LlamaCpp.LlamaBinary, binary),
        Effect.provideService(CommandExecutor.CommandExecutor, executor),
      ))),
      onSome: Effect.succeed,
    }))),
    instances: resolveActive.pipe(Effect.map((value) => value.registry)),
  }
}))
