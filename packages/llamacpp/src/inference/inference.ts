import { Context, Effect, Layer, pipe, Ref, Scope } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import { BunCommandExecutor, BunFileSystem, BunPath } from "@effect/platform-bun"
import { FetchHttpClient } from "@effect/platform"
import * as Path from "@effect/platform/Path"
import {
  LlamaCppServerStartFailed,
  LlamaCppServerTimeout,
  LlamaCppServerOutOfMemory,
  LlamaCppModelNotFound,
  LlamaCppBinaryNotFound,
  LlamaCppBinaryVersionTooOld,
  LlamaCppBinaryDownloadFailed,
  LlamaCppUnsupportedPlatform,
  LlamaCppBinaryValidationFailed,
  LlamaCppEndpointError,
} from "../errors"
import type { LlamaCppInstancesApi } from "./instances"
import { type LlamaCppServerApi, type ServerHandle, makeLlamaCppServer } from "./server"
import type {
  LoadedModel,
  AvailableModel,
  EnsureModelOptions,
  PresetDefaults,
  LoadType,
} from "./types"
import type { LocalModelInfo } from "../models/types"
import type { LlamaCppBinaryApi } from "../binary/resolve"
import type { LlamaCppModelStoreApi } from "../models/store"

// ── Service Tag ──

export interface LlamaCppInferenceApi {
  readonly ensureModelLoaded: (
    modelId: string,
    options?: EnsureModelOptions,
  ) => Effect.Effect<
    LoadedModel,
    LlamaCppServerStartFailed | LlamaCppServerTimeout | LlamaCppServerOutOfMemory
      | LlamaCppModelNotFound | LlamaCppBinaryNotFound | LlamaCppBinaryVersionTooOld
      | LlamaCppBinaryDownloadFailed | LlamaCppUnsupportedPlatform
      | LlamaCppBinaryValidationFailed | LlamaCppEndpointError,
    FileSystem.FileSystem | HttpClient.HttpClient | CommandExecutor.CommandExecutor | Path.Path | Scope.Scope
  >

  readonly unloadModel: (modelId: string) => Effect.Effect<
    void,
    LlamaCppEndpointError,
    HttpClient.HttpClient | CommandExecutor.CommandExecutor | FileSystem.FileSystem
  >

  readonly listAvailableModels: () => Effect.Effect<
    readonly AvailableModel[],
    never,
    FileSystem.FileSystem | HttpClient.HttpClient | CommandExecutor.CommandExecutor
  >

  readonly getEndpoint: (modelId: string) => Effect.Effect<
    string | null,
    never,
    HttpClient.HttpClient | CommandExecutor.CommandExecutor
  >
}

export class LlamaCppInference extends Context.Tag("LlamaCppInference")<
  LlamaCppInference,
  LlamaCppInferenceApi
>() {}

// ── Platform layer (baked in) ──

const PlatformLayer = Layer.mergeAll(
  BunCommandExecutor.layer,
  BunPath.layer,
  FetchHttpClient.layer,
).pipe(Layer.provideMerge(BunFileSystem.layer))

// ── Factory ──

export interface LlamaCppInferenceDeps {
  readonly binary: LlamaCppBinaryApi
  readonly modelStore: LlamaCppModelStoreApi
  readonly instances: LlamaCppInstancesApi
  readonly configuredEndpoint?: string
}

export function makeLlamaCppInference(
  deps: LlamaCppInferenceDeps,
): LlamaCppInferenceApi {
  const loadedByUsRef = Effect.runSync(Ref.make<Map<string, string>>(new Map()))
  const server: LlamaCppServerApi = makeLlamaCppServer()

  const ensureModelLoaded: LlamaCppInferenceApi["ensureModelLoaded"] = (modelId, options) =>
    doEnsureModelLoaded(deps, server, loadedByUsRef, modelId, options).pipe(
      Effect.provide(PlatformLayer),
    )

  const unloadModel: LlamaCppInferenceApi["unloadModel"] = (modelId) =>
    doUnloadModel(deps, loadedByUsRef, modelId).pipe(Effect.provide(PlatformLayer))

  const listAvailableModels: LlamaCppInferenceApi["listAvailableModels"] = () =>
    doListAvailableModels(deps).pipe(Effect.provide(PlatformLayer))

  const getEndpoint: LlamaCppInferenceApi["getEndpoint"] = (modelId) =>
    doGetEndpoint(deps, modelId).pipe(Effect.provide(PlatformLayer))

  return { ensureModelLoaded, unloadModel, listAvailableModels, getEndpoint }
}

// ── ensureModelLoaded ──

function doEnsureModelLoaded(
  deps: LlamaCppInferenceDeps,
  server: LlamaCppServerApi,
  loadedByUsRef: Ref.Ref<Map<string, string>>,
  modelId: string,
  options?: EnsureModelOptions,
): Effect.Effect<
  LoadedModel,
  | LlamaCppServerStartFailed | LlamaCppServerTimeout | LlamaCppServerOutOfMemory
  | LlamaCppModelNotFound | LlamaCppBinaryNotFound | LlamaCppBinaryVersionTooOld
  | LlamaCppBinaryDownloadFailed | LlamaCppUnsupportedPlatform
  | LlamaCppBinaryValidationFailed | LlamaCppEndpointError,
  FileSystem.FileSystem | HttpClient.HttpClient | CommandExecutor.CommandExecutor | Path.Path | Scope.Scope
> {
  return Effect.gen(function* () {
    // 1. List all running instances (fresh)
    const instances = yield* deps.instances.list()

    // 2. Is modelId already loaded on ANY instance?
    for (const instance of instances) {
      const modelRef = instance.models.find(
        (m) => m.id === modelId && (m.status === "loaded" || m.status === "sleeping"),
      )
      if (modelRef) {
        const contextSize = yield* getContextSize(instance.endpoint)
        yield* Ref.update(loadedByUsRef, (map) => new Map(map).set(modelId, instance.id))
        return {
          endpoint: `${instance.endpoint}/v1`,
          modelId,
          contextSize,
          loadType: "already-loaded" as LoadType,
          instanceId: instance.id,
        } satisfies LoadedModel
      }
    }

    // 3. Resolve model file path
    const model = yield* deps.modelStore.get(modelId)
    if (!model) {
      return yield* new LlamaCppModelNotFound({ modelId })
    }

    // 4. Find router mode instances
    const routers = instances.filter(
      (i) => i.mode === "router" && i.health === "healthy",
    )
    const adoptedRouters = routers.filter((r) => !r.managed)
    const managedRouters = routers.filter((r) => r.managed)

    // 4a. Try adopted routers first
    for (const router of adoptedRouters) {
      const loaded = yield* tryLoadModelOnRouter(router.endpoint, model.filePath, modelId)
      if (loaded) {
        const contextSize = yield* getContextSize(router.endpoint)
        yield* Ref.update(loadedByUsRef, (map) => new Map(map).set(modelId, router.id))
        return {
          endpoint: `${router.endpoint}/v1`,
          modelId,
          contextSize,
          loadType: "hot-swapped" as LoadType,
          instanceId: router.id,
        } satisfies LoadedModel
      }
    }

    // 4b. Try our managed routers
    for (const router of managedRouters) {
      const loaded = yield* tryLoadModelOnRouter(router.endpoint, model.filePath, modelId)
      if (loaded) {
        const contextSize = yield* getContextSize(router.endpoint)
        yield* Ref.update(loadedByUsRef, (map) => new Map(map).set(modelId, router.id))
        return {
          endpoint: `${router.endpoint}/v1`,
          modelId,
          contextSize,
          loadType: "hot-swapped" as LoadType,
          instanceId: router.id,
        } satisfies LoadedModel
      }
    }

    // 5. Start our own managed router
    const binary = yield* deps.binary.resolve({
      gpuPreference: options?.gpuPreference ?? "auto",
    })

    const presetModels: LocalModelInfo[] = [model]
    if (options?.additionalModels) {
      for (const additionalId of options.additionalModels) {
        const additional = yield* deps.modelStore.get(additionalId)
        if (additional) presetModels.push(additional)
      }
    }

    const defaults: PresetDefaults = {
      ngl: options?.gpuLayers ?? -1,
      ctx: options?.contextSize ?? 0,
      sleepIdleSeconds: options?.sleepIdleSeconds,
      loadOnStartup: modelId,
    }

    const handle = yield* server.start(
      binary,
      presetModels,
      modelId,
      defaults,
      options?.port,
    )

    const instanceId = `managed-${handle.port}`
    yield* Ref.update(loadedByUsRef, (map) => new Map(map).set(modelId, instanceId))

    const contextSize = yield* getContextSize(handle.endpoint)
    return {
      endpoint: `${handle.endpoint}/v1`,
      modelId,
      contextSize,
      loadType: "server-started" as LoadType,
      instanceId,
    } satisfies LoadedModel
  })
}

// ── unloadModel ──

function doUnloadModel(
  deps: LlamaCppInferenceDeps,
  loadedByUsRef: Ref.Ref<Map<string, string>>,
  modelId: string,
): Effect.Effect<
  void,
  LlamaCppEndpointError,
  HttpClient.HttpClient | CommandExecutor.CommandExecutor | FileSystem.FileSystem
> {
  return Effect.gen(function* () {
    const loadedByUs = yield* Ref.get(loadedByUsRef)
    const instanceId = loadedByUs.get(modelId)
    if (!instanceId) return

    const instances = yield* deps.instances.list()
    const instance = instances.find((i) => i.id === instanceId)
    if (!instance) {
      yield* Ref.update(loadedByUsRef, (map) => {
        const next = new Map(map)
        next.delete(modelId)
        return next
      })
      return
    }

    if (instance.mode === "router" && instance.capabilities.canHotSwap) {
      const client = yield* HttpClient.HttpClient
      const req = yield* pipe(
        HttpClientRequest.bodyJson({ model: modelId })(
          HttpClientRequest.post(`${instance.endpoint}/models/unload`),
        ),
        Effect.mapError((err) => new LlamaCppEndpointError({ reason: String(err) })),
      )
      yield* pipe(
        client.execute(req),
        Effect.mapError((err) => new LlamaCppEndpointError({ reason: String(err) })),
      )
    } else if (instance.managed) {
      yield* deps.instances.stop(instanceId)
    }

    yield* Ref.update(loadedByUsRef, (map) => {
      const next = new Map(map)
      next.delete(modelId)
      return next
    })
  })
}

// ── listAvailableModels ──

function doListAvailableModels(
  deps: LlamaCppInferenceDeps,
): Effect.Effect<
  readonly AvailableModel[],
  never,
  FileSystem.FileSystem | HttpClient.HttpClient | CommandExecutor.CommandExecutor
> {
  return Effect.gen(function* () {
    const diskModels = yield* deps.modelStore.discover()
    const instances = yield* deps.instances.list()

    const byPath = new Map<string, AvailableModel>()

    for (const model of diskModels) {
      byPath.set(model.filePath, {
        id: model.id,
        displayName: model.displayName,
        availability: "available",
        endpoint: null,
        instanceId: null,
        info: model,
      })
    }

    for (const instance of instances) {
      for (const modelRef of instance.models) {
        const filePath = modelRef.path ?? modelRef.id
        const existing = byPath.get(filePath)

        if (existing) {
          if (!existing.endpoint || !instance.managed) {
            byPath.set(filePath, {
              ...existing,
              availability: modelRef.status === "sleeping" ? "sleeping"
                : modelRef.status === "loading" ? "loading"
                : "loaded",
              endpoint: `${instance.endpoint}/v1`,
              instanceId: instance.id,
            })
          }
        } else {
          byPath.set(filePath, {
            id: modelRef.id,
            displayName: modelRef.id,
            availability: modelRef.status === "sleeping" ? "sleeping"
              : modelRef.status === "loading" ? "loading"
              : "loaded",
            endpoint: `${instance.endpoint}/v1`,
            instanceId: instance.id,
            info: {
              id: modelRef.id,
              displayName: modelRef.id,
              filePath,
              fileSizeBytes: 0,
              vision: false,
              audio: false,
              moe: false,
              source: { _tag: "user-dir" as const, dir: filePath },
            },
          })
        }
      }
    }

    return Array.from(byPath.values())
  })
}

// ── getEndpoint ──

function doGetEndpoint(
  deps: LlamaCppInferenceDeps,
  modelId: string,
): Effect.Effect<string | null, never, HttpClient.HttpClient | CommandExecutor.CommandExecutor> {
  return Effect.gen(function* () {
    const instances = yield* deps.instances.list()
    for (const instance of instances) {
      const modelRef = instance.models.find(
        (m) => m.id === modelId && (m.status === "loaded" || m.status === "sleeping"),
      )
      if (modelRef) {
        return `${instance.endpoint}/v1`
      }
    }
    return null
  })
}

// ── Helpers ──

function tryLoadModelOnRouter(
  endpoint: string,
  modelPath: string,
  modelId: string,
): Effect.Effect<boolean, never, HttpClient.HttpClient> {
  return Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    const req = yield* pipe(
      HttpClientRequest.bodyJson({ model: modelPath, alias: modelId })(
        HttpClientRequest.post(`${endpoint}/models/load`),
      ),
      Effect.catchAll(() => Effect.succeed(null as HttpClientRequest.HttpClientRequest | null)),
    )
    if (!req) return false
    const res = yield* pipe(
      client.execute(req),
      Effect.timeout("30 seconds"),
      Effect.catchAll(() => Effect.succeed(null)),
    )
    if (!res) return false
    return res.status >= 200 && res.status < 300
  })
}

function getContextSize(
  endpoint: string,
): Effect.Effect<number, never, HttpClient.HttpClient> {
  return Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    const res = yield* pipe(
      client.execute(HttpClientRequest.get(`${endpoint}/props`)),
      Effect.timeout("5 seconds"),
      Effect.catchAll(() => Effect.succeed(null)),
    )
    if (!res || res.status < 200 || res.status >= 300) return 0

    const body = yield* pipe(
      res.json,
      Effect.orElseSucceed(() => null),
    )
    if (typeof body !== "object" || body === null) return 0
    const props = body as Record<string, unknown>
    const ctx = props.default_generation_settings
    if (typeof ctx !== "object" || ctx === null) return 0
    const ctxObj = ctx as Record<string, unknown>
    const nCtx = ctxObj.n_ctx
    return typeof nCtx === "number" ? nCtx : 0
  })
}
