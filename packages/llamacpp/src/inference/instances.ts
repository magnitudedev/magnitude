import { Context, Effect, Layer, pipe, Ref, Scope } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as FileSystem from "@effect/platform/FileSystem"
import { BunCommandExecutor, BunFileSystem, BunPath } from "@effect/platform-bun"
import { FetchHttpClient } from "@effect/platform"
import * as Path from "@effect/platform/Path"
import {
  LlamaCppServerStartFailed,
  LlamaCppServerTimeout,
  LlamaCppServerOutOfMemory,
  LlamaCppEndpointError,
  LlamaCppBinaryNotFound,
  LlamaCppBinaryVersionTooOld,
  LlamaCppBinaryDownloadFailed,
  LlamaCppUnsupportedPlatform,
  LlamaCppBinaryValidationFailed,
} from "../errors"
import { type LlamaCppDetectorApi, makeLlamaCppDetector, type LlamaCppDetectorDeps } from "./detector"
import { type LlamaCppServerApi, type ServerHandle, makeLlamaCppServer } from "./server"
import type {
  InstanceInfo,
  InstanceOptions,
  InstanceCapabilities,
  InstanceModelRef,
  ServerMode,
  PresetDefaults,
} from "./types"
import type { LocalModelInfo } from "../models/types"
import type { LlamaCppBinaryApi } from "../binary/resolve"
import type { LlamaCppModelStoreApi } from "../models/store"

// ── Service Tag ──

export interface LlamaCppInstancesApi {
  readonly list: () => Effect.Effect<
    readonly InstanceInfo[],
    never,
    HttpClient.HttpClient | CommandExecutor.CommandExecutor
  >

  readonly get: (instanceId: string) => Effect.Effect<
    InstanceInfo | null,
    never,
    HttpClient.HttpClient | CommandExecutor.CommandExecutor
  >

  readonly stop: (instanceId: string) => Effect.Effect<void, never, FileSystem.FileSystem>

  readonly restart: (
    instanceId: string,
    options?: InstanceOptions,
  ) => Effect.Effect<
    InstanceInfo,
    LlamaCppServerStartFailed | LlamaCppServerTimeout | LlamaCppServerOutOfMemory
      | LlamaCppEndpointError
      | LlamaCppBinaryNotFound | LlamaCppBinaryVersionTooOld
      | LlamaCppBinaryDownloadFailed | LlamaCppUnsupportedPlatform
      | LlamaCppBinaryValidationFailed,
    FileSystem.FileSystem | HttpClient.HttpClient | CommandExecutor.CommandExecutor | Path.Path | Scope.Scope
  >

  readonly stopAllManaged: () => Effect.Effect<void, never, FileSystem.FileSystem>
}

export class LlamaCppInstances extends Context.Tag("LlamaCppInstances")<
  LlamaCppInstances,
  LlamaCppInstancesApi
>() {}

// ── Platform layer (baked in) ──

const PlatformLayer = Layer.mergeAll(
  BunCommandExecutor.layer,
  BunPath.layer,
  FetchHttpClient.layer,
).pipe(Layer.provideMerge(BunFileSystem.layer))

// ── Factory ──

export interface LlamaCppInstancesDeps {
  readonly binary: LlamaCppBinaryApi
  readonly modelStore: LlamaCppModelStoreApi
  readonly configuredEndpoint?: string
}

export function makeLlamaCppInstances(
  deps: LlamaCppInstancesDeps,
): LlamaCppInstancesApi {
  const managedHandles = Effect.runSync(Ref.make<Map<string, ServerHandle>>(new Map()))

  const detector: LlamaCppDetectorApi = makeLlamaCppDetector({
    configuredEndpoint: deps.configuredEndpoint,
  } as LlamaCppDetectorDeps)

  const server: LlamaCppServerApi = makeLlamaCppServer()

  const list: LlamaCppInstancesApi["list"] = () =>
    listInstances(detector).pipe(Effect.provide(PlatformLayer))

  const get: LlamaCppInstancesApi["get"] = (instanceId) =>
    Effect.gen(function* () {
      const instances = yield* listInstances(detector).pipe(Effect.provide(PlatformLayer))
      return instances.find((i) => i.id === instanceId) ?? null
    })

  const stop: LlamaCppInstancesApi["stop"] = (instanceId) =>
    Effect.gen(function* () {
      const handles = yield* Ref.get(managedHandles)
      const handle = handles.get(instanceId)
      if (!handle) return
      yield* server.stop(handle)
      yield* Ref.update(managedHandles, (map) => {
        const next = new Map(map)
        next.delete(instanceId)
        return next
      })
    })

  const restart: LlamaCppInstancesApi["restart"] = (instanceId, options) =>
    Effect.gen(function* () {
      const handles = yield* Ref.get(managedHandles)
      const oldHandle = handles.get(instanceId)
      if (!oldHandle) {
        return yield* new LlamaCppEndpointError({ reason: `Instance ${instanceId} is not managed by us` })
      }

      yield* server.stop(oldHandle)

      // Find what models were loaded
      const instances = yield* listInstances(detector).pipe(Effect.provide(PlatformLayer))
      const oldInstance = instances.find((i) => i.id === instanceId)
      const loadedModelIds = oldInstance?.models
        .filter((m) => m.status === "loaded" || m.status === "sleeping")
        .map((m) => m.id) ?? []

      const models: LocalModelInfo[] = []
      for (const modelId of loadedModelIds) {
        const model = yield* deps.modelStore.get(modelId)
        if (model) models.push(model)
      }

      const binary = yield* deps.binary.resolve({
        gpuPreference: options?.gpuPreference ?? "auto",
      })

      const defaults: PresetDefaults = {
        ngl: options?.gpuLayers ?? -1,
        ctx: options?.contextSize ?? 0,
        sleepIdleSeconds: options?.sleepIdleSeconds,
        loadOnStartup: loadedModelIds[0],
      }

      const newHandle = yield* server.start(
        binary,
        models,
        loadedModelIds[0] ?? "",
        defaults,
        options?.port ?? oldHandle.port,
      )

      const newId = `managed-${newHandle.port}`
      yield* Ref.update(managedHandles, (map) => {
        const next = new Map(map)
        next.delete(instanceId)
        next.set(newId, newHandle)
        return next
      })

      return buildInstanceInfo(newId, newHandle.port, newHandle.endpoint, newHandle.mode, true, [], null)
    })

  const stopAllManaged: LlamaCppInstancesApi["stopAllManaged"] = () =>
    Effect.gen(function* () {
      const handles = yield* Ref.get(managedHandles)
      for (const [, handle] of handles) {
        yield* server.stop(handle)
      }
      yield* Ref.set(managedHandles, new Map())
    })

  return { list, get, stop, restart, stopAllManaged }
}

// ── List implementation ──

function listInstances(
  detector: LlamaCppDetectorApi,
): Effect.Effect<
  readonly InstanceInfo[],
  never,
  HttpClient.HttpClient | CommandExecutor.CommandExecutor
> {
  return Effect.gen(function* () {
    const detected = yield* detector.detect()

    return detected.map((server) => {
      const id = `adopted-127.0.0.1-${server.port}`
      const capabilities: InstanceCapabilities = {
        canManage: false,
        canHotSwap: server.mode === "router",
      }
      return buildInstanceInfo(
        id,
        server.port,
        server.endpoint,
        server.mode,
        false,
        server.models,
        server.buildInfo,
      )
    })
  })
}

function buildInstanceInfo(
  id: string,
  port: number,
  endpoint: string,
  mode: ServerMode,
  managed: boolean,
  models: readonly InstanceModelRef[],
  buildInfo: string | null,
): InstanceInfo {
  return {
    id,
    endpoint,
    port,
    mode,
    health: "healthy",
    managed,
    pid: null,
    capabilities: {
      canManage: managed,
      canHotSwap: mode === "router",
    },
    models,
    buildInfo,
  }
}
