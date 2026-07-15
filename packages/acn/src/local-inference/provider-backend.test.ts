import { describe, expect, it } from "vitest"
import { Effect, Layer, Option, Stream } from "effect"
import { FetchHttpClient } from "@effect/platform"
import {
  LlamaCppHost,
  LlamaCppModelStore,
  LlamaCppRuntime,
  type LlamaCppHostApi,
  type LlamaCppModelStoreApi,
  type LlamaCppRuntimeApi,
} from "@magnitudedev/llamacpp"
import {
  LocalModelConfiguration,
  type LocalModelConfigurationApi,
} from "./model-configuration"
import {
  LocalModelProviderBackend,
  LocalModelProviderBackendLive,
} from "./provider-backend"
import { providerModelIdForArtifact } from "./identity"

const host: LlamaCppHostApi = {
  inspect: Effect.die("external bindings do not inspect host fit"),
  plan: () => Effect.die("external bindings do not plan managed placement"),
}

const models: LlamaCppModelStoreApi = {
  inspect: Effect.succeed({ artifacts: [], warnings: [] }),
  resolve: () => Effect.die("external bindings do not resolve model artifacts"),
  download: () => Stream.empty,
  deleteOwned: () => Effect.void,
}

const configuration: LocalModelConfigurationApi = {
  get: Effect.succeed({
    usage: { localModelRole: "main", sessionConcurrency: "one" },
    binding: {
      _tag: "External",
      selectionId: "external-choice",
      endpointConfigId: "llamacpp",
      providerModelId: "external-model",
      contextTokens: 48_000,
    },
  }),
  updateUsage: () => Effect.void,
  updateSlots: () => Effect.void,
  activateLocal: () => Effect.void,
  disableLocal: Effect.void,
  changes: Stream.empty,
}

describe("LocalModelProviderBackend", () => {
  it("lists an active external binding and resolves it through the runtime contract", async () => {
    const requests: unknown[] = []
    const connection = { baseUrl: "http://127.0.0.1:9090", apiKey: Option.none() }
    const runtime: LlamaCppRuntimeApi = {
      inspect: Effect.succeed({
        managed: null,
        external: [{
          serverId: "llamacpp",
          ownership: "external",
          health: "ready",
          models: [{ providerModelId: "external-model", contextTokens: 48_000 }],
          build: "test",
        }],
      }),
      ensureServing: (request) => Effect.sync(() => {
        requests.push(request)
        return {
          serverId: "llamacpp",
          ownership: "external" as const,
          providerModelId: request.providerModelId,
          configuredContextTokens: request.contextTokens,
          metadata: { architecture: null, quantization: null, sizeBytes: null },
          connection,
        }
      }),
      stopManaged: Effect.void,
    }
    const backendLayer = LocalModelProviderBackendLive.pipe(Layer.provide(Layer.mergeAll(
      Layer.succeed(LlamaCppHost, host),
      Layer.succeed(LlamaCppModelStore, models),
      Layer.succeed(LlamaCppRuntime, runtime),
      Layer.succeed(LocalModelConfiguration, configuration),
    )))
    const layer = Layer.merge(backendLayer, FetchHttpClient.layer)

    const catalog = await Effect.runPromise(
      Effect.flatMap(LocalModelProviderBackend, (backend) => backend.listModels).pipe(
        Effect.provide(layer),
      ),
    )
    expect(requests).toEqual([])
    const resolved = await Effect.runPromise(
      Effect.flatMap(LocalModelProviderBackend, (backend) => backend.resolveConnection("external-model")).pipe(
        Effect.provide(layer),
      ),
    )

    expect(catalog).toEqual([expect.objectContaining({
      providerId: "llamacpp",
      providerModelId: "external-model",
      contextWindow: 48_000,
      serverContextSize: 48_000,
    })])
    expect(resolved).toEqual(connection)
    expect(requests).toEqual([{
      _tag: "External",
      connectionId: "llamacpp",
      providerModelId: "external-model",
      contextTokens: 48_000,
    }])
  })

  it("advertises only the active managed binding", async () => {
    const activeArtifactId = "active-artifact"
    const inactiveArtifactId = "inactive-artifact"
    const activeProviderModelId = providerModelIdForArtifact(activeArtifactId)
    const artifact = (modelId: string, displayName: string) => ({
      modelId,
      source: { _tag: "MagnitudeOwned" as const, manifestId: modelId },
      sizeBytes: 1_024,
      metadata: {
        displayName,
        architecture: "llama",
        quantization: "Q4_K_M",
        contextLength: 131_072,
        parameterCount: null,
        layerCount: 32,
        tokenizerModel: "llama",
        tokenizerPre: null,
        baseModelNames: [],
      },
      hasVisionProjector: false,
    })
    const managedModels: LlamaCppModelStoreApi = {
      ...models,
      inspect: Effect.succeed({
        artifacts: [
          artifact(activeArtifactId, "Active model"),
          artifact(inactiveArtifactId, "Inactive model"),
        ],
        warnings: [],
      }),
    }
    const managedConfiguration: LocalModelConfigurationApi = {
      ...configuration,
      get: Effect.succeed({
        usage: { localModelRole: "main", sessionConcurrency: "one" },
        binding: {
          _tag: "Managed",
          selectionId: "active-selection",
          artifactId: activeArtifactId,
          providerModelId: activeProviderModelId,
          contextTokens: 65_536,
          parallelSlots: 1,
        },
      }),
    }
    const runtime: LlamaCppRuntimeApi = {
      inspect: Effect.succeed({ managed: null, external: [] }),
      ensureServing: () => Effect.die("catalog inspection must not start the runtime"),
      stopManaged: Effect.void,
    }
    const backendLayer = LocalModelProviderBackendLive.pipe(Layer.provide(Layer.mergeAll(
      Layer.succeed(LlamaCppHost, host),
      Layer.succeed(LlamaCppModelStore, managedModels),
      Layer.succeed(LlamaCppRuntime, runtime),
      Layer.succeed(LocalModelConfiguration, managedConfiguration),
    )))

    const catalog = await Effect.runPromise(
      Effect.flatMap(LocalModelProviderBackend, (backend) => backend.listModels).pipe(
        Effect.provide(Layer.merge(backendLayer, FetchHttpClient.layer)),
      ),
    )

    expect(catalog).toEqual([expect.objectContaining({
      providerModelId: activeProviderModelId,
      displayName: "Active model",
      contextWindow: 65_536,
      serverContextSize: 65_536,
    })])
  })

  it("lazily restores a durable managed binding when a successor resolves inference", async () => {
    const artifactId = "durable-artifact"
    const providerModelId = providerModelIdForArtifact(artifactId)
    const connection = { baseUrl: "http://127.0.0.1:18080", apiKey: Option.none() }
    const requests: unknown[] = []
    const managedHost: LlamaCppHostApi = {
      inspect: Effect.die("connection resolution does not inspect the host"),
      plan: (request) => Effect.succeed({
        requiredBytes: request.modelBytes + request.contextBytesPerSlot * request.parallelSlots,
        stableCapacityBytes: 64 * 1024 ** 3,
        parallelSlots: request.parallelSlots,
        gpuLayers: 32,
        splitMode: "none",
        fits: true,
      }),
    }
    const managedModels: LlamaCppModelStoreApi = {
      ...models,
      resolve: (requestedId) => Effect.succeed({
        modelId: requestedId,
        source: { _tag: "MagnitudeOwned", manifestId: requestedId },
        sizeBytes: 4 * 1024 ** 3,
        metadata: {
          displayName: "Durable model",
          architecture: "llama",
          quantization: "Q4_K_M",
          contextLength: 131_072,
          parameterCount: null,
          layerCount: 32,
          tokenizerModel: "llama",
          tokenizerPre: null,
          baseModelNames: [],
        },
        hasVisionProjector: false,
        primaryPath: "/managed/model.gguf",
        shardPaths: ["/managed/model.gguf"],
        projectorPath: null,
      }),
    }
    const managedConfiguration: LocalModelConfigurationApi = {
      ...configuration,
      get: Effect.succeed({
        usage: { localModelRole: "main", sessionConcurrency: "one" },
        binding: {
          _tag: "Managed",
          selectionId: "durable-selection",
          artifactId,
          providerModelId,
          contextTokens: 65_536,
          parallelSlots: 1,
        },
      }),
    }
    const runtime: LlamaCppRuntimeApi = {
      inspect: Effect.succeed({ managed: null, external: [] }),
      ensureServing: (request) => Effect.sync(() => {
        requests.push(request)
        return {
          serverId: "successor-managed-server",
          ownership: "managed",
          providerModelId: request.providerModelId,
          configuredContextTokens: request.contextTokens,
          metadata: { architecture: "llama", quantization: "Q4_K_M", sizeBytes: 4 * 1024 ** 3 },
          connection,
        }
      }),
      stopManaged: Effect.void,
    }
    const backendLayer = LocalModelProviderBackendLive.pipe(Layer.provide(Layer.mergeAll(
      Layer.succeed(LlamaCppHost, managedHost),
      Layer.succeed(LlamaCppModelStore, managedModels),
      Layer.succeed(LlamaCppRuntime, runtime),
      Layer.succeed(LocalModelConfiguration, managedConfiguration),
    )))

    const resolved = await Effect.runPromise(
      Effect.flatMap(
        LocalModelProviderBackend,
        (backend) => backend.resolveConnection(providerModelId),
      ).pipe(Effect.provide(Layer.merge(backendLayer, FetchHttpClient.layer))),
    )

    expect(resolved).toEqual(connection)
    expect(requests).toEqual([expect.objectContaining({
      _tag: "Managed",
      modelId: artifactId,
      providerModelId,
      contextTokens: 65_536,
    })])
  })

  it("does not advertise stored artifacts without an active binding", async () => {
    const inactiveModels: LlamaCppModelStoreApi = {
      ...models,
      inspect: Effect.succeed({
        artifacts: [{
          modelId: "stored-artifact",
          source: { _tag: "MagnitudeOwned", manifestId: "stored-artifact" },
          sizeBytes: 1_024,
          metadata: {
            displayName: "Stored model",
            architecture: null,
            quantization: null,
            contextLength: 32_768,
            parameterCount: null,
            layerCount: null,
            tokenizerModel: null,
            tokenizerPre: null,
            baseModelNames: [],
          },
          hasVisionProjector: false,
        }],
        warnings: [],
      }),
    }
    const unboundConfiguration: LocalModelConfigurationApi = {
      ...configuration,
      get: Effect.succeed({
        usage: { localModelRole: "main", sessionConcurrency: "one" },
      }),
    }
    const runtime: LlamaCppRuntimeApi = {
      inspect: Effect.succeed({ managed: null, external: [] }),
      ensureServing: () => Effect.die("catalog inspection must not start the runtime"),
      stopManaged: Effect.void,
    }
    const backendLayer = LocalModelProviderBackendLive.pipe(Layer.provide(Layer.mergeAll(
      Layer.succeed(LlamaCppHost, host),
      Layer.succeed(LlamaCppModelStore, inactiveModels),
      Layer.succeed(LlamaCppRuntime, runtime),
      Layer.succeed(LocalModelConfiguration, unboundConfiguration),
    )))

    const catalog = await Effect.runPromise(
      Effect.flatMap(LocalModelProviderBackend, (backend) => backend.listModels).pipe(
        Effect.provide(Layer.merge(backendLayer, FetchHttpClient.layer)),
      ),
    )

    expect(catalog).toEqual([])
  })
})
