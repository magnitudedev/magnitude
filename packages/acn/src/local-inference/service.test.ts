import { describe, expect, it } from "vitest"
import { Deferred, Effect, Fiber, Layer, Option, Stream } from "effect"
import {
  LlamaCppDistribution,
  LlamaCppHost,
  LlamaCppModelStore,
  LlamaCppRuntime,
  DistributionInstallError,
  LlamaCppModelStoreError,
  LlamaCppRuntimeError,
  type LlamaCppDistributionApi,
  type LlamaCppHostApi,
  type LlamaCppHostProfile,
  type LlamaCppModelStoreApi,
  type LlamaCppRuntimeApi,
  type ModelArtifactSummary,
} from "@magnitudedev/llamacpp"
import type { DurableLocalModelBinding, LocalInferenceConfig } from "@magnitudedev/storage"
import {
  LocalInference,
  LocalInferenceLive,
  localInferenceErrorFromDistribution,
  localInferenceErrorFromModelStore,
  localInferenceErrorFromRuntime,
  type LocalInferenceApi,
} from "./service"
import {
  LocalModelConfiguration,
  type LocalModelConfigurationApi,
} from "./model-configuration"

const hostProfile: LlamaCppHostProfile = {
  system: { totalMemoryBytes: 64 * 1024 ** 3, cpuModel: "test", logicalCores: 8 },
  memoryDomains: [{
    id: "system",
    kind: "system",
    stableCapacityBytes: 56 * 1024 ** 3,
    currentFreeBytes: null,
    sharesSystemMemory: false,
    devices: [],
    splitGroupId: null,
  }],
  runtimeProbe: "not_installed",
  warnings: [],
}

const artifact = (modelId: string, source: ModelArtifactSummary["source"]): ModelArtifactSummary => ({
  modelId,
  source,
  sizeBytes: 4 * 1024 ** 3,
  metadata: {
    displayName: "Discovered model",
    architecture: "llama",
    quantization: "Q4_K_M",
    contextLength: 65_536,
    parameterCount: null,
    layerCount: 32,
    tokenizerModel: null,
    tokenizerPre: null,
    baseModelNames: [],
  },
  hasVisionProjector: false,
})

const distribution: LlamaCppDistributionApi = {
  inspect: Effect.succeed({ _tag: "Missing" }),
  install: Stream.empty,
}

const host: LlamaCppHostApi = {
  inspect: Effect.succeed(hostProfile),
  plan: (request) => Effect.succeed({
    requiredBytes: request.modelBytes + request.contextBytesPerSlot * request.parallelSlots,
    stableCapacityBytes: 56 * 1024 ** 3,
    parallelSlots: request.parallelSlots,
    gpuLayers: 0,
    splitMode: "none",
    fits: true,
  }),
}

const makeConfiguration = (
  initial: LocalInferenceConfig,
  onActivate: (binding: DurableLocalModelBinding) => void = () => undefined,
): LocalModelConfigurationApi => {
  let current = initial
  return {
    get: Effect.sync(() => current),
    updateUsage: (usage) => Effect.sync(() => { current = { ...current, usage } }),
    updateSlots: () => Effect.void,
    activateLocal: (binding) => Effect.sync(() => {
      current = { ...current, binding }
      onActivate(binding)
    }),
    disableLocal: Effect.sync(() => { current = { ...current, binding: undefined } }),
    changes: Stream.empty,
  }
}

const localInferenceLayer = (
  modelStore: LlamaCppModelStoreApi,
  runtime: LlamaCppRuntimeApi,
  configuration: LocalModelConfigurationApi,
) => LocalInferenceLive.pipe(Layer.provide(Layer.mergeAll(
  Layer.succeed(LlamaCppDistribution, distribution),
  Layer.succeed(LlamaCppHost, host),
  Layer.succeed(LlamaCppModelStore, modelStore),
  Layer.succeed(LlamaCppRuntime, runtime),
  Layer.succeed(LocalModelConfiguration, configuration),
)))

describe("LocalInference service", () => {
  it("maps every package failure code into the closed product error vocabulary", () => {
    const distributionCodes = [
      ["unsupported_platform", "unsupported_platform"],
      ["download_failed", "configuration_failed"],
      ["integrity_failed", "integrity_failed"],
      ["storage_failed", "configuration_failed"],
    ] as const
    for (const [code, expected] of distributionCodes) {
      const mapped = localInferenceErrorFromDistribution(new DistributionInstallError({
        operation: "install",
        code,
        stage: "resolving",
        reason: code,
      }))
      expect(mapped.code).toBe(expected)
    }

    const storeCodes = [
      ["artifact_not_found", "artifact_unavailable"],
      ["artifact_not_owned", "artifact_not_owned"],
      ["invalid_plan", "invalid_selection"],
      ["insufficient_space", "insufficient_disk_space"],
      ["download_failed", "artifact_unavailable"],
      ["integrity_failed", "integrity_failed"],
      ["storage_failed", "configuration_failed"],
    ] as const
    for (const [code, expected] of storeCodes) {
      const mapped = localInferenceErrorFromModelStore(new LlamaCppModelStoreError({
        operation: "resolve",
        code,
        reason: code,
      }))
      expect(mapped.code).toBe(expected)
    }

    const runtimeCodes = [
      ["distribution_unavailable", "distribution_missing"],
      ["model_unavailable", "artifact_unavailable"],
      ["external_unavailable", "external_server_unavailable"],
      ["server_start_failed", "server_start_failed"],
      ["server_timeout", "server_start_failed"],
      ["identity_mismatch", "invalid_selection"],
      ["context_mismatch", "context_mismatch"],
      ["endpoint_failed", "runtime_probe_failed"],
    ] as const
    for (const [code, expected] of runtimeCodes) {
      const mapped = localInferenceErrorFromRuntime(new LlamaCppRuntimeError({
        operation: "ensure_serving",
        code,
        reason: code,
      }))
      expect(mapped.code).toBe(expected)
    }
  })

  it("composes install, recommendation, download, and activation through the four package contracts", async () => {
    let distributionReady = false
    let downloaded: ModelArtifactSummary | null = null
    const ensured: unknown[] = []
    const bindings: DurableLocalModelBinding[] = []
    const flowDistribution: LlamaCppDistributionApi = {
      inspect: Effect.sync(() => distributionReady
        ? {
            _tag: "Ready",
            distribution: {
              executablePath: "/runtime/llama-server",
              directory: "/runtime",
              build: 10011,
              source: "managed",
            },
          }
        : { _tag: "Missing" }),
      install: Stream.make(
        { _tag: "Resolving" as const },
        {
          _tag: "Ready" as const,
          distribution: {
            executablePath: "/runtime/llama-server",
            directory: "/runtime",
            build: 10011,
            source: "managed" as const,
          },
        },
      ).pipe(Stream.tap((event) => Effect.sync(() => {
        if (event._tag === "Ready") distributionReady = true
      }))),
    }
    const modelStore: LlamaCppModelStoreApi = {
      inspect: Effect.sync(() => ({ artifacts: downloaded ? [downloaded] : [], warnings: [] })),
      resolve: (modelId) => downloaded && downloaded.modelId === modelId
        ? Effect.succeed({
            ...downloaded,
            primaryPath: `/models/${modelId}.gguf`,
            shardPaths: [`/models/${modelId}.gguf`],
            projectorPath: null,
          })
        : Effect.dieMessage(`Unknown artifact ${modelId}`),
      download: (plan) => {
        const artifactSummary: ModelArtifactSummary = {
          modelId: plan.artifactId,
          source: { _tag: "MagnitudeOwned", manifestId: plan.artifactId },
          sizeBytes: plan.files.reduce((total, file) => total + file.sizeBytes, 0),
          metadata: {
            displayName: "Downloaded model",
            architecture: "qwen",
            quantization: "Q4_K_M",
            contextLength: 262_144,
            parameterCount: null,
            layerCount: 32,
            tokenizerModel: null,
            tokenizerPre: null,
            baseModelNames: [],
          },
          hasVisionProjector: false,
        }
        return Stream.make(
          { _tag: "Resolving" as const, artifactId: plan.artifactId },
          { _tag: "Ready" as const, artifact: artifactSummary },
        ).pipe(Stream.tap((event) => Effect.sync(() => {
          if (event._tag === "Ready") downloaded = artifactSummary
        })))
      },
      deleteOwned: () => Effect.void,
    }
    const runtime: LlamaCppRuntimeApi = {
      inspect: Effect.succeed({ managed: null, external: [] }),
      ensureServing: (request) => Effect.sync(() => {
        ensured.push(request)
        return {
          serverId: "managed-flow",
          ownership: "managed" as const,
          providerModelId: request.providerModelId,
          configuredContextTokens: request.contextTokens,
          metadata: { architecture: "qwen", quantization: "Q4_K_M", sizeBytes: downloaded?.sizeBytes ?? null },
          connection: { baseUrl: "http://127.0.0.1:18080", apiKey: Option.none() },
        }
      }),
      stopManaged: Effect.void,
    }
    const configuration = makeConfiguration({}, (binding) => bindings.push(binding))
    const layer = LocalInferenceLive.pipe(Layer.provide(Layer.mergeAll(
      Layer.succeed(LlamaCppDistribution, flowDistribution),
      Layer.succeed(LlamaCppHost, host),
      Layer.succeed(LlamaCppModelStore, modelStore),
      Layer.succeed(LlamaCppRuntime, runtime),
      Layer.succeed(LocalModelConfiguration, configuration),
    )))

    await Effect.runPromise(Effect.gen(function* () {
      const service = yield* LocalInference
      yield* service.installDistribution
      yield* service.configureUsage({ localModelRole: "main", sessionConcurrency: "one" })
      const recommended = (yield* service.state).recommendations[0]
      if (!recommended) return yield* Effect.dieMessage("Expected a recommendation")
      yield* service.downloadModel(recommended.configurationId)
      const stored = (yield* service.state).choices.find(
        (choice) => choice._tag === "StoredOwned" && choice.choiceId === recommended.configurationId,
      )
      if (!stored) return yield* Effect.dieMessage("Expected the downloaded recommendation to become a stored choice")
      yield* service.activateModel(stored.choiceId)
    }).pipe(Effect.provide(layer)))

    expect(distributionReady).toBe(true)
    expect(downloaded).not.toBeNull()
    expect(ensured).toHaveLength(1)
    expect(bindings).toHaveLength(1)
    expect(bindings[0]).toMatchObject({ _tag: "Managed" })
  })

  it("activates only the explicitly selected external connection", async () => {
    const ensured: unknown[] = []
    const activated: DurableLocalModelBinding[] = []
    let managedStops = 0
    const runtime: LlamaCppRuntimeApi = {
      inspect: Effect.succeed({
        managed: null,
        external: [{
          serverId: "llamacpp",
          ownership: "external",
          health: "ready",
          models: [{ providerModelId: "external-model", contextTokens: 32_768 }],
          build: "test",
        }],
      }),
      ensureServing: (request) => Effect.sync(() => {
        ensured.push(request)
        return {
          serverId: "llamacpp",
          ownership: "external" as const,
          providerModelId: "external-model",
          configuredContextTokens: 32_768,
          metadata: { architecture: null, quantization: null, sizeBytes: null },
          connection: { baseUrl: "http://127.0.0.1:8080", apiKey: Option.none() },
        }
      }),
      stopManaged: Effect.sync(() => { managedStops++ }),
    }
    const modelStore: LlamaCppModelStoreApi = {
      inspect: Effect.succeed({ artifacts: [], warnings: [] }),
      resolve: () => Effect.die("external activation must not resolve an artifact"),
      download: () => Stream.empty,
      deleteOwned: () => Effect.void,
    }
    const layer = localInferenceLayer(
      modelStore,
      runtime,
      makeConfiguration({
        usage: { localModelRole: "main", sessionConcurrency: "one" },
        binding: {
          _tag: "Managed",
          selectionId: "previous",
          artifactId: "previous-artifact",
          providerModelId: "previous-model",
          contextTokens: 32_768,
          parallelSlots: 1,
        },
      }, (binding) => activated.push(binding)),
    )

    const selectionId = await Effect.runPromise(
      Effect.flatMap(LocalInference, (service) => service.state).pipe(
        Effect.flatMap((state) => {
          const choice = state.choices.find((candidate) => candidate._tag === "RunningExternal")
          return choice ? Effect.succeed(choice.choiceId) : Effect.dieMessage("missing external choice")
        }),
        Effect.provide(layer),
      ),
    )
    await Effect.runPromise(
      Effect.flatMap(LocalInference, (service) => service.activateModel(selectionId)).pipe(Effect.provide(layer)),
    )

    expect(ensured).toEqual([{
      _tag: "External",
      connectionId: "llamacpp",
      providerModelId: "external-model",
      contextTokens: 32_768,
    }])
    expect(activated).toEqual([{
      _tag: "External",
      selectionId,
      endpointConfigId: "llamacpp",
      providerModelId: "external-model",
      contextTokens: 32_768,
    }])
    expect(managedStops).toBe(1)
  })

  it("deletes an arbitrary Magnitude-owned stored choice without requiring a catalog entry", async () => {
    const deleted: string[] = []
    const owned = artifact("user-owned-artifact", {
      _tag: "MagnitudeOwned",
      manifestId: "user-owned-artifact",
    })
    const modelStore: LlamaCppModelStoreApi = {
      inspect: Effect.succeed({ artifacts: [owned], warnings: [] }),
      resolve: () => Effect.die("deletion must not resolve model contents"),
      download: () => Stream.empty,
      deleteOwned: (modelId) => Effect.sync(() => { deleted.push(modelId) }),
    }
    const runtime: LlamaCppRuntimeApi = {
      inspect: Effect.succeed({ managed: null, external: [] }),
      ensureServing: () => Effect.die("deletion must not start a runtime"),
      stopManaged: Effect.void,
    }
    const layer = localInferenceLayer(modelStore, runtime, makeConfiguration({}))
    const selectionId = await Effect.runPromise(
      Effect.flatMap(LocalInference, (service) => service.state).pipe(
        Effect.flatMap((state) => {
          const choice = state.choices[0]
          return choice?.compatible
            ? Effect.succeed(choice.choiceId)
            : Effect.dieMessage("missing compatible stored choice")
        }),
        Effect.provide(layer),
      ),
    )

    await Effect.runPromise(
      Effect.flatMap(LocalInference, (service) => service.deleteModel(selectionId)).pipe(Effect.provide(layer)),
    )

    expect(deleted).toEqual(["user-owned-artifact"])
  })

  it("treats activation of the durable active selection as idempotent", async () => {
    const binding: DurableLocalModelBinding = {
      _tag: "External",
      selectionId: "active-selection",
      endpointConfigId: "external",
      providerModelId: "served-model",
      contextTokens: 8192,
    }
    const modelStore: LlamaCppModelStoreApi = {
      inspect: Effect.succeed({ artifacts: [], warnings: [] }),
      resolve: () => Effect.die("idempotent activation must not resolve a model"),
      download: () => Stream.empty,
      deleteOwned: () => Effect.void,
    }
    const runtime: LlamaCppRuntimeApi = {
      inspect: Effect.die("idempotent activation must not probe the runtime"),
      ensureServing: () => Effect.die("idempotent activation must not ensure a runtime"),
      stopManaged: Effect.void,
    }
    const layer = localInferenceLayer(modelStore, runtime, makeConfiguration({ binding }))

    await Effect.runPromise(
      Effect.flatMap(LocalInference, (service) => service.activateModel(binding.selectionId)).pipe(Effect.provide(layer)),
    )
  })

  it("restarts an arbitrary managed artifact from its durable serving parameters", async () => {
    const ensured: unknown[] = []
    let stops = 0
    const bindings: DurableLocalModelBinding[] = []
    const stored = artifact("discovered-cache-artifact", {
      _tag: "HuggingFaceCache",
      repo: "example/model",
      revision: "revision",
    })
    const modelStore: LlamaCppModelStoreApi = {
      inspect: Effect.succeed({ artifacts: [stored], warnings: [] }),
      resolve: () => Effect.succeed({
        ...stored,
        primaryPath: "/models/model.gguf",
        shardPaths: ["/models/model.gguf"],
        projectorPath: null,
      }),
      download: () => Stream.empty,
      deleteOwned: () => Effect.void,
    }
    const runtime: LlamaCppRuntimeApi = {
      inspect: Effect.succeed({ managed: null, external: [] }),
      ensureServing: (request) => Effect.sync(() => {
        ensured.push(request)
        return {
          serverId: "managed-test",
          ownership: "managed" as const,
          providerModelId: request.providerModelId,
          configuredContextTokens: request.contextTokens,
          metadata: { architecture: null, quantization: null, sizeBytes: null },
          connection: { baseUrl: "http://127.0.0.1:8080", apiKey: Option.none() },
        }
      }),
      stopManaged: Effect.sync(() => { stops += 1 }),
    }
    const configuration = makeConfiguration({
      usage: { localModelRole: "subagent", sessionConcurrency: "one" },
    }, (binding) => bindings.push(binding))
    const layer = localInferenceLayer(modelStore, runtime, configuration)
    const selectionId = await Effect.runPromise(
      Effect.flatMap(LocalInference, (service) => service.state).pipe(
        Effect.flatMap((state) => {
          const choice = state.choices[0]
          return choice ? Effect.succeed(choice.choiceId) : Effect.dieMessage("missing stored choice")
        }),
        Effect.provide(layer),
      ),
    )
    await Effect.runPromise(
      Effect.flatMap(LocalInference, (service) => service.activateModel(selectionId)).pipe(Effect.provide(layer)),
    )
    await Effect.runPromise(
      Effect.flatMap(LocalInference, (service) => service.restart).pipe(Effect.provide(layer)),
    )

    expect(bindings).toHaveLength(1)
    expect(bindings[0]).toMatchObject({
      _tag: "Managed",
      selectionId,
      artifactId: "discovered-cache-artifact",
      contextTokens: 64_000,
      parallelSlots: 3,
    })
    expect(stops).toBe(1)
    expect(ensured).toHaveLength(2)
    expect(ensured[1]).toEqual(ensured[0])
  })

  it("clears durable configuration before stopping the managed runtime", async () => {
    const events: string[] = []
    const binding: DurableLocalModelBinding = {
      _tag: "Managed",
      selectionId: "selection",
      artifactId: "artifact",
      providerModelId: "local:model",
      contextTokens: 8192,
      parallelSlots: 1,
    }
    const configuration: LocalModelConfigurationApi = {
      get: Effect.succeed({
        usage: { localModelRole: "main", sessionConcurrency: "one" },
        binding,
      }),
      updateUsage: () => Effect.void,
      updateSlots: () => Effect.void,
      activateLocal: () => Effect.void,
      disableLocal: Effect.sync(() => { events.push("configuration") }),
      changes: Stream.empty,
    }
    const runtime: LlamaCppRuntimeApi = {
      inspect: Effect.succeed({ managed: null, external: [] }),
      ensureServing: () => Effect.die("disable must not ensure serving"),
      stopManaged: Effect.sync(() => { events.push("runtime") }),
    }
    const modelStore: LlamaCppModelStoreApi = {
      inspect: Effect.succeed({ artifacts: [], warnings: [] }),
      resolve: () => Effect.die("disable must not resolve a model"),
      download: () => Stream.empty,
      deleteOwned: () => Effect.void,
    }
    const layer = localInferenceLayer(modelStore, runtime, configuration)

    await Effect.runPromise(
      Effect.flatMap(LocalInference, (service) => service.disable).pipe(Effect.provide(layer)),
    )

    expect(events).toEqual(["configuration", "runtime"])
  })

  it("serializes configuration and runtime lifecycle operations", async () => {
    const events: string[] = []
    const started = await Effect.runPromise(Deferred.make<void>())
    const release = await Effect.runPromise(Deferred.make<void>())
    const configuration: LocalModelConfigurationApi = {
      get: Effect.succeed({}),
      updateUsage: () => Effect.sync(() => { events.push("usage:start") }).pipe(
        Effect.zipRight(Deferred.succeed(started, undefined)),
        Effect.zipRight(Deferred.await(release)),
        Effect.zipRight(Effect.sync(() => { events.push("usage:end") })),
      ),
      updateSlots: () => Effect.void,
      activateLocal: () => Effect.void,
      disableLocal: Effect.sync(() => { events.push("disable") }),
      changes: Stream.empty,
    }
    const runtime: LlamaCppRuntimeApi = {
      inspect: Effect.succeed({ managed: null, external: [] }),
      ensureServing: () => Effect.die("configuration operations must not start a runtime"),
      stopManaged: Effect.sync(() => { events.push("stop") }),
    }
    const modelStore: LlamaCppModelStoreApi = {
      inspect: Effect.succeed({ artifacts: [], warnings: [] }),
      resolve: () => Effect.die("configuration operations must not resolve models"),
      download: () => Stream.empty,
      deleteOwned: () => Effect.void,
    }

    await Effect.runPromise(Effect.gen(function* () {
      const service = yield* LocalInference
      const configuring = yield* service.configureUsage({ localModelRole: "main", sessionConcurrency: "one" }).pipe(Effect.fork)
      yield* Deferred.await(started)
      const disabling = yield* service.disable.pipe(Effect.fork)
      yield* Effect.sleep("20 millis")
      expect(events).toEqual(["usage:start"])
      yield* Deferred.succeed(release, undefined)
      yield* Fiber.join(configuring)
      yield* Fiber.join(disabling)
    }).pipe(Effect.provide(localInferenceLayer(modelStore, runtime, configuration))))

    expect(events).toEqual(["usage:start", "usage:end", "disable", "stop"])
  })
})
