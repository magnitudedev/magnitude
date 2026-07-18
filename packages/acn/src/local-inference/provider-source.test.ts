import { describe, expect, it } from "vitest"
import { FetchHttpClient } from "@effect/platform"
import { Effect, Layer, Option, Stream } from "effect"
import { LlamaCpp } from "@magnitudedev/local-inference"
import { ProviderModelIdSchema, ReasoningEffortSchema } from "@magnitudedev/sdk"
import { LocalInferencePlatform, type LocalInferencePlatformApi } from "./platform"
import { LocalModelConfiguration, type LocalModelConfigurationApi } from "./model-configuration"
import { LocalModelProviderSource, LocalModelProviderSourceLive, assessLlamaFitStableCapacity, llamaCppRequestOptionsForReasoningMapping, llamaFitEstimateFitsStableCapacity, resolveLlamaModelInformation, type LlamaLogicalRoute } from "./provider-source"

const GIBIBYTE = 1024 ** 3

describe("LocalModelProviderSource", () => {
  it("serializes a reasoning mapping into separate template and budget options", () => {
    const requestOptions = llamaCppRequestOptionsForReasoningMapping({
      reasoningEffort: ReasoningEffortSchema.make("high"),
      templateOptions: {
        enableThinking: Option.some(true),
        reasoningEffort: Option.some("high"),
      },
      thinkingBudget: { _tag: "Enabled", tokens: 4_096 },
    })

    expect(requestOptions).toEqual({
      chatTemplateKwargs: Option.some({ enable_thinking: true, reasoning_effort: "high" }),
      thinkingBudgetTokens: Option.some(4_096),
    })
  })

  it("omits empty template options and a disabled thinking budget", () => {
    expect(llamaCppRequestOptionsForReasoningMapping({
      reasoningEffort: ReasoningEffortSchema.make("none"),
      templateOptions: {
        enableThinking: Option.none(),
        reasoningEffort: Option.none(),
      },
      thinkingBudget: { _tag: "Disabled" },
    })).toEqual({
      chatTemplateKwargs: Option.none(),
      thinkingBudgetTokens: Option.none(),
    })
  })

  it("compares Apple Silicon placements as one stable unified-memory domain", () => {
    const plan = LlamaCpp.makeLlamaFitPlan({
      fitExecutableFingerprint: LlamaCpp.LlamaCppExecutableFingerprint.make("fit"),
      profileId: LlamaCpp.LlamaExecutionProfileId.make("profile"),
      fileVersion: [],
      arguments: [],
      placement: [
        { device: LlamaCpp.LlamaDeviceId.make("MTL0"), modelBytes: 32 * GIBIBYTE, contextBytes: 0, computeBytes: 0 },
        { device: LlamaCpp.LlamaDeviceId.make("Host"), modelBytes: 20 * GIBIBYTE, contextBytes: 0, computeBytes: 0 },
      ],
      memory: { baseBytes: 52 * GIBIBYTE, vision: Option.none(), estimatedTotalBytes: 52 * GIBIBYTE },
      rawOutput: "",
    })
    const host = {
      capturedAt: new Date(), platform: "darwin" as const, processArchitecture: "arm64", nativeArchitecture: "arm64",
      cpuModel: Option.none<string>(), logicalCores: 1, totalMemoryBytes: 64 * GIBIBYTE, availableMemoryBytes: GIBIBYTE,
    }
    const devices: readonly LlamaCpp.LlamaDevice[] = [{
      id: LlamaCpp.LlamaDeviceId.make("MTL0"), name: Option.none(), backend: Option.some("Metal"),
      type: Option.some("IGPU"), physicalId: Option.none(),
      totalMemoryBytes: Option.some(48 * GIBIBYTE), freeMemoryBytes: Option.some(GIBIBYTE),
    }]
    expect(llamaFitEstimateFitsStableCapacity(plan, host, devices)).toBe(false)
    const assessment = assessLlamaFitStableCapacity(plan, host, devices)
    expect(Option.isSome(assessment)).toBe(true)
    if (Option.isNone(assessment)) return
    expect(assessment.value.result).toBe("capacity_risk")
    expect(assessment.value.estimatedTotalBytes).toBe(52 * GIBIBYTE)
    expect(assessment.value.domains[0]).toMatchObject({
      memoryDomainId: "system",
      estimatedBytes: 52 * GIBIBYTE,
      stableCapacityBytes: 51.2 * GIBIBYTE,
    })
    expect(assessment.value.domains[0]!.marginBytes).toBeCloseTo(-0.8 * GIBIBYTE)
    expect(llamaFitEstimateFitsStableCapacity({
      ...plan,
      memory: { ...plan.memory, estimatedTotalBytes: 50 * GIBIBYTE },
    }, host, devices)).toBe(true)
  })

  it("charges vision adjustment to the primary discrete accelerator", () => {
    const baseBytes = 12 * GIBIBYTE
    const vision = Option.some({ projectorFileBytes: GIBIBYTE, estimatedProjectorBytes: 1.2 * GIBIBYTE, uncertaintyBytes: 1.5 * GIBIBYTE })
    const plan = LlamaCpp.makeLlamaFitPlan({
      fitExecutableFingerprint: LlamaCpp.LlamaCppExecutableFingerprint.make("fit"),
      profileId: LlamaCpp.LlamaExecutionProfileId.make("profile"),
      fileVersion: [],
      arguments: [],
      placement: [{ device: LlamaCpp.LlamaDeviceId.make("CUDA0"), modelBytes: baseBytes, contextBytes: 0, computeBytes: 0 }],
      memory: { baseBytes, vision, estimatedTotalBytes: baseBytes + 2.7 * GIBIBYTE },
      rawOutput: "",
    })
    const host = {
      capturedAt: new Date(), platform: "linux" as const, processArchitecture: "x64", nativeArchitecture: "x64",
      cpuModel: Option.none<string>(), logicalCores: 1, totalMemoryBytes: 64 * GIBIBYTE, availableMemoryBytes: GIBIBYTE,
    }
    const devices: readonly LlamaCpp.LlamaDevice[] = [{
      id: LlamaCpp.LlamaDeviceId.make("CUDA0"), name: Option.none(), backend: Option.some("CUDA"),
      type: Option.some("GPU"), physicalId: Option.some("0000:01:00.0"),
      totalMemoryBytes: Option.some(16 * GIBIBYTE), freeMemoryBytes: Option.some(GIBIBYTE),
    }]
    expect(llamaFitEstimateFitsStableCapacity(plan, host, devices)).toBe(false)
    expect(llamaFitEstimateFitsStableCapacity({
      ...plan,
      memory: { baseBytes, vision: Option.none(), estimatedTotalBytes: baseBytes },
    }, host, devices)).toBe(true)
  })

  it("uses exactly associated indexed GGUF metadata over an opaque external alias", () => {
    const path = LlamaCpp.NormalizedLlamaModelPath.make("/models/gemma.gguf")
    const metadata = {
      name: Option.some("Gemma-4-E4B-It"), architecture: Option.some("gemma3"), ggufFileType: Option.none(), quantization: Option.none(),
      trainedContextTokens: Option.some(131_072), parameterCount: Option.none(), embeddingLength: Option.none(), blockCount: Option.none(),
      attentionHeadCount: Option.none(), vocabularySize: Option.none(), feedForwardLength: Option.none(), expertCount: Option.none(),
      expertUsedCount: Option.none(), tokenizerModel: Option.none(), tokenizerPre: Option.none(), chatTemplate: Option.none(),
      baseModelNames: [], baseModelRepositories: [], inputModalities: Option.none(), outputModalities: Option.none(),
    }
    const routes = [{
      _tag: "Managed", modelPath: path, loaded: false, availability: { _tag: "Available" }, fitAssessment: Option.none(), productRank: 0,
      record: { id: "file" as never, sourceId: "source" as never, displayName: "gemma.gguf", format: "gguf" as never, sizeBytes: 1, files: [], metadata, ownership: "magnitude", operations: { delete: true }, warnings: [] },
      request: { modelFileId: "file" as never, servedModelId: LlamaCpp.LlamaServedModelId.make("lmp_test"), profile: {} as never },
    }, {
      _tag: "External", modelPath: path, healthy: true, priority: 0, mode: "router",
      request: { instanceId: LlamaCpp.LlamaInstanceId.make("external"), servedModelId: LlamaCpp.LlamaServedModelId.make(`lmp_${"a".repeat(64)}`) },
      observation: {
        id: LlamaCpp.LlamaServedModelId.make(`lmp_${"a".repeat(64)}`), status: "loaded", reportedModelPath: Option.some(path),
        serverDisplayName: Option.none(), activeContextTokens: Option.some(131_072), architecture: Option.none(), serverFileType: Option.none(),
        serverReportedSizeBytes: Option.none(), inputModalities: Option.none(), outputModalities: Option.none(), loadProgress: Option.none(), failure: Option.none(),
      },
    }] as const satisfies readonly LlamaLogicalRoute[]
    const resolved = resolveLlamaModelInformation(ProviderModelIdSchema.make("lmp_test"), path, routes)
    expect(resolved.displayName).toBe("Gemma-4-E4B-It")
    expect(resolved.displayNameSource).toBe("gguf_metadata")
    expect(resolved.metadata.name).toEqual(Option.some({ value: "Gemma-4-E4B-It", source: "indexed_artifact" }))
  })

  it("projects an already-loaded external model without requiring a managed installation", async () => {
    const instanceId = LlamaCpp.LlamaInstanceId.make("external_test")
    const servedModelId = LlamaCpp.LlamaServedModelId.make("user-model")
    const observation: LlamaCpp.LlamaInstanceObservation = {
      id: instanceId,
      ownership: "external",
      health: "ready",
      mode: "router",
      build: Option.some("b10011"),
      capabilities: { models: "supported", modelEvents: "supported", load: "supported", unload: "supported", sleep: "supported" },
      models: [{
        id: servedModelId,
        status: "loaded",
        serverDisplayName: Option.some("User model"),
        reportedModelPath: Option.some(LlamaCpp.NormalizedLlamaModelPath.make("/models/user.gguf")),
        activeContextTokens: Option.some(65_536),
        architecture: Option.none(),
        serverFileType: Option.none(),
        serverReportedSizeBytes: Option.none(),
        inputModalities: Option.some(["text"]),
        outputModalities: Option.some(["text"]),
        loadProgress: Option.none(),
        failure: Option.none(),
      }],
      diagnostics: [],
    }
    const snapshot = { instances: [observation], failures: [], capturedAt: new Date(), activeManagedInstallationId: Option.none<LlamaCpp.LlamaCppInstallationId>() }
    const registry: LlamaCpp.LlamaInstanceRegistryApi = {
      snapshot: Effect.succeed(snapshot),
      changes: Stream.empty,
      refresh: Effect.succeed(snapshot),
      get: () => Effect.dieMessage("unused"),
      ensureManagedLoaded: () => Effect.dieMessage("managed loading must not run"),
      acquireLoadedManaged: () => Effect.dieMessage("managed loading must not run"),
      acquire: (request) => request._tag === "External"
        ? Effect.succeed({
            instanceId,
            model: observation.models[0]!,
            target: {
              origin: new URL("http://127.0.0.1:8080"),
              authorization: Option.none(),
              model: servedModelId,
            },
          })
        : Effect.dieMessage("managed acquisition must not run"),
      stopManaged: Effect.void,
      reconcileManagedInstallation: Effect.void,
    }
    const missingInstallation = new LlamaCpp.LlamaCppInstallationUnavailable({ reason: "missing" })
    const platform = {
      modelIndex: {
        snapshot: Effect.dieMessage("unused"),
        artifacts: Effect.dieMessage("unused"),
        replaceArtifacts: () => Effect.void,
        discoveredProperties: () => Effect.succeed(Option.none()),
        putDiscoveredProperties: () => Effect.void,
        fitAssessment: () => Effect.succeed(Option.none()),
        putFitAssessment: () => Effect.void,
      },
      files: {
        inspect: () => Effect.succeed({ records: [], issues: [], capturedAt: new Date() }),
        get: () => Effect.dieMessage("unused"),
        resolve: () => Effect.dieMessage("unused"),
        remove: () => Effect.dieMessage("unused"),
        artifactIndex: Effect.succeed({ capturedAt: new Date(), sets: [], issues: [] }),
        changes: Stream.empty,
      },
      hardware: { inspect: Effect.dieMessage("unused") },
      installations: {
        snapshot: Effect.dieMessage("unused"),
        changes: Stream.empty,
        refresh: Effect.void,
        selected: Effect.fail(missingInstallation),
        installManaged: Effect.dieMessage("unused"),
      },
      hub: { searchModels: () => Stream.empty, resolveArtifact: () => Effect.dieMessage("unused") },
      downloads: { download: () => Stream.empty },
      cli: Effect.fail(missingInstallation),
      instances: Effect.succeed(registry),
      instanceChanges: Stream.empty,
      serverClient: () => Effect.dieMessage("unused"),
    } as LocalInferencePlatformApi
    const configuration = {
      get: Effect.succeed({}), getModels: Effect.succeed({}), updateUsage: () => Effect.void, selectProfile: () => Effect.void, updateSlots: () => Effect.void,
      reconcileSlots: () => Effect.succeed(false), recordUse: () => Effect.void,
      activateLocal: () => Effect.void, disableLocal: Effect.void, changes: Stream.empty,
    } as LocalModelConfigurationApi
    const layer = LocalModelProviderSourceLive.pipe(Layer.provide(Layer.mergeAll(
      Layer.succeed(LocalInferencePlatform, platform),
      Layer.succeed(LocalModelConfiguration, configuration),
    )))
    const result = await Effect.runPromise(Effect.scoped(Effect.flatMap(
      LocalModelProviderSource,
      (source) => Effect.gen(function* () {
        const models = yield* source.catalog.refresh
        const information = yield* source.getModelInformation(models[0]!.providerModelId)
        return { models, information }
      }),
    )).pipe(Effect.provide(Layer.merge(layer, FetchHttpClient.layer))))
    const { models, information } = result
    expect(models).toHaveLength(1)
    expect(models[0]).toMatchObject({ displayName: "User model", availability: { _tag: "Available" } })
    expect(information.displayName).toBe("User model")
    expect(models[0]).not.toHaveProperty("ownership")
    expect(models[0]).not.toHaveProperty("residency")
  })
})
