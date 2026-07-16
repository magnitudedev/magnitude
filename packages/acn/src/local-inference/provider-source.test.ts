import { describe, expect, it } from "vitest"
import { FetchHttpClient } from "@effect/platform"
import { Effect, Layer, Option, Stream } from "effect"
import { LlamaCpp } from "@magnitudedev/local-inference"
import { LocalInferencePlatform, type LocalInferencePlatformApi } from "./platform"
import { LocalModelConfiguration, type LocalModelConfigurationApi } from "./model-configuration"
import { LocalModelProviderSource, LocalModelProviderSourceLive, resolveLlamaModelInformation, type LlamaLogicalRoute } from "./provider-source"

describe("LocalModelProviderSource", () => {
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
      _tag: "Managed", modelPath: path, loaded: false, availability: { _tag: "Available" }, productRank: 0,
      record: { id: "file" as never, sourceId: "source" as never, displayName: "gemma.gguf", format: "gguf" as never, sizeBytes: 1, files: [], metadata, ownership: "magnitude", operations: { delete: true }, warnings: [] },
      request: { modelFileId: "file" as never, servedModelId: LlamaCpp.LlamaServedModelId.make("lmp_test"), profile: {} as never },
    }, {
      _tag: "External", modelPath: path, healthy: true, priority: 0,
      request: { instanceId: LlamaCpp.LlamaInstanceId.make("external"), servedModelId: LlamaCpp.LlamaServedModelId.make(`lmp_${"a".repeat(64)}`) },
      observation: {
        id: LlamaCpp.LlamaServedModelId.make(`lmp_${"a".repeat(64)}`), status: "loaded", reportedModelPath: Option.some(path),
        serverDisplayName: Option.none(), activeContextTokens: Option.some(131_072), architecture: Option.none(), serverFileType: Option.none(),
        serverReportedSizeBytes: Option.none(), inputModalities: Option.none(), outputModalities: Option.none(), loadProgress: Option.none(), failure: Option.none(),
      },
    }] as const satisfies readonly LlamaLogicalRoute[]
    const resolved = resolveLlamaModelInformation("lmp_test", path, routes)
    expect(resolved.information.displayName).toBe("Gemma-4-E4B-It")
    expect(resolved.information.displayNameSource).toBe("gguf_metadata")
    expect(resolved.evidence.map((item) => item._tag)).toEqual(["IndexedArtifact", "ServerReported"])
  })

  it("projects an already-loaded external model without requiring a managed binary", async () => {
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
    const snapshot = { instances: [observation], failures: [], capturedAt: new Date() }
    const registry: LlamaCpp.LlamaInstanceRegistryApi = {
      inspect: Effect.succeed(snapshot),
      refreshExternal: Effect.succeed(snapshot),
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
    }
    const missingBinary = new LlamaCpp.LlamaDistributionError({
      operation: "resolve",
      reason: "not-found",
      variant: Option.none(),
      status: Option.none(),
      path: Option.none(),
    })
    const platform = {
      files: {
        inspect: () => Effect.succeed({ records: [], issues: [], capturedAt: new Date() }),
        get: () => Effect.dieMessage("unused"),
        resolve: () => Effect.dieMessage("unused"),
        remove: () => Effect.dieMessage("unused"),
        index: Effect.succeed({ capturedAt: new Date(), sets: [], issues: [] }),
      },
      hardware: { inspect: Effect.dieMessage("unused") },
      distribution: { status: Effect.dieMessage("unused"), resolve: Effect.fail(missingBinary), install: () => Effect.dieMessage("unused") },
      hub: { searchModels: () => Stream.empty, resolveArtifact: () => Effect.dieMessage("unused") },
      downloads: { download: () => Stream.empty },
      cli: Effect.fail(missingBinary),
      instances: Effect.succeed(registry),
    } as LocalInferencePlatformApi
    const configuration = {
      get: Effect.succeed({}), getModels: Effect.succeed({}), updateUsage: () => Effect.void, updateSlots: () => Effect.void,
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
