import { describe, expect, it } from "vitest"
import { FetchHttpClient } from "@effect/platform"
import { Effect, Layer, Option, Stream } from "effect"
import { LlamaCpp } from "@magnitudedev/local-inference"
import { LocalInferencePlatform, type LocalInferencePlatformApi } from "./platform"
import { LocalModelConfiguration, type LocalModelConfigurationApi } from "./model-configuration"
import { LocalModelProviderSource, LocalModelProviderSourceLive } from "./provider-source"

describe("LocalModelProviderSource", () => {
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
        index: Effect.succeed({ schemaVersion: 1, capturedAt: new Date(), sets: [], issues: [] }),
      },
      hardware: { inspect: Effect.dieMessage("unused") },
      distribution: { status: Effect.dieMessage("unused"), resolve: Effect.fail(missingBinary), install: () => Effect.dieMessage("unused") },
      hub: { searchModels: () => Stream.empty, resolveArtifact: () => Effect.dieMessage("unused") },
      downloads: { download: () => Stream.empty },
      cli: Effect.fail(missingBinary),
      instances: Effect.succeed(registry),
    } as LocalInferencePlatformApi
    const configuration = {
      get: Effect.succeed({}), updateUsage: () => Effect.void, updateSlots: () => Effect.void,
      reconcileSlots: () => Effect.succeed(false), recordUse: () => Effect.void,
      activateLocal: () => Effect.void, disableLocal: Effect.void, changes: Stream.empty,
    } as LocalModelConfigurationApi
    const layer = LocalModelProviderSourceLive.pipe(Layer.provide(Layer.mergeAll(
      Layer.succeed(LocalInferencePlatform, platform),
      Layer.succeed(LocalModelConfiguration, configuration),
    )))
    const models = await Effect.runPromise(Effect.scoped(Effect.flatMap(
      LocalModelProviderSource,
      (source) => source.catalog.refresh,
    )).pipe(Effect.provide(layer), Effect.provide(FetchHttpClient.layer)))
    expect(models).toHaveLength(1)
    expect(models[0]).toMatchObject({ displayName: "User model", ownership: "external", residency: "loaded", availability: { _tag: "Available" } })
  })
})
