import { describe, expect, test } from "vitest"
import { Effect, Layer, Stream } from "effect"
import type { LocalInferenceCapabilities, LocalModelChoice } from "@magnitudedev/protocol"
import {
  MagnitudeStorage,
  type MagnitudeStorageShape,
  type ModelConfig,
  type OnboardingConfig,
} from "@magnitudedev/storage"
import { Account, type AccountApi } from "../account"
import { GIB } from "./recommendations"
import { LlamaCppRuntimeBridge } from "./runtime-bridge"
import { LocalInferenceOnboarding, LocalInferenceOnboardingLive } from "./service"
import type {
  LlamaCppHuggingFaceSource,
  LlamaCppRuntimeBridgeShape,
} from "./types"

interface Harness {
  readonly layer: Layer.Layer<LocalInferenceOnboarding>
  readonly downloadedSources: LlamaCppHuggingFaceSource[]
  readonly activatedSelections: unknown[]
  readonly state: {
    modelConfig: ModelConfig | null
    onboarding: OnboardingConfig | null
  }
}

const capabilities: LocalInferenceCapabilities = {
  binary: { identity: "managed-test-llama-server", version: "test" },
  system: { totalMemoryBytes: 32 * GIB, cpuModel: "test cpu", logicalCores: 8 },
  accelerators: [],
  warnings: [],
}

const makeHarness = (
  inventory: readonly LocalModelChoice[] = [],
  activatedContextTokens?: number,
  hostedConfigured = false,
): Harness => {
  const downloadedSources: LlamaCppHuggingFaceSource[] = []
  const activatedSelections: unknown[] = []
  const state: Harness["state"] = { modelConfig: null, onboarding: null }

  const storage = {
    auth: {
      get: () => Effect.succeed(hostedConfigured ? { type: "api" as const, key: "configured-key" } : undefined),
    },
    config: {
      getModelConfig: () => Effect.succeed(state.modelConfig),
      getOnboardingConfig: () => Effect.succeed(state.onboarding),
      completeCliModelSetupOnboarding: (version: number, completedAt: string) => Effect.sync(() => {
        state.onboarding = { cliModelSetupVersion: version, completedAt }
      }),
    },
  } as unknown as MagnitudeStorageShape

  const account = {
    getCachedModelList: () => Effect.succeed({
      slotProfiles: hostedConfigured ? { primary: {}, secondary: {} } : {},
    }),
    updateModelConfig: (slots: ModelConfig["slots"]) => Effect.sync(() => {
      state.modelConfig = { slots }
    }),
  } as unknown as AccountApi

  const bridge: LlamaCppRuntimeBridgeShape = {
    getReadiness: Effect.succeed({ status: "ready", canDownload: true, canActivate: true }),
    getCapabilities: Effect.succeed(capabilities),
    getInventory: Effect.succeed({ running: [], downloaded: inventory }),
    startDownload: (source) => Effect.sync(() => {
      downloadedSources.push(source)
      return { operationId: "download-1" }
    }),
    subscribeDownload: () => Stream.empty,
    cancelDownload: () => Effect.void,
    activate: (selection) => Effect.sync(() => {
      activatedSelections.push(selection)
      return {
        providerId: "llamacpp",
        providerModelId: "local-test-model",
        contextTokens: activatedContextTokens
          ?? ("contextTokens" in selection ? selection.contextTokens : 32_768),
      }
    }),
  }

  const dependencies = Layer.mergeAll(
    Layer.succeed(MagnitudeStorage, MagnitudeStorage.of(storage)),
    Layer.succeed(Account, Account.of(account)),
    Layer.succeed(LlamaCppRuntimeBridge, LlamaCppRuntimeBridge.of(bridge)),
  )

  return {
    layer: LocalInferenceOnboardingLive.pipe(Layer.provide(dependencies)),
    downloadedSources,
    activatedSelections,
    state,
  }
}

const run = <A, E>(harness: Harness, effect: Effect.Effect<A, E, LocalInferenceOnboarding>) =>
  Effect.runPromise(effect.pipe(Effect.provide(harness.layer)))

describe("LocalInferenceOnboarding service", () => {
  test("aggregates a new-user snapshot with deterministic recommendations", async () => {
    const harness = makeHarness()
    const snapshot = await run(
      harness,
      Effect.flatMap(LocalInferenceOnboarding, (service) => service.getSnapshot),
    )
    expect(snapshot.onboarding.required).toBe(true)
    expect(snapshot.capabilities?.system.totalMemoryBytes).toBe(32 * GIB)
    expect(snapshot.recommendations[0]).toMatchObject({
      displayName: "Qwen3.6 35B-A3B",
      contextTokens: 32_768,
      quantization: { format: "UD-Q4_K_XL" },
    })
  })

  test("requires the first-run walkthrough even when Cloud is already usable", async () => {
    const harness = makeHarness([], undefined, true)
    const snapshot = await run(
      harness,
      Effect.flatMap(LocalInferenceOnboarding, (service) => service.getSnapshot),
    )
    expect(snapshot.onboarding.required).toBe(true)
    expect(snapshot.configuration.usable).toBe(true)
  })

  test("resolves an opaque configuration to an exact pinned Hugging Face source", async () => {
    const harness = makeHarness()
    await run(harness, Effect.gen(function* () {
      const service = yield* LocalInferenceOnboarding
      const snapshot = yield* service.getSnapshot
      yield* service.startDownload(snapshot.recommendations[0]!.configurationId)
    }))
    expect(harness.downloadedSources).toHaveLength(1)
    expect(harness.downloadedSources[0]).toMatchObject({
      repo: "unsloth/Qwen3.6-35B-A3B-GGUF",
      revision: "a483e9e6cbd595906af30beda3187c2663a1118c",
      quantTag: "UD-Q4_K_XL",
      contextTokens: 32_768,
      expectedFiles: [{
        path: "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        sha256: "707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450",
      }],
    })
  })

  test("rejects client-supplied arbitrary artifact combinations", async () => {
    const harness = makeHarness()
    await expect(run(
      harness,
      Effect.flatMap(
        LocalInferenceOnboarding,
        (service) => service.startDownload("org/repo:Q4@ctx-999999"),
      ),
    )).rejects.toThrow("resolve local model configuration")
    expect(harness.downloadedSources).toHaveLength(0)
  })

  test("activation configures both slots without completing the combined walkthrough", async () => {
    const harness = makeHarness()
    await run(harness, Effect.gen(function* () {
      const service = yield* LocalInferenceOnboarding
      const snapshot = yield* service.getSnapshot
      yield* service.activate(snapshot.recommendations[0]!.configurationId)
    }))
    expect(harness.activatedSelections).toHaveLength(1)
    expect(harness.state.modelConfig?.slots).toEqual({
      primary: { providerId: "llamacpp", providerModelId: "local-test-model" },
      secondary: { providerId: "llamacpp", providerModelId: "local-test-model" },
    })
    expect(harness.state.onboarding).toBeNull()
  })

  test("completion is allowed after the user skips both optional providers", async () => {
    const harness = makeHarness()
    await run(harness, Effect.flatMap(
      LocalInferenceOnboarding,
      (service) => service.complete,
    ))
    expect(harness.state.modelConfig).toBeNull()
    expect(harness.state.onboarding?.cliModelSetupVersion).toBe(2)

    const snapshot = await run(
      harness,
      Effect.flatMap(LocalInferenceOnboarding, (service) => service.getSnapshot),
    )
    expect(snapshot.onboarding.required).toBe(false)
    expect(snapshot.configuration.usable).toBe(false)
  })

  test("rejects a silent context reduction before changing app configuration", async () => {
    const harness = makeHarness([], 4_096)
    await expect(run(harness, Effect.gen(function* () {
      const service = yield* LocalInferenceOnboarding
      const snapshot = yield* service.getSnapshot
      return yield* service.activate(snapshot.recommendations[0]!.configurationId)
    }))).rejects.toThrow("silent context reduction")
    expect(harness.state.modelConfig).toBeNull()
    expect(harness.state.onboarding).toBeNull()
  })
})
