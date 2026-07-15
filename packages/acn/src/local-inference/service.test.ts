import { describe, expect, test } from "vitest"
import { Effect, Layer, Stream } from "effect"
import type { LocalInferenceCapabilities, LocalModelChoice } from "@magnitudedev/protocol"
import {
  MagnitudeStorage,
  type MagnitudeStorageShape,
  type LocalInferenceConfig,
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
    localInference: LocalInferenceConfig | null
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
  activatedParallelSlots?: number,
): Harness => {
  const downloadedSources: LlamaCppHuggingFaceSource[] = []
  const activatedSelections: unknown[] = []
  const state: Harness["state"] = { modelConfig: null, onboarding: null, localInference: null }

  const storage = {
    auth: {
      get: () => Effect.succeed(hostedConfigured ? { type: "api" as const, key: "configured-key" } : undefined),
    },
    config: {
      getModelConfig: () => Effect.succeed(state.modelConfig),
      getOnboardingConfig: () => Effect.succeed(state.onboarding),
      getLocalInferenceConfig: () => Effect.succeed(state.localInference),
      setLocalInferenceConfig: (config: LocalInferenceConfig) => Effect.sync(() => {
        state.localInference = config
      }),
      completeCliModelSetupOnboarding: (completedAt: string) => Effect.sync(() => {
        state.onboarding = { completedAt }
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
        ...(activatedParallelSlots !== undefined ? { parallelSlots: activatedParallelSlots } : {}),
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
  test("waits for usage answers before generating deterministic recommendations", async () => {
    const harness = makeHarness()
    const snapshot = await run(
      harness,
      Effect.flatMap(LocalInferenceOnboarding, (service) => service.getSnapshot),
    )
    expect(snapshot.onboarding.required).toBe(true)
    expect(snapshot.capabilities?.system.totalMemoryBytes).toBe(32 * GIB)
    expect(snapshot.usage.selection).toBeUndefined()
    expect(snapshot.recommendations).toEqual([])

    const configured = await run(
      harness,
      Effect.flatMap(LocalInferenceOnboarding, (service) => service.configureUsage({
        localModelRole: "main",
        sessionConcurrency: "one",
      })),
    )
    expect(configured.usage.selection).toEqual({
      localModelRole: "main",
      sessionConcurrency: "one",
    })
    expect(configured.recommendations[0]).toMatchObject({
      displayName: "Qwen3.6 27B",
      contextTokens: 200_000,
      quantization: { format: "UD-Q4_K_XL" },
      servingProfile: { parallelSlots: 1, totalContextCapacityTokens: 200_000 },
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
      const snapshot = yield* service.configureUsage({
        localModelRole: "main",
        sessionConcurrency: "one",
      })
      yield* service.startDownload(snapshot.recommendations[0]!.configurationId)
    }))
    expect(harness.downloadedSources).toHaveLength(1)
    expect(harness.downloadedSources[0]).toMatchObject({
      repo: "unsloth/Qwen3.6-27B-GGUF",
      revision: "82d411acf4a06cfb8d9b073a5211bf410bfc29bf",
      quantTag: "UD-Q4_K_XL",
      contextTokens: 200_000,
      servingProfile: {
        localModelRole: "main",
        sessionConcurrency: "one",
        parallelSlots: 1,
        contextTokensPerSlot: 200_000,
        totalContextCapacityTokens: 200_000,
      },
      expectedFiles: [{
        path: "Qwen3.6-27B-UD-Q4_K_XL.gguf",
        sha256: "ff6941ded525b34eb159496762c29dd0ec6e71dc31b74d57e75d871a03eec259",
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

  test("keeps download progress as authoritative daemon state", async () => {
    const harness = makeHarness()
    const progress = await run(harness, Effect.gen(function* () {
      const service = yield* LocalInferenceOnboarding
      const snapshot = yield* service.configureUsage({
        localModelRole: "main",
        sessionConcurrency: "one",
      })
      const started = yield* service.startDownload(snapshot.recommendations[0]!.configurationId)
      const queued = yield* service.getDownloadProgress(started.operationId)
      yield* service.cancelDownload(started.operationId)
      const cancelled = yield* service.getDownloadProgress(started.operationId)
      return { queued, cancelled }
    }))

    expect(progress.queued).toMatchObject({
      operationId: "download-1",
      status: "queued",
      completedBytes: 0,
      selectionId: expect.any(String),
    })
    expect(progress.queued?.totalBytes).toBeGreaterThan(0)
    expect(progress.cancelled?.status).toBe("cancelled")
  })

  test("activation configures both slots without completing the combined walkthrough", async () => {
    const harness = makeHarness()
    await run(harness, Effect.gen(function* () {
      const service = yield* LocalInferenceOnboarding
      const snapshot = yield* service.configureUsage({
        localModelRole: "main",
        sessionConcurrency: "one",
      })
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
    expect(harness.state.onboarding?.completedAt).toBeDefined()

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
      const snapshot = yield* service.configureUsage({
        localModelRole: "main",
        sessionConcurrency: "one",
      })
      return yield* service.activate(snapshot.recommendations[0]!.configurationId)
    }))).rejects.toThrow("silent context reduction")
    expect(harness.state.modelConfig).toBeNull()
    expect(harness.state.onboarding).toBeNull()
  })

  test("filters discovered models against the selected context and parallel-slot requirements", async () => {
    const discovered: LocalModelChoice = {
      choiceId: "running:test",
      source: "running",
      displayName: "Test model",
      providerModelId: "test-model",
      contextTokens: 64_000,
      parallelSlots: 3,
      fitClass: "unknown",
      managed: false,
      compatible: true,
      explanation: "test",
    }
    const harness = makeHarness([discovered])

    const subagentsOneSession = await run(
      harness,
      Effect.flatMap(LocalInferenceOnboarding, (service) => service.configureUsage({
        localModelRole: "subagent",
        sessionConcurrency: "one",
      })),
    )
    expect(subagentsOneSession.downloaded[0]?.compatible).toBe(true)

    const subagentsMultipleSessions = await run(
      harness,
      Effect.flatMap(LocalInferenceOnboarding, (service) => service.configureUsage({
        localModelRole: "subagent",
        sessionConcurrency: "up_to_three",
      })),
    )
    expect(subagentsMultipleSessions.downloaded[0]?.compatible).toBe(false)

    const mainOneSession = await run(
      harness,
      Effect.flatMap(LocalInferenceOnboarding, (service) => service.configureUsage({
        localModelRole: "main",
        sessionConcurrency: "one",
      })),
    )
    expect(mainOneSession.downloaded[0]?.compatible).toBe(false)
  })

  test("rejects a silent parallel-slot reduction before changing app configuration", async () => {
    const harness = makeHarness([], undefined, false, 1)
    await expect(run(harness, Effect.gen(function* () {
      const service = yield* LocalInferenceOnboarding
      const snapshot = yield* service.configureUsage({
        localModelRole: "main",
        sessionConcurrency: "up_to_three",
      })
      return yield* service.activate(snapshot.recommendations[0]!.configurationId)
    }))).rejects.toThrow("silent slot reduction")
    expect(harness.state.modelConfig).toBeNull()
    expect(harness.state.onboarding).toBeNull()
  })
})
