import { Cause, Context, Effect, Layer, Ref, Stream } from "effect"
import {
  SessionOperationFailed,
  type LocalInferenceOnboardingSnapshot,
  type LocalInferenceUsageSelection,
  type LocalModelDownloadProgress,
  type LocalModelChoice,
  type SessionError,
} from "@magnitudedev/protocol"
import { MagnitudeStorage, type MagnitudeStorageShape, type ModelConfig } from "@magnitudedev/storage"
import { Account } from "../account"
import { LlamaCppRuntimeBridge } from "./runtime-bridge"
import { recommendLocalModels, resolveConfiguration, stableCapacityFromCapabilities } from "./recommendations"
import type {
  EvaluatedLocalConfiguration,
  LlamaCppHuggingFaceSource,
  LlamaCppRuntimeBridgeShape,
} from "./types"

export interface LocalInferenceOnboardingApi {
  readonly getSnapshot: Effect.Effect<LocalInferenceOnboardingSnapshot, SessionError>
  readonly configureUsage: (
    usage: LocalInferenceUsageSelection,
  ) => Effect.Effect<LocalInferenceOnboardingSnapshot, SessionError>
  readonly startDownload: (configurationId: string) => Effect.Effect<{ readonly operationId: string }, SessionError>
  readonly getDownloadProgress: (
    operationId: string,
  ) => Effect.Effect<LocalModelDownloadProgress | null, SessionError>
  readonly subscribeDownload: (operationId: string) => Stream.Stream<LocalModelDownloadProgress, SessionError>
  readonly cancelDownload: (operationId: string) => Effect.Effect<void, SessionError>
  readonly activate: (selectionId: string) => Effect.Effect<{
    readonly providerId: string
    readonly providerModelId: string
    readonly contextTokens: number
  }, SessionError>
  readonly complete: Effect.Effect<void, SessionError>
}

export class LocalInferenceOnboarding extends Context.Tag("LocalInferenceOnboarding")<
  LocalInferenceOnboarding,
  LocalInferenceOnboardingApi
>() {}

const toLocalInferenceError = (operation: string) => (cause: unknown): SessionError =>
  new SessionOperationFailed({
    operation,
    reason: Cause.pretty(Cause.fail(cause)),
  })

const hasExplicitSlots = (config: ModelConfig | null): boolean =>
  Boolean(
    config?.slots?.primary?.providerId
    && config.slots.primary.providerModelId
    && config.slots.secondary?.providerId
    && config.slots.secondary.providerModelId,
  )

const makeSource = (configuration: EvaluatedLocalConfiguration): LlamaCppHuggingFaceSource => ({
  catalogModelId: configuration.entry.id,
  configurationId: configuration.configurationId,
  repo: configuration.entry.repo,
  revision: configuration.entry.revision,
  quantTag: configuration.entry.quantTag,
  contextTokens: configuration.contextTokens,
  servingProfile: configuration.servingProfile,
  expectedFiles: configuration.entry.files,
})

const getConfiguration = (
  bridge: LlamaCppRuntimeBridgeShape,
  storage: MagnitudeStorageShape,
  configurationId: string,
): Effect.Effect<EvaluatedLocalConfiguration, SessionError> =>
  Effect.gen(function* () {
    const usage = yield* storage.config.getLocalInferenceConfig().pipe(
      Effect.mapError(toLocalInferenceError("read local inference usage")),
    )
    if (!usage) {
      return yield* (new SessionOperationFailed({
        operation: "resolve local model configuration",
        reason: "Choose how the local model will be used before selecting a recommendation.",
      }))
    }
    const capabilities = yield* bridge.getCapabilities
    const configuration = resolveConfiguration(
      configurationId,
      stableCapacityFromCapabilities(capabilities),
      usage,
    )
    if (!configuration) {
      return yield* (new SessionOperationFailed({
        operation: "resolve local model configuration",
        reason: "The selection is unknown or no longer fits the machine's stable capacity. Refresh onboarding and choose a server-issued configuration.",
      }))
    }
    return configuration
  })

export const LocalInferenceOnboardingLive: Layer.Layer<
  LocalInferenceOnboarding,
  never,
  LlamaCppRuntimeBridge | MagnitudeStorage | Account
> = Layer.effect(
  LocalInferenceOnboarding,
  Effect.gen(function* () {
    const bridge = yield* LlamaCppRuntimeBridge
    const storage = yield* MagnitudeStorage
    const account = yield* Account
    const downloadProgress = yield* Ref.make(new Map<string, LocalModelDownloadProgress>())

    const storeDownloadProgress = (progress: LocalModelDownloadProgress) => Ref.update(
      downloadProgress,
      (current) => {
        const next = new Map(current)
        next.set(progress.operationId, progress)
        return next
      },
    )

    const hasUsableConfiguration = Effect.gen(function* () {
      const modelConfig = yield* storage.config.getModelConfig().pipe(
        Effect.mapError(toLocalInferenceError("read model configuration")),
      )
      if (hasExplicitSlots(modelConfig)) return true

      const storedMagnitudeAuth = yield* storage.auth.get("magnitude").pipe(
        Effect.catchAll(() => Effect.void),
      )
      const hasHostedCredential = (
        storedMagnitudeAuth?.type === "api" && storedMagnitudeAuth.key.trim().length > 0
      ) || Boolean(process.env.MAGNITUDE_API_KEY?.trim()) || Boolean(process.env.MAGNITUDE_LOCAL_API_KEY?.trim())
      if (!hasHostedCredential) return false

      // Existing hosted users may be using resolved defaults rather than
      // explicit slot overrides. Treat them as configured only when the live
      // provider catalog resolves both slots, not merely because an API key is
      // present.
      return yield* account.getCachedModelList().pipe(
        Effect.map((models) => Boolean(models.slotProfiles.primary && models.slotProfiles.secondary)),
        Effect.catchAll(() => Effect.succeed(false)),
      )
    })

    // Completing the first-run walkthrough is independent of provider
    // readiness: both Local Models and Cloud Fallback are explicitly skippable.
    // The CLI renders a separate no-provider state when neither is configured.
    const complete = storage.config.completeCliModelSetupOnboarding(
        new Date().toISOString(),
      ).pipe(Effect.mapError(toLocalInferenceError("persist CLI model setup onboarding")))

    const getSnapshot: Effect.Effect<LocalInferenceOnboardingSnapshot, SessionError> = Effect.gen(function* () {
      const [onboarding, usage, readiness, capabilities, inventory, usableConfiguration] = yield* Effect.all([
        storage.config.getOnboardingConfig().pipe(
          Effect.mapError(toLocalInferenceError("read local inference onboarding")),
        ),
        storage.config.getLocalInferenceConfig().pipe(
          Effect.mapError(toLocalInferenceError("read local inference usage")),
        ),
        bridge.getReadiness,
        bridge.getCapabilities,
        bridge.getInventory,
        hasUsableConfiguration,
      ], { concurrency: "unbounded" })

      const stableCapacity = stableCapacityFromCapabilities(capabilities)
      const recommendations = usage ? recommendLocalModels(stableCapacity, usage) : []
      const completed = onboarding?.completedAt !== undefined
      const warnings = [...capabilities.warnings]
      if (usage && recommendations.length === 0) {
        warnings.push({
          code: "no_stable_capacity_recommendation",
          message: `No curated model can reserve the requested ${usage.localModelRole === "main" ? "main-agent" : "subagent"} context windows after the stable operating-system reserve. Try fewer simultaneous sessions or use Cloud Fallback.`,
        })
      }

      const requiredParallelSlots = usage
        ? (usage.localModelRole === "main" ? 1 : 3) * (usage.sessionConcurrency === "one" ? 1 : 3)
        : undefined
      const minimumContextTokens = usage?.localModelRole === "main" ? 100_000 : 64_000
      const applyUsageCompatibility = (choice: LocalModelChoice): LocalModelChoice => {
        if (!usage || requiredParallelSlots === undefined) return choice
        const contextCompatible = choice.contextTokens >= minimumContextTokens
        const slotsCompatible = choice.parallelSlots === undefined || choice.parallelSlots >= requiredParallelSlots
        // TODO(llamacpp-serving-profile-integration, CTO-owned): Replace this
        // partial check with the managed bridge's normalized uniform slot and
        // total-context-capacity report. Unknown slot counts remain governed by
        // the bridge's compatibility result until that contract lands.
        return {
          ...choice,
          compatible: choice.compatible && contextCompatible && slotsCompatible,
        }
      }

      return {
        onboarding: {
          required: !completed,
          ...(onboarding?.completedAt ? { completedAt: onboarding.completedAt } : {}),
        },
        configuration: { usable: usableConfiguration },
        usage: {
          ...(usage ? { selection: usage } : {}),
        },
        runtime: readiness,
        capabilities,
        running: inventory.running.map(applyUsageCompatibility),
        downloaded: inventory.downloaded.map(applyUsageCompatibility),
        recommendations,
        warnings,
      }
    })

    return LocalInferenceOnboarding.of({
      getSnapshot,

      configureUsage: (usage) => storage.config.setLocalInferenceConfig(usage).pipe(
        Effect.mapError(toLocalInferenceError("persist local inference usage")),
        Effect.flatMap(() => getSnapshot),
      ),

      startDownload: (configurationId) => Effect.gen(function* () {
        const configuration = yield* getConfiguration(bridge, storage, configurationId)
        const started = yield* bridge.startDownload(makeSource(configuration))
        yield* storeDownloadProgress({
          operationId: started.operationId,
          status: "queued",
          completedBytes: 0,
          totalBytes: configuration.entry.files.reduce((total, file) => total + file.sizeBytes, 0),
          resumable: true,
          selectionId: configurationId,
        })
        return started
      }),

      getDownloadProgress: (operationId) => Ref.get(downloadProgress).pipe(
        Effect.map((current) => current.get(operationId) ?? null),
      ),

      subscribeDownload: (operationId) => bridge.subscribeDownload(operationId).pipe(
        Stream.tap(storeDownloadProgress),
      ),
      cancelDownload: (operationId) => bridge.cancelDownload(operationId).pipe(
        Effect.zipRight(Ref.update(downloadProgress, (current) => {
          const existing = current.get(operationId)
          if (!existing) return current
          const next = new Map(current)
          next.set(operationId, { ...existing, status: "cancelled" })
          return next
        })),
      ),

      activate: (selectionId) => Effect.gen(function* () {
        const inventory = yield* bridge.getInventory
        const existing = [...inventory.running, ...inventory.downloaded]
          .find((choice) => choice.choiceId === selectionId)
        const selection = existing ?? (yield* getConfiguration(bridge, storage, selectionId))
        const activated = yield* bridge.activate(selection)

        if (activated.contextTokens !== selection.contextTokens) {
          return yield* (new SessionOperationFailed({
            operation: "activate local model",
            reason: `llama.cpp reported ${activated.contextTokens} context tokens after ${selection.contextTokens} were explicitly selected. Refusing to complete onboarding after a silent context reduction.`,
          }))
        }

        const expectedServingProfile = "servingProfile" in selection
          ? selection.servingProfile
          : undefined
        if (expectedServingProfile && activated.parallelSlots !== undefined
          && activated.parallelSlots !== expectedServingProfile.parallelSlots) {
          return yield* (new SessionOperationFailed({
            operation: "activate local model",
            reason: `llama.cpp reported ${activated.parallelSlots} parallel slots after ${expectedServingProfile.parallelSlots} were selected. Refusing to complete onboarding after a silent slot reduction.`,
          }))
        }

        // TODO(local-model-role-routing): Consume the persisted local-model
        // role when the agent/provider routing contract lands. Main mode must
        // disable subagent spawning; subagent mode must retain the cloud main
        // slot and route at most three subagents per session to this provider.
        // Do not duplicate runtime role routing inside onboarding.
        // Activation is not complete until the final provider identity is
        // explicitly assigned to both coding-agent slots.
        yield* account.updateModelConfig({
          primary: {
            providerId: activated.providerId,
            providerModelId: activated.providerModelId,
          },
          secondary: {
            providerId: activated.providerId,
            providerModelId: activated.providerModelId,
          },
        })
        return activated
      }),

      complete,
    })
  }),
)
