import { Cause, Context, Effect, Layer, Stream } from "effect"
import {
  SessionOperationFailed,
  type LocalInferenceOnboardingSnapshot,
  type LocalModelDownloadProgress,
  type SessionError,
} from "@magnitudedev/protocol"
import { MagnitudeStorage, type ModelConfig } from "@magnitudedev/storage"
import { Account } from "../account"
import { LlamaCppRuntimeBridge } from "./runtime-bridge"
import { recommendLocalModels, resolveConfiguration, stableCapacityFromCapabilities } from "./recommendations"
import type {
  EvaluatedLocalConfiguration,
  LlamaCppHuggingFaceSource,
  LlamaCppRuntimeBridgeShape,
} from "./types"

export const CLI_MODEL_SETUP_ONBOARDING_VERSION = 2

export interface LocalInferenceOnboardingApi {
  readonly getSnapshot: Effect.Effect<LocalInferenceOnboardingSnapshot, SessionError>
  readonly startDownload: (configurationId: string) => Effect.Effect<{ readonly operationId: string }, SessionError>
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
  expectedFiles: configuration.entry.files,
})

const getConfiguration = (
  bridge: LlamaCppRuntimeBridgeShape,
  configurationId: string,
): Effect.Effect<EvaluatedLocalConfiguration, SessionError> =>
  Effect.gen(function* () {
    const capabilities = yield* bridge.getCapabilities
    const configuration = resolveConfiguration(
      configurationId,
      stableCapacityFromCapabilities(capabilities),
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
        CLI_MODEL_SETUP_ONBOARDING_VERSION,
        new Date().toISOString(),
      ).pipe(Effect.mapError(toLocalInferenceError("persist CLI model setup onboarding")))

    return LocalInferenceOnboarding.of({
      getSnapshot: Effect.gen(function* () {
        const [onboarding, readiness, capabilities, inventory, usableConfiguration] = yield* Effect.all([
          storage.config.getOnboardingConfig().pipe(
            Effect.mapError(toLocalInferenceError("read local inference onboarding")),
          ),
          bridge.getReadiness,
          bridge.getCapabilities,
          bridge.getInventory,
          hasUsableConfiguration,
        ], { concurrency: "unbounded" })

        const stableCapacity = stableCapacityFromCapabilities(capabilities)
        const recommendations = recommendLocalModels(stableCapacity)
        const completedVersion = onboarding?.cliModelSetupVersion
        const completedCurrent = completedVersion !== undefined
          && completedVersion >= CLI_MODEL_SETUP_ONBOARDING_VERSION
        const warnings = [...capabilities.warnings]
        if (recommendations.length === 0) {
          warnings.push({
            code: "no_stable_capacity_recommendation",
            message: "No curated 16K-or-larger configuration fits after the stable operating-system reserve. Existing running or downloaded models can still be selected when the runtime integration reports them.",
          })
        }

        return {
          schemaVersion: CLI_MODEL_SETUP_ONBOARDING_VERSION,
          onboarding: {
            required: !completedCurrent,
            ...(completedVersion !== undefined ? { completedVersion } : {}),
            ...(onboarding?.completedAt ? { completedAt: onboarding.completedAt } : {}),
          },
          configuration: { usable: usableConfiguration },
          runtime: readiness,
          capabilities,
          running: [...inventory.running],
          downloaded: [...inventory.downloaded],
          recommendations,
          warnings,
        }
      }),

      startDownload: (configurationId) =>
        getConfiguration(bridge, configurationId).pipe(
          Effect.flatMap((configuration) => bridge.startDownload(makeSource(configuration))),
        ),

      subscribeDownload: (operationId) => bridge.subscribeDownload(operationId),
      cancelDownload: (operationId) => bridge.cancelDownload(operationId),

      activate: (selectionId) => Effect.gen(function* () {
        const inventory = yield* bridge.getInventory
        const existing = [...inventory.running, ...inventory.downloaded]
          .find((choice) => choice.choiceId === selectionId)
        const selection = existing ?? (yield* getConfiguration(bridge, selectionId))
        const activated = yield* bridge.activate(selection)

        if (activated.contextTokens !== selection.contextTokens) {
          return yield* (new SessionOperationFailed({
            operation: "activate local model",
            reason: `llama.cpp reported ${activated.contextTokens} context tokens after ${selection.contextTokens} were explicitly selected. Refusing to complete onboarding after a silent context reduction.`,
          }))
        }

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
