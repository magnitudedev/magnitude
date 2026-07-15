import { Context, Effect, Layer } from "effect"
import {
  OnboardingError,
  type OnboardingFlowId,
  type OnboardingState,
} from "@magnitudedev/protocol"
import { MagnitudeStorage } from "@magnitudedev/storage"
import type { ConfigStorageShape } from "@magnitudedev/storage"

const FLOW_VERSIONS = {
  model_setup: 1,
} as const satisfies Record<OnboardingFlowId, number>

export interface OnboardingApi {
  readonly state: Effect.Effect<OnboardingState, OnboardingError>
  readonly complete: (flowId: OnboardingFlowId) => Effect.Effect<void, OnboardingError>
}

export class Onboarding extends Context.Tag("Onboarding")<Onboarding, OnboardingApi>() {}

const onboardingError = (operation: string, cause: unknown): OnboardingError =>
  new OnboardingError({
    operation,
    message: cause instanceof Error ? cause.message : String(cause),
  })

type OnboardingStorage = Pick<
  ConfigStorageShape,
  "getOnboardingConfig" | "completeOnboardingFlow"
>

export const makeOnboarding = (storage: OnboardingStorage): OnboardingApi => {
  const state = storage.getOnboardingConfig().pipe(
    Effect.map((config): OnboardingState => {
      const completion = config?.completions?.model_setup ?? null
      const currentVersion = FLOW_VERSIONS.model_setup
      return {
        flows: {
          model_setup: {
            currentVersion,
            completedVersion: completion?.version ?? null,
            completedAt: completion?.completedAt ?? null,
            required: (completion?.version ?? 0) < currentVersion,
          },
        },
      }
    }),
    Effect.mapError((cause) => onboardingError("read onboarding state", cause)),
  )

  return Onboarding.of({
    state,
    complete: (flowId) => storage.completeOnboardingFlow(
      flowId,
      FLOW_VERSIONS[flowId],
      new Date().toISOString(),
    ).pipe(
      Effect.mapError((cause) => onboardingError("complete onboarding flow", cause)),
    ),
  })
}

export const OnboardingLive: Layer.Layer<Onboarding, never, MagnitudeStorage> = Layer.effect(
  Onboarding,
  Effect.gen(function* () {
    const storage = yield* MagnitudeStorage
    return makeOnboarding(storage.config)
  }),
)
