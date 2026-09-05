import { Context, Effect, Layer } from "effect"
import {
  Onboarding as OnboardingBoundary,
  OnboardingError,
  OnboardingState,
} from "@magnitudedev/acn-protocol"
import { MagnitudeStorage } from "@magnitudedev/storage"
import type { StateHandle } from "@magnitudedev/storage"
import { AcnChanges } from "../changes"

export interface OnboardingApi {
  readonly state: Effect.Effect<OnboardingState, OnboardingError>
  readonly complete: Effect.Effect<void, OnboardingError>
}

export class Onboarding extends Context.Tag("Onboarding")<Onboarding, OnboardingApi>() {}

const onboardingError = (operation: string, cause: unknown): OnboardingError =>
  new OnboardingError({
    operation,
    message: cause instanceof Error ? cause.message : String(cause),
  })

type OnboardingStorage = Pick<StateHandle<OnboardingState, unknown>, "get" | "update">

export const makeOnboarding = (storage: OnboardingStorage): OnboardingApi => {
  const state = storage.get.pipe(
    Effect.mapError((cause) => onboardingError("read onboarding state", cause)),
  )

  return {
    state,
    complete: storage.update((current) => current.completed ? current : { completed: true }).pipe(
      Effect.asVoid,
      Effect.mapError((cause) => onboardingError("complete onboarding", cause)),
    ),
  }
}

export const OnboardingLive: Layer.Layer<
  Onboarding,
  never,
  MagnitudeStorage | AcnChanges
> = Layer.effect(
  Onboarding,
  Effect.gen(function* () {
    const storage = yield* MagnitudeStorage
    const persisted = makeOnboarding(storage.onboarding)
    const changes = yield* AcnChanges
    return Onboarding.of({
      state: persisted.state,
      complete: persisted.complete.pipe(
        Effect.zipRight(changes.publish({ operation: OnboardingBoundary.getOnboardingState._tag })),
      ),
    })
  }),
)
