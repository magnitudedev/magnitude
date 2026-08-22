import { Context, Effect, Layer, Schema, Stream } from "effect"
import {
  Onboarding as OnboardingBoundary,
  OnboardingError,
  OnboardingState,
  type MirroredSnapshot,
} from "@magnitudedev/acn-protocol"
import { MagnitudeStorage } from "@magnitudedev/storage"
import type { StateHandle } from "@magnitudedev/storage"
import { AcnChanges } from "../changes"
import { makeMirroredState } from "../mirrored-state"

export interface OnboardingApi {
  readonly state: Effect.Effect<OnboardingState, OnboardingError>
  readonly snapshot: Effect.Effect<MirroredSnapshot<OnboardingState>>
  readonly changes: Stream.Stream<MirroredSnapshot<OnboardingState>>
  readonly complete: Effect.Effect<void, OnboardingError>
}

export class Onboarding extends Context.Tag("Onboarding")<Onboarding, OnboardingApi>() {}

const onboardingError = (operation: string, cause: unknown): OnboardingError =>
  new OnboardingError({
    operation,
    message: cause instanceof Error ? cause.message : String(cause),
  })

type OnboardingStorage = Pick<StateHandle<OnboardingState, unknown>, "get" | "update">

type OnboardingPersistenceApi = Omit<OnboardingApi, "snapshot" | "changes">

export const makeOnboarding = (storage: OnboardingStorage): OnboardingPersistenceApi => {
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
    const initial = yield* persisted.state.pipe(Effect.orDie)
    const mirror = yield* makeMirroredState<OnboardingState>(
      { name: OnboardingBoundary.GetOnboardingState.name },
      initial,
    )
    const refresh = persisted.state.pipe(
      Effect.flatMap((state) => mirror.setIfChanged(
        state,
        Schema.equivalence(OnboardingState),
      )),
      Effect.asVoid,
    )
    return Onboarding.of({
      state: mirror.get.pipe(Effect.map(({ state }) => state)),
      snapshot: mirror.get,
      changes: mirror.changes,
      complete: persisted.complete.pipe(Effect.zipRight(refresh)),
    })
  }),
)
