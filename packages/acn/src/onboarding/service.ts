import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import {
  OnboardingError,
  OnboardingMirror,
  type MirroredSnapshot,
  type OnboardingState,
} from "@magnitudedev/acn-protocol"
import { MagnitudeStorage } from "@magnitudedev/storage"
import type { ConfigStorageShape } from "@magnitudedev/storage"
import { makeMirroredState, MirroredStateChanges } from "../mirrored-state"

export interface OnboardingApi {
  readonly state: Effect.Effect<OnboardingState, OnboardingError>
  readonly snapshot: Effect.Effect<MirroredSnapshot<OnboardingState>>
  readonly changes: Stream.Stream<MirroredSnapshot<OnboardingState>>
  readonly update: (completed: boolean) => Effect.Effect<void, OnboardingError>
}

export class Onboarding extends Context.Tag("Onboarding")<Onboarding, OnboardingApi>() {}

const onboardingError = (operation: string, cause: unknown): OnboardingError =>
  new OnboardingError({
    operation,
    message: cause instanceof Error ? cause.message : String(cause),
  })

type OnboardingStorage = Pick<
  ConfigStorageShape,
  "getOnboardingConfig" | "updateOnboardingState"
>

type OnboardingPersistenceApi = Omit<OnboardingApi, "snapshot" | "changes">

export const makeOnboarding = (storage: OnboardingStorage): OnboardingPersistenceApi => {
  const state = storage.getOnboardingConfig().pipe(
    Effect.map((config): OnboardingState => ({
      completed: Option.match(config, {
        onNone: () => false,
        onSome: (value) => value.completed,
      }),
    })),
    Effect.mapError((cause) => onboardingError("read onboarding state", cause)),
  )

  return {
    state,
    update: (completed) => storage.updateOnboardingState(completed).pipe(
      Effect.mapError((cause) => onboardingError("update onboarding state", cause)),
    ),
  }
}

export const OnboardingLive: Layer.Layer<
  Onboarding,
  never,
  MagnitudeStorage | MirroredStateChanges
> = Layer.effect(
  Onboarding,
  Effect.gen(function* () {
    const storage = yield* MagnitudeStorage
    const persisted = makeOnboarding(storage.config)
    const initial = yield* persisted.state.pipe(Effect.orDie)
    const mirror = yield* makeMirroredState(OnboardingMirror, initial)
    const refresh = persisted.state.pipe(
      Effect.flatMap((state) => mirror.setIfChanged(
        state,
        Schema.equivalence(OnboardingMirror.stateSchema),
      )),
      Effect.asVoid,
    )
    return Onboarding.of({
      state: mirror.get.pipe(Effect.map(({ state }) => state)),
      snapshot: mirror.get,
      changes: mirror.changes,
      update: (completed) => persisted.update(completed).pipe(Effect.zipRight(refresh)),
    })
  }),
)
