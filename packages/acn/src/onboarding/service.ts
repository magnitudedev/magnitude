import { Context, Effect, Layer, Schema, Stream } from "effect"
import {
  OnboardingError,
  OnboardingMirror,
  type MirroredSnapshot,
  type OnboardingFlowId,
  type OnboardingState,
} from "@magnitudedev/protocol"
import { MagnitudeStorage } from "@magnitudedev/storage"
import type { ConfigStorageShape } from "@magnitudedev/storage"
import { makeMirroredState, MirroredStateChanges } from "../mirrored-state"

const FLOW_VERSIONS = {
  model_setup: 1,
} as const satisfies Record<OnboardingFlowId, number>

export interface OnboardingApi {
  readonly state: Effect.Effect<OnboardingState, OnboardingError>
  readonly snapshot: Effect.Effect<MirroredSnapshot<OnboardingState>>
  readonly changes: Stream.Stream<MirroredSnapshot<OnboardingState>>
  readonly complete: (flowId: OnboardingFlowId) => Effect.Effect<void, OnboardingError>
  readonly reopen: (flowId: OnboardingFlowId) => Effect.Effect<void, OnboardingError>
}

export class Onboarding extends Context.Tag("Onboarding")<Onboarding, OnboardingApi>() {}

const onboardingError = (operation: string, cause: unknown): OnboardingError =>
  new OnboardingError({
    operation,
    message: cause instanceof Error ? cause.message : String(cause),
  })

type OnboardingStorage = Pick<
  ConfigStorageShape,
  "getOnboardingConfig" | "completeOnboardingFlow" | "reopenOnboardingFlow"
>

type OnboardingPersistenceApi = Omit<OnboardingApi, "snapshot" | "changes">

export const makeOnboarding = (storage: OnboardingStorage): OnboardingPersistenceApi => {
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

  return {
    state,
    complete: (flowId) => storage.completeOnboardingFlow(
      flowId,
      FLOW_VERSIONS[flowId],
      new Date().toISOString(),
    ).pipe(
      Effect.mapError((cause) => onboardingError("complete onboarding flow", cause)),
    ),
    reopen: (flowId) => storage.reopenOnboardingFlow(flowId).pipe(
      Effect.mapError((cause) => onboardingError("reopen onboarding flow", cause)),
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
      complete: (flowId) => persisted.complete(flowId).pipe(Effect.zipRight(refresh)),
      reopen: (flowId) => persisted.reopen(flowId).pipe(Effect.zipRight(refresh)),
    })
  }),
)
