import { Context, Effect, Layer, Stream } from "effect"
import { PRIMARY_SLOT_ID, type ModelSlot, type OnboardingState } from "@magnitudedev/protocol"
import { Onboarding } from "./service"
import { ModelSlotCoordinator } from "../model-slot-coordinator"

export interface OnboardingModelActivationApi {
  readonly reconcile: Effect.Effect<void>
}

export class OnboardingModelActivation extends Context.Tag("OnboardingModelActivation")<
  OnboardingModelActivation,
  OnboardingModelActivationApi
>() {}

export type OnboardingModelActivationAction = "load" | "complete" | "none"

export const onboardingModelActivationAction = (
  state: OnboardingState,
  primary: ModelSlot,
): OnboardingModelActivationAction => {
  if (!state.flows.model_setup.required) return "none"
  if (primary._tag === "Ready") return "complete"
  if (primary._tag === "UnloadedLocalModel") return "load"
  return "none"
}

/**
 * Turns the durable onboarding selection into a resident primary model.
 * Package and catalog reconciliation make an installed selection Unloaded;
 * this service then admits the ordinary slot-owned load and completes setup
 * only after that same slot reaches Ready.
 */
export const OnboardingModelActivationLive: Layer.Layer<
  OnboardingModelActivation,
  never,
  Onboarding | ModelSlotCoordinator
> = Layer.scoped(OnboardingModelActivation, Effect.gen(function* () {
  const onboarding = yield* Onboarding
  const modelSlots = yield* ModelSlotCoordinator
  const lock = yield* Effect.makeSemaphore(1)

  const reconcile = lock.withPermits(1)(Effect.gen(function* () {
    const state = yield* onboarding.state
    const primary = (yield* modelSlots.snapshot).state.slots.primary
    switch (onboardingModelActivationAction(state, primary)) {
      case "complete":
        yield* onboarding.complete("model_setup")
        return
      case "load":
        if (primary._tag !== "UnloadedLocalModel") return
        yield* Effect.scoped(modelSlots.acquireLocalModel(
          PRIMARY_SLOT_ID,
          primary.selection.providerModelId,
        ))
        return
      case "none":
        return
    }
  })).pipe(
    Effect.catchAll((error) => Effect.logWarning("Unable to advance onboarding model activation").pipe(
      Effect.annotateLogs({ error: error.message }),
    )),
  )

  yield* reconcile.pipe(Effect.forkScoped)
  yield* Stream.merge(modelSlots.changes, onboarding.changes).pipe(
    Stream.runForEach(() => reconcile),
    Effect.forkScoped,
  )

  return OnboardingModelActivation.of({ reconcile })
}))
