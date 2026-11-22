import { Effect } from "effect"
import { localModelIsOwnedByMagnitude, type LocalModel, type OnboardingSettings } from "@magnitudedev/acn-protocol"

export interface OnboardingEvidence<E> {
  readonly readSaved: Effect.Effect<OnboardingSettings, E>
  readonly markCompleted: Effect.Effect<void, E>
  readonly hasExistingChats: Effect.Effect<boolean>
  readonly localModels: Effect.Effect<readonly LocalModel[]>
}

/**
 * Resolves onboarding state for the service: upgrades from versions without an `onboarding` setting
 * auto-complete when the profile already has saved chats or Magnitude-downloaded models, while fresh
 * installs with 0 Magnitude downloads (even with externally discovered LM Studio / Ollama models) stay pending.
 */
export const resolveOnboardingState = <E>(evidence: OnboardingEvidence<E>): Effect.Effect<OnboardingSettings, E> =>
  Effect.gen(function* () {
    const saved = yield* evidence.readSaved
    if (saved.onboarding === "Completed") return saved
    const hasChats = yield* evidence.hasExistingChats
    const hasOwnedModel = hasChats
      ? false
      : (yield* evidence.localModels).some(localModelIsOwnedByMagnitude)
    if (hasChats || hasOwnedModel) {
      yield* evidence.markCompleted.pipe(Effect.ignore)
      return { onboarding: "Completed" as const }
    }
    return saved
  })
