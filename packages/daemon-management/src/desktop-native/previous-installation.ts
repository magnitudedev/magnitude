import { Context, Effect, Option, Schema } from "effect"
import { LegacyStartupRegistration } from "./legacy-startup"
import { LegacyTree } from "./legacy-tree"

export class PreviousInstallationFailed extends Schema.TaggedError<PreviousInstallationFailed>()("PreviousInstallationFailed", {
  message: Schema.String,
}) {}
export const PreviousInstallationPlan = Schema.TaggedStruct("Unix", {
  startup: Schema.optionalWith(LegacyStartupRegistration, { as: "Option", exact: true }),
  tree: Schema.optionalWith(LegacyTree, { as: "Option", exact: true }),
})
export type PreviousInstallationPlan = typeof PreviousInstallationPlan.Type
export interface PreviousInstallation {
  readonly inspect: Effect.Effect<Option.Option<PreviousInstallationPlan>, PreviousInstallationFailed>
  /** Idempotent; the caller has saved the process tree before registration removal. */
  readonly retire: (plan: PreviousInstallationPlan, checkpoint: (plan: PreviousInstallationPlan) => Effect.Effect<void, PreviousInstallationFailed>) => Effect.Effect<void, PreviousInstallationFailed>
}
export const PreviousInstallation = Context.GenericTag<PreviousInstallation>("@magnitudedev/daemon-management/PreviousInstallation")
export interface PreviousInstallationJournal {
  readonly read: Effect.Effect<Option.Option<PreviousInstallationPlan>, PreviousInstallationFailed>
  readonly write: (plan: PreviousInstallationPlan) => Effect.Effect<void, PreviousInstallationFailed>
  readonly clear: Effect.Effect<void, PreviousInstallationFailed>
}
export const PreviousInstallationJournal = Context.GenericTag<PreviousInstallationJournal>("@magnitudedev/daemon-management/PreviousInstallationJournal")

/** Runs under the desktop lock. No completion flag can conceal a reinstalled old service. */
export const makePreviousInstallationUpgrade = Effect.gen(function* () {
  const installation = yield* PreviousInstallation
  const journal = yield* PreviousInstallationJournal
  const lock = yield* Effect.makeSemaphore(1)
  const finish = (plan: PreviousInstallationPlan) => installation.retire(plan, journal.write).pipe(Effect.zipRight(journal.clear))
  return lock.withPermits(1)(Effect.gen(function* () {
    const pending = yield* journal.read
    if (Option.isSome(pending)) yield* finish(pending.value)
    const plan = yield* installation.inspect
    if (Option.isNone(plan)) return
    yield* journal.write(plan.value)
    yield* finish(plan.value)
    if (Option.isSome(yield* installation.inspect)) {
      return yield* new PreviousInstallationFailed({ message: "The previous Magnitude service restarted during upgrade." })
    }
  }).pipe(Effect.timeoutFail({ duration: "60 seconds", onTimeout: () => new PreviousInstallationFailed({
    message: "Magnitude could not finish stopping its previous installation within the startup deadline.",
  }) })))
})
