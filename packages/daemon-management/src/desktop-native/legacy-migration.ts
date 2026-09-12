import { Context, Effect, Option, Schema } from "effect"
import { FSM } from "@magnitudedev/utils"
import { LegacyOwner } from "./legacy-owner"
import { LegacyTree } from "./legacy-tree"
import { LegacyStartupRegistration } from "./legacy-startup"

export const LegacyServiceSource = Schema.Union(
  Schema.TaggedStruct("Missing", {}),
  Schema.TaggedStruct("Absent", { owner: LegacyOwner }),
  Schema.TaggedStruct("Captured", { tree: LegacyTree }),
)
export type LegacyServiceSource = typeof LegacyServiceSource.Type
export const LegacyMigrationPlan = Schema.Struct({
  startup: Schema.optionalWith(LegacyStartupRegistration, { as: "Option", exact: true }),
  service: LegacyServiceSource,
})
export type LegacyMigrationPlan = typeof LegacyMigrationPlan.Type
class Prepared extends Schema.TaggedClass<Prepared>()("Prepared", { plan: LegacyMigrationPlan }) {}
class Unregistered extends Schema.TaggedClass<Unregistered>()("Unregistered", { plan: LegacyMigrationPlan }) {}
class Retired extends Schema.TaggedClass<Retired>()("Retired", { plan: LegacyMigrationPlan }) {}
class LoginTransferred extends Schema.TaggedClass<LoginTransferred>()("LoginTransferred", { plan: LegacyMigrationPlan }) {}
class Complete extends Schema.TaggedClass<Complete>()("Complete", {}) {}
export const LegacyMigrationState = Schema.Union(Prepared, Unregistered, Retired, LoginTransferred, Complete)
export type LegacyMigrationState = typeof LegacyMigrationState.Type
const machine = FSM.defineFSM({ Prepared, Unregistered, Retired, LoginTransferred, Complete }, {
  Prepared: ["Unregistered"], Unregistered: ["Retired"], Retired: ["LoginTransferred"], LoginTransferred: ["Complete"], Complete: [],
} as const)
export class LegacyMigrationFailed extends Schema.TaggedError<LegacyMigrationFailed>()("LegacyMigrationFailed", { message: Schema.String }) {}

/** Each action must tolerate replay after a crash between its external commit and checkpoint. */
export interface LegacyMigrationActions {
  readonly prepare: Effect.Effect<LegacyMigrationPlan, LegacyMigrationFailed>
  readonly unregister: (plan: LegacyMigrationPlan) => Effect.Effect<void, LegacyMigrationFailed>
  readonly retire: (service: LegacyServiceSource) => Effect.Effect<void, LegacyMigrationFailed>
  readonly transferLogin: (enabled: boolean) => Effect.Effect<void, LegacyMigrationFailed>
  readonly cleanup: (service: LegacyServiceSource) => Effect.Effect<void, LegacyMigrationFailed>
}
export const LegacyMigrationActions = Context.GenericTag<LegacyMigrationActions>("@magnitudedev/daemon-management/LegacyMigrationActions")
export interface LegacyMigrationJournal {
  readonly read: Effect.Effect<Option.Option<LegacyMigrationState>, LegacyMigrationFailed>
  readonly write: (state: LegacyMigrationState) => Effect.Effect<void, LegacyMigrationFailed>
}
export const LegacyMigrationJournal = Context.GenericTag<LegacyMigrationJournal>("@magnitudedev/daemon-management/LegacyMigrationJournal")

/** Called only while the desktop owns the application lock, before its service can start. */
export const makeLegacyMigration = Effect.gen(function* () {
  const actions = yield* LegacyMigrationActions
  const journal = yield* LegacyMigrationJournal
  const lock = yield* Effect.makeSemaphore(1)
  const run = lock.withPermits(1)(Effect.gen(function* () {
    const previous = yield* journal.read
    let state: LegacyMigrationState
    if (Option.isSome(previous)) state = previous.value
    else {
      state = new Prepared({ plan: yield* actions.prepare })
      // The original login preference and exact groups survive bootout and process death.
      yield* journal.write(state)
    }
    for (;;) {
      switch (state._tag) {
        case "Prepared":
          if (Option.isSome(state.plan.startup)) yield* actions.unregister(state.plan)
          state = machine.transition(state, "Unregistered", { plan: state.plan })
          break
        case "Unregistered":
          yield* actions.retire(state.plan.service)
          state = machine.transition(state, "Retired", { plan: state.plan })
          break
        case "Retired":
          if (Option.isSome(state.plan.startup)) yield* actions.transferLogin(state.plan.startup.value.enabled)
          state = machine.transition(state, "LoginTransferred", { plan: state.plan })
          break
        case "LoginTransferred":
          yield* actions.cleanup(state.plan.service)
          state = machine.transition(state, "Complete", {})
          break
        case "Complete": return
      }
      yield* journal.write(state)
    }
  }))
  return { run }
})
