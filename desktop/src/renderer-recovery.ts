import { Effect, Ref, Schema } from "effect"
import { FSM } from "@magnitudedev/utils"

class Operational extends Schema.TaggedClass<Operational>()("Operational", {
  retries: Schema.Number.pipe(Schema.int(), Schema.between(0, 3)),
}) {}
class Suspended extends Schema.TaggedClass<Suspended>()("Suspended", {}) {}
const machine = FSM.defineFSM({ Operational, Suspended }, {
  Operational: ["Suspended"], Suspended: ["Operational"],
} as const)

/** Automatic window recovery is bounded; only an explicit Open renews an exhausted budget. */
export const makeRendererRecovery = Effect.gen(function* () {
  const state = yield* Ref.make<Operational | Suspended>(new Operational({ retries: 0 }))
  return {
    crashed: Ref.modify(state, current => current._tag === "Suspended" ? [false, current]
      : current.retries === 3 ? [false, machine.transition(current, "Suspended", {})]
      : [true, machine.hold(current, { retries: current.retries + 1 })]),
    loadFailed: Ref.update(state, current => current._tag === "Suspended" ? current : machine.transition(current, "Suspended", {})),
    open: Ref.modify(state, current => current._tag === "Operational" ? [false, current]
      : [true, machine.transition(current, "Operational", { retries: 0 })]),
  }
})
