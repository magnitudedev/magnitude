import { Effect, Option, Ref, Schema } from "effect"
import { defineFSM } from "@magnitudedev/utils/fsm"
import { Candidate } from "./candidate"
import { InstalledApplication, Installer } from "./installer"
import { AssertionFailure, Target } from "./domain"

class Absent extends Schema.TaggedClass<Absent>()("Absent", { candidate: Candidate }) {}
class Present extends Schema.TaggedClass<Present>()("Present", { application: InstalledApplication }) {}
const lifecycle = defineFSM({ Absent, Present }, { Absent: ["Absent", "Present"], Present: ["Absent", "Present"] })
/** Explicit removal updates ownership, so final cleanup cannot uninstall the same package twice. */
export const installationSession = (candidate: Candidate, onCleanupError: (detail: string) => void) => Effect.gen(function* () {
  const installer = yield* Installer
  const state = yield* Ref.make<Absent | Present>(new Absent({ candidate }))
  const semaphore = yield* Effect.makeSemaphore(1)
  // Callers hold the gate across native mutation and its corresponding ownership transition.
  const install = Effect.gen(function* () {
    const current = yield* Ref.get(state)
    if (current._tag === "Present") return current.application
    const application = yield* installer.install(current.candidate)
    yield* Ref.set(state, lifecycle.transition(current, "Present", { application }))
    return application
  })
  const uninstall = Effect.gen(function* () {
    const current = yield* Ref.get(state)
    if (current._tag === "Absent") return
    yield* installer.uninstall(current.application)
    yield* Ref.set(state, lifecycle.transition(current, "Absent", { candidate: current.application.candidate }))
  })
  const get = semaphore.withPermits(1)(Effect.uninterruptible(install))
  const remove = semaphore.withPermits(1)(Effect.uninterruptible(uninstall))
  const current = semaphore.withPermits(1)(Ref.get(state).pipe(Effect.map(value => value._tag === "Present" ? Option.some(value.application) : Option.none())))
  const resetTo = (next: Candidate) => Effect.gen(function* () {
    if (!Schema.equivalence(Target)(candidate.target, next.target)) return yield* new AssertionFailure({ message: "Cannot replace an installation with a different test target" })
    yield* uninstall
    const absent = yield* Ref.get(state)
    if (absent._tag !== "Absent") return yield* Effect.dieMessage("Installation gate did not retain absent ownership")
    yield* Ref.set(state, lifecycle.transition(absent, "Absent", { candidate: next }))
  })
  /** Select the next fixture without installing it; preserve lazy ownership after temporary suites. */
  const reset = (next: Candidate) => semaphore.withPermits(1)(Effect.uninterruptible(resetTo(next)))
  /** Fixture preparation only: replacing an installer cannot qualify an application's self-update. */
  const replace = (next: Candidate) => semaphore.withPermits(1)(Effect.uninterruptible(Effect.gen(function* () {
    const current = yield* Ref.get(state)
    if (current._tag === "Present" && Schema.equivalence(Candidate)(current.application.candidate, next)) return current.application
    yield* resetTo(next)
    return yield* install
  })))
  /** Observation only: the application's updater already performed the replacement. */
  const adoptReplacement = (observed: InstalledApplication) => semaphore.withPermits(1)(Effect.gen(function* () {
    const present = yield* Ref.get(state)
    if (present._tag !== "Present" || !Schema.equivalence(Target)(present.application.candidate.target, observed.candidate.target)
      || ["root", "executable", "cli"].some(key => present.application[key as "root" | "executable" | "cli"] !== observed[key as "root" | "executable" | "cli"])) {
      return yield* new AssertionFailure({ message: "Observed update does not belong to the owned installation" })
    }
    yield* Ref.set(state, lifecycle.transition(present, "Present", { application: observed }))
  }))
  yield* Effect.addFinalizer(() => remove.pipe(Effect.catchAll(error => Effect.sync(() => { onCleanupError(error.message) }))))
  return { get, remove, replace, reset, current, adoptReplacement }
})
