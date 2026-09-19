import { Effect, Ref, Schema } from "effect"
import { defineFSM } from "@magnitudedev/utils/fsm"
import { Candidate } from "./candidate"
import { InstalledApplication, Installer } from "./installer"

class Absent extends Schema.TaggedClass<Absent>()("Absent", {}) {}
class Present extends Schema.TaggedClass<Present>()("Present", { application: InstalledApplication }) {}
const lifecycle = defineFSM({ Absent, Present }, { Absent: ["Present"], Present: ["Absent"] })
/** Explicit removal updates ownership, so final cleanup cannot uninstall the same package twice. */
export const installationSession = (candidate: Candidate, onCleanupError: (detail: string) => void) => Effect.gen(function* () {
  const installer = yield* Installer
  const state = yield* Ref.make<Absent | Present>(new Absent({}))
  const semaphore = yield* Effect.makeSemaphore(1)
  const get = semaphore.withPermits(1)(Effect.uninterruptible(Effect.gen(function* () {
    const current = yield* Ref.get(state)
    if (current._tag === "Present") return current.application
    const application = yield* installer.install(candidate)
    yield* Ref.set(state, lifecycle.transition(current, "Present", { application }))
    return application
  })))
  const remove = semaphore.withPermits(1)(Effect.uninterruptible(Effect.gen(function* () {
    const current = yield* Ref.get(state)
    if (current._tag === "Absent") return
    yield* installer.uninstall(current.application)
    yield* Ref.set(state, lifecycle.transition(current, "Absent", {}))
  })))
  yield* Effect.addFinalizer(() => remove.pipe(Effect.catchAll(error => Effect.sync(() => { onCleanupError(error.message) }))))
  return { get, remove }
})
