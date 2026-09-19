import { Effect } from "effect"
import { ApplicationIdentity, assertServiceExited } from "../application-identity"
import { desktopSession } from "../desktop-session"
import { AssertionFailure } from "../domain"
import { CliTests } from "./cli"

type Session = Effect.Effect.Success<ReturnType<typeof desktopSession>>
export const verifyServiceOwnership = (session: Session) => Effect.gen(function* () {
  const cli = yield* CliTests
  const observations: ApplicationIdentity[] = []
  let driver = yield* session.driver
  yield* driver.ready()
  let owner = yield* driver.identity()
  observations.push(owner)
  for (let cycle = 0; cycle < 3; cycle++) {
    for (let start = 0; start < 2; start++) {
      yield* cli.ensureService
      const current = yield* driver.identity()
      if (current.applicationPid !== owner.applicationPid || current.servicePid !== owner.servicePid || current.serviceInstance !== owner.serviceInstance) return yield* new AssertionFailure({ message: "Repeated service start replaced the existing owning application or service" })
      observations.push(current)
    }
    if (cycle === 2) break
    yield* driver.quit()
    yield* session.stop
    yield* assertServiceExited(owner.servicePid)
    driver = yield* session.driver
    yield* driver.ready()
    const replacement = yield* driver.identity()
    if (replacement.applicationPid === owner.applicationPid || replacement.serviceInstance === owner.serviceInstance) return yield* new AssertionFailure({ message: "Relaunch retained the old application or service identity" })
    owner = replacement
    observations.push(owner)
  }
  return observations
})
