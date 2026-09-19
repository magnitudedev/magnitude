import { Effect, Schema } from "effect"
import { ApplicationIdentity, assertServiceExited } from "../application-identity"
import { desktopSession } from "../desktop-session"
import { AssertionFailure } from "../domain"

type Session = Effect.Effect.Success<ReturnType<typeof desktopSession>>
export const UpdateBaseline = Schema.Struct({ version: Schema.String, theme: Schema.Literal("dark"),
  previousOwner: ApplicationIdentity, reopenedOwner: ApplicationIdentity })

/** Establish persisted state in the real older application before any update offer is published. */
export const verifyUpdateBaseline = (session: Pick<Session, "driver" | "stop">, version: string) => Effect.gen(function* () {
  let driver = yield* session.driver
  yield* driver.ready()
  if ((yield* driver.host()) !== version) return yield* new AssertionFailure({ message: "Update baseline desktop does not match the admitted previous version" })
  const previousOwner = yield* driver.identity()
  yield* driver.theme("dark")
  yield* driver.updates.automatic(false)
  yield* driver.quit()
  yield* session.stop
  yield* assertServiceExited(previousOwner.servicePid)
  driver = yield* session.driver
  yield* driver.ready()
  if ((yield* driver.host()) !== version) return yield* new AssertionFailure({ message: "Reopened update baseline changed its version before updating" })
  yield* driver.verifyTheme("dark")
  const reopenedOwner = yield* driver.identity()
  if (reopenedOwner.applicationPid === previousOwner.applicationPid || reopenedOwner.serviceInstance === previousOwner.serviceInstance) {
    return yield* new AssertionFailure({ message: "Update baseline persistence was not observed across a new application and service instance" })
  }
  return UpdateBaseline.make({ version, theme: "dark", previousOwner, reopenedOwner })
})
