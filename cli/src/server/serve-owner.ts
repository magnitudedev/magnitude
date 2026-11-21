import { Effect } from "effect"
import type { UpdateOwner } from "@magnitudedev/release/hosted-update"

/**
 * Who is running `serve`, which update checks report: a service manager (systemd sets INVOCATION_ID for a unit's
 * processes; launchd is the parent of a job it starts; a Windows service or a task run before sign-in runs in
 * session 0, `windowsServiceSession`), or a person in a terminal.
 */
export const resolveServeOwner = (windowsServiceSession: Effect.Effect<boolean>) => Effect.gen(function* () {
  if (process.platform === "win32") {
    const owner: UpdateOwner = (yield* windowsServiceSession) ? "service" : "headless"
    return owner
  }
  const owner: UpdateOwner = (process.platform === "linux" && process.env.INVOCATION_ID !== undefined) || (process.platform === "darwin" && process.ppid === 1)
    ? "service" : "headless"
  return owner
})
