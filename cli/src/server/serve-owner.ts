import { Effect } from "effect"
import type { UpdateOwner } from "@magnitudedev/release/hosted-update"

/**
 * Who is running `serve`, which update checks report: a service manager (systemd sets INVOCATION_ID for a unit's
 * processes; launchd is the parent of a job it starts), or a person in a terminal.
 */
export const resolveServeOwner = Effect.sync((): UpdateOwner =>
  (process.platform === "linux" && process.env.INVOCATION_ID !== undefined) || (process.platform === "darwin" && process.ppid === 1)
    ? "service" : "headless")
