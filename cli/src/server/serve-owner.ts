import { Effect } from "effect"
import type { UpdateOwner } from "@magnitudedev/release/hosted-update"

/**
 * Classifies `serve` for update checks and shutdown instructions: a service manager
 * (systemd sets INVOCATION_ID and, since systemd 248, SYSTEMD_EXEC_PID for the process
 * it launched; launchd is the parent of a job it starts; a Windows service or a task run
 * before sign-in runs in session 0, `windowsServiceSession`), or a person in a terminal.
 */
export const resolveServeOwner = (windowsServiceSession: Effect.Effect<boolean>) => Effect.gen(function* () {
  if (process.platform === "win32") {
    const owner: UpdateOwner = (yield* windowsServiceSession) ? "service" : "headless"
    return owner
  }
  // Child processes inherit INVOCATION_ID from a parent systemd unit; systemd 248+
  // sets SYSTEMD_EXEC_PID to the unit's main process PID so descendants are not
  // misclassified as managed services.
  const systemd = process.platform === "linux"
    && process.env.INVOCATION_ID !== undefined
    && (process.env.SYSTEMD_EXEC_PID === undefined || process.env.SYSTEMD_EXEC_PID === String(process.pid))
  const launchd = process.platform === "darwin" && process.ppid === 1
  const owner: UpdateOwner = systemd || launchd ? "service" : "headless"
  return owner
})
