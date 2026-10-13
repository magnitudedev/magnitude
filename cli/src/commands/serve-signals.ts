import * as FileSystem from "@effect/platform/FileSystem"
import { Effect } from "effect"

const SIGHUP_MASK = 1n

/** True when Linux process status reports SIGHUP as ignored, as `nohup` leaves it. */
export const statusIgnoresHangup = (status: string): boolean => {
  const mask = /^SigIgn:\s*([0-9a-fA-F]+)\s*$/m.exec(status)?.[1]
  return mask !== undefined && (BigInt(`0x${mask}`) & SIGHUP_MASK) === SIGHUP_MASK
}

/**
 * Signals that stop the foreground server. A hang-up the launcher already ignores stays ignored,
 * so `nohup magnitude serve` survives the end of its terminal session. Only Linux exposes the
 * inherited disposition; elsewhere SIGHUP always stops the server.
 */
export const serveStopSignals = (platform: NodeJS.Platform) => Effect.gen(function* () {
  if (platform === "win32") return ["SIGINT", "SIGTERM", "SIGBREAK"] as const
  if (platform !== "linux") return ["SIGINT", "SIGTERM", "SIGHUP"] as const
  const fs = yield* FileSystem.FileSystem
  const status = yield* fs.readFileString("/proc/self/status").pipe(Effect.orElseSucceed(() => ""))
  return statusIgnoresHangup(status) ? ["SIGINT", "SIGTERM"] as const : ["SIGINT", "SIGTERM", "SIGHUP"] as const
})
