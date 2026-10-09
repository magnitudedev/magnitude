import { FileSystem } from "@effect/platform"
import { Effect, Option } from "effect"
import { dirname } from "node:path"

/** Data directory and home of the `magnitude` system user that runs the Linux server service. */
export const SERVER_DATA_DIRECTORY = "/var/lib/magnitude"
export const SERVER_USER = "magnitude"
/** Written by the root step of `magnitude server setup`; its presence selects the server profile. */
export const SERVER_MARKER_PATH = "/etc/magnitude/server"
export const SERVER_MARKER_CONTENT = "Magnitude runs as the system service on this machine. Remove it with `magnitude server remove`.\n"

/** macOS runs the server as a LaunchDaemon for the person who set it up; no server profile is needed. */
export const MAC_SERVER_LABEL = "dev.magnitude.server"
export const MAC_SERVER_PLIST = `/Library/LaunchDaemons/${MAC_SERVER_LABEL}.plist`

/** Root-owned, not writable by others, and reached without symbolic links. */
const rootProtected = (path: string, type: "File" | "Directory") => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const info = yield* fs.stat(path)
  return info.type === type && Option.getOrUndefined(info.uid) === 0 && (info.mode & 0o022) === 0
    && (yield* fs.realPath(path)) === path
})

/**
 * Linux only: the server profile applies to the service and to every CLI command once the marker
 * exists. A missing, foreign-owned or writable marker leaves the per-user profile in place.
 */
export const isServerProfileActive = (platform: string) => platform !== "linux" ? Effect.succeed(false) : Effect.all([
  rootProtected(SERVER_MARKER_PATH, "File"),
  rootProtected(dirname(SERVER_MARKER_PATH), "Directory"),
]).pipe(Effect.map(checks => checks.every(Boolean)), Effect.orElseSucceed(() => false))
