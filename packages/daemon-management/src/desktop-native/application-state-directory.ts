import { join, win32 } from "node:path"
import { Effect, Option } from "effect"
import { NativeHostUnavailable } from "./index"

/** Ownership is local to this machine, independently of durable model data and roaming profiles. */
export const applicationStateDirectory = (options: {
  readonly platform: NodeJS.Platform
  readonly dataDirectory: string
  readonly development: boolean
  readonly override: Option.Option<string>
  readonly localAppDataDirectory: Effect.Effect<string, NativeHostUnavailable>
}) => Effect.gen(function* () {
  if (Option.isSome(options.override)) return options.override.value
  if (options.platform !== "win32") return join(options.dataDirectory, "desktop")
  const local = yield* options.localAppDataDirectory
  if (!/^(?:\\\\\?\\)?[a-zA-Z]:[\\/]/.test(local) || local.includes("\0")) {
    return yield* new NativeHostUnavailable({ message: "Windows application ownership requires a local application-data directory." })
  }
  return win32.join(local, options.development ? "Magnitude Development" : "Magnitude", "desktop")
})
