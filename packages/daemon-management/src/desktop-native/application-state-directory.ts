import { join, win32 } from "node:path"
import { Effect, Option } from "effect"
import { NativeHostUnavailable } from "./index"

/** All participants derive the same lock path; native admission additionally validates its volume and ACL. */
export const applicationStateDirectory = (options: {
  readonly platform: NodeJS.Platform
  readonly dataDirectory: string
  readonly override: Option.Option<string>
}) => Effect.gen(function* () {
  const path = options.platform === "win32" ? win32 : { join }
  const directory = Option.getOrElse(options.override, () => path.join(options.dataDirectory, "state"))
  if (options.platform === "win32" && (!/^(?:\\\\\?\\)?[a-zA-Z]:[\\/]/.test(directory) || directory.includes("\0"))) {
    return yield* new NativeHostUnavailable({ message: "Magnitude requires a local Windows user-data directory. Network and relative paths cannot hold application ownership." })
  }
  return directory
})
