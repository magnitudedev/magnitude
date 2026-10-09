import { userInfo } from "node:os"
import { Effect } from "effect"
import { FileSystem } from "@effect/platform"
import { BunFileSystem } from "@effect/platform-bun"
import { MAC_SERVER_LABEL, MAC_SERVER_PLIST, SERVER_USER, isServerProfileActive } from "@magnitudedev/daemon-management/desktop-native"
import type { UpdateOwner } from "@magnitudedev/release/hosted-update"

/**
 * Who is running `serve`: the system service set up by `magnitude server setup`, or a person in a
 * terminal. On Linux the service is the `magnitude` user under the server profile; on macOS it is the
 * LaunchDaemon job (launchd names it in XPC_SERVICE_NAME), which runs as the person who set it up.
 */
export const resolveServeOwner = Effect.gen(function* () {
  const server = yield* isServerProfileActive(process.platform).pipe(Effect.provide(BunFileSystem.layer))
  const service = process.platform === "linux" ? server && userInfo().username === SERVER_USER
    : process.platform === "darwin" && process.env.XPC_SERVICE_NAME === MAC_SERVER_LABEL
  return { server, owner: (service ? "service" : "headless") satisfies UpdateOwner as UpdateOwner }
})

/** Whether `magnitude server setup` has registered the service on this machine. */
export const isServerSetUp = (process.platform === "darwin"
  ? Effect.flatMap(FileSystem.FileSystem, fs => fs.exists(MAC_SERVER_PLIST)).pipe(Effect.orElseSucceed(() => false))
  : isServerProfileActive(process.platform)).pipe(Effect.provide(BunFileSystem.layer))
