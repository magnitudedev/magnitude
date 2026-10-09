import { BunContext } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import { ServerInstallationFailed, installLinuxServer, installMacServer, parseServerInstallation, removeLinuxServer, removeMacServer,
  requireInstalledRoot } from "@magnitudedev/daemon-management/desktop-native"
import { ServerSetupHostLive } from "../server/server-setup-live"
import { serverRemove, serverSetup } from "../server/server-setup"

const report = <E extends { readonly message: string }>(effect: Effect.Effect<void, E>) => Effect.runPromise(effect.pipe(
  Effect.catchAll(error => Effect.sync(() => { process.stderr.write(`${error.message}\n`); process.exitCode = 1 }))))

export const runServerSetup = () => report(serverSetup.pipe(Effect.provide(ServerSetupHostLive)))
export const runServerRemove = () => report(serverRemove.pipe(Effect.provide(ServerSetupHostLive)))

/** The hidden root command. It accepts no input beyond the user name and runs only from the installed CLI as root. */
export const runServerRootStep = (argv: readonly string[]) => report(Effect.gen(function* () {
  const request = yield* parseServerInstallation(argv, process.env)
  const cli = yield* requireInstalledRoot(process.platform, process.execPath)
  if (process.platform === "darwin") {
    if (request._tag === "Remove") return yield* removeMacServer
    // The LaunchDaemon serves as a person; macOS has no root-login server.
    if (Option.isNone(request.user)) return yield* new ServerInstallationFailed({ message: "Run `magnitude server setup` as yourself, without sudo." })
    return yield* installMacServer(request.user.value, cli)
  }
  if (request._tag === "Remove") return yield* removeLinuxServer
  yield* installLinuxServer(request.user)
}).pipe(Effect.provide(BunContext.layer)))
