import { BunContext } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import { ServerInstallationFailed, installLinuxServer, installMacServer, openServerFirewall, parseServerInstallation, removeLinuxServer, restoreLinuxServerGrant,
  removeMacServer, requireInstalledRoot } from "@magnitudedev/daemon-management/desktop-native"
import { ServerSetupHostLive } from "../server/server-setup-live"
import { serverRemove, serverSetup, serverWaitReady } from "../server/server-setup"

const report = <E extends { readonly message: string }>(effect: Effect.Effect<void, E>) => Effect.runPromise(effect.pipe(
  Effect.catchAll(error => Effect.sync(() => { process.stderr.write(`${error.message}\n`); process.exitCode = 1 }))))

// A request to a service that never answered may still be pending after setup gave up on it.
export const runServerSetup = () => report(serverSetup.pipe(Effect.provide(ServerSetupHostLive))).then(() => process.exit())
export const runServerRemove = () => report(serverRemove.pipe(Effect.provide(ServerSetupHostLive)))
export const runServerWaitReady = () => report(serverWaitReady.pipe(Effect.provide(ServerSetupHostLive))).then(() => process.exit())

/** The hidden root command. It accepts no input beyond the user name and runs only from the installed CLI as root. */
export const runServerRootStep = (argv: readonly string[]) => report(Effect.gen(function* () {
  const request = yield* parseServerInstallation(argv, process.env)
  const cli = yield* requireInstalledRoot(process.platform, process.execPath,
    request._tag === "Remove" ? "magnitude server remove" : "magnitude server setup")
  if (request._tag === "RestoreGrant") {
    if (process.platform !== "linux") return yield* new ServerInstallationFailed({ message: "Only Linux packages restore the server's update permission." })
    return yield* restoreLinuxServerGrant
  }
  if (process.platform === "darwin") {
    if (request._tag === "Remove") return yield* removeMacServer
    // The LaunchDaemon serves as a person; macOS has no root-login server.
    if (Option.isNone(request.user)) return yield* new ServerInstallationFailed({ message: "Run `magnitude server setup` as yourself, without sudo." })
    return yield* installMacServer(request.user.value, cli)
  }
  if (request._tag === "Remove") return yield* removeLinuxServer
  yield* installLinuxServer
  const firewall = yield* openServerFirewall
  if (Option.isSome(firewall)) yield* Effect.sync(() => { process.stdout.write(`${firewall.value}\n`) })
}).pipe(Effect.provide(BunContext.layer)))
