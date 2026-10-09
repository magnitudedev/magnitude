import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { installLinuxServer, parseServerInstallation, removeLinuxServer, requireInstalledRoot } from "@magnitudedev/daemon-management/desktop-native"
import { ServerSetupHostLive } from "../server/server-setup-live"
import { serverRemove, serverSetup } from "../server/server-setup"

const report = <E extends { readonly message: string }>(effect: Effect.Effect<void, E>) => Effect.runPromise(effect.pipe(
  Effect.catchAll(error => Effect.sync(() => { process.stderr.write(`${error.message}\n`); process.exitCode = 1 }))))

export const runServerSetup = () => report(serverSetup.pipe(Effect.provide(ServerSetupHostLive)))
export const runServerRemove = () => report(serverRemove.pipe(Effect.provide(ServerSetupHostLive)))

/** The hidden root command. It accepts no input beyond the user name and runs only from the installed CLI as root. */
export const runServerRootStep = (argv: readonly string[]) => report(Effect.gen(function* () {
  const request = yield* parseServerInstallation(argv, process.env)
  yield* requireInstalledRoot(process.platform, process.execPath)
  if (process.platform !== "linux") return yield* Effect.dieMessage("Server root steps are only implemented for Linux.")
  if (request._tag === "Remove") return yield* removeLinuxServer
  yield* installLinuxServer(request.user, request.uid)
}).pipe(Effect.provide(BunContext.layer)))
