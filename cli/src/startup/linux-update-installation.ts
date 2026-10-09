import { BunContext } from "@effect/platform-bun"
import { acquireUpdateInstallationLease, completeLinuxUpdateHandoff, installLinuxApplicationUpdate, guardLinuxInstallerParent, LinuxUpdateHandoffRequest, linuxPackageUpdateExitCode,
  parseLinuxUpdateInstallation,
  nativeHostLayer, relaunchLinuxAfterUpdate, unixPrivateFilePermissions } from "@magnitudedev/daemon-management/desktop-native"
import { guardedCommandLayer } from "@magnitudedev/utils/guarded-command"
import { Effect } from "effect"
import { CLI_VERSION } from "../version"
import { writeSync } from "node:fs"
import { readUpdateHandoff, UpdateHandoffChannelFailed } from "./update-handoff-channel"

/** Entered as root through sudo or Polkit; the raw arguments must match the installer's exact grammar. */
export const runLinuxUpdateInstallation = (argv: readonly string[]) => Effect.runPromise(
  parseLinuxUpdateInstallation(argv).pipe(Effect.flatMap(({ request, parentStdin }) =>
    (parentStdin ? guardLinuxInstallerParent : Effect.void).pipe(Effect.zipRight(installLinuxApplicationUpdate(request, CLI_VERSION))))).pipe(
    Effect.provide([BunContext.layer, guardedCommandLayer("/usr/lib/magnitude-desktop/resources/magnitude-command")]),
    Effect.catchAll(error => Effect.sync(() => {
      process.stderr.write(`${error.message}\n`)
      process.exitCode = linuxPackageUpdateExitCode[error.reason]
    })),
  ),
)

export const runLinuxUpdateHandoff = () => Effect.runPromise(Effect.gen(function* () {
  const channel = yield* readUpdateHandoff(process.stdin, LinuxUpdateHandoffRequest)
  const completed = yield* Effect.scoped(Effect.gen(function* () {
    yield* acquireUpdateInstallationLease(channel.request.stateDirectory)
    yield* Effect.try({ try: () => writeSync(3, "ready\n"), catch: () => new UpdateHandoffChannelFailed() })
    yield* channel.awaitOwnerExit
    return yield* completeLinuxUpdateHandoff(channel.request).pipe(Effect.exit)
  })).pipe(Effect.provide([nativeHostLayer("/usr/lib/magnitude-desktop/resources/desktop-host.node"), unixPrivateFilePermissions]))
  yield* relaunchLinuxAfterUpdate(channel.request)
  yield* completed
}).pipe(Effect.provide(BunContext.layer), Effect.catchAll(() => Effect.sync(() => { process.exitCode = 1 }))))
