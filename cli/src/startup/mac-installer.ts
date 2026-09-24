import { BunContext } from "@effect/platform-bun"
import { runMacInstallerCommand, runMacArchiveInstallerCommand } from "@magnitudedev/daemon-management/application-update"
import { applicationStateDirectory } from "@magnitudedev/daemon-management/desktop-native"
import { Effect, Option } from "effect"
import { CLI_VERSION } from "../version"
import { desktopDataDirectory } from "../server/application"

export const runMacApplicationInstallation = (request: string) => Effect.runPromise(runMacInstallerCommand(request, CLI_VERSION).pipe(
  Effect.provide(BunContext.layer),
  Effect.catchAll(error => Effect.sync(() => { process.stderr.write(`${error.message}\n`); process.exitCode = 1 })),
))

export const runMacArchiveInstallation = (request: string) => Effect.runPromise(Effect.gen(function* () {
  const stateDirectory = yield* applicationStateDirectory({ platform: process.platform, dataDirectory: desktopDataDirectory,
    override: Option.fromNullable(process.env.MAGNITUDE_DESKTOP_STATE_DIR) })
  yield* runMacArchiveInstallerCommand(request, stateDirectory, CLI_VERSION)
}).pipe(Effect.provide(BunContext.layer),
  Effect.catchAll(error => Effect.sync(() => { process.stderr.write(`${error.message}\n`); process.exitCode = 1 }))))
