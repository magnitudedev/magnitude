import { BunContext } from "@effect/platform-bun"
import { runMacInstallerCommand } from "@magnitudedev/daemon-management/application-update"
import { Effect } from "effect"
import { CLI_VERSION } from "../version"

export const runMacApplicationInstallation = (request: string) => Effect.runPromise(runMacInstallerCommand(request, CLI_VERSION).pipe(
  Effect.provide(BunContext.layer),
  Effect.catchAll(error => Effect.sync(() => { process.stderr.write(`${error.message}\n`); process.exitCode = 1 })),
))
