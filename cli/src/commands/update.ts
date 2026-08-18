import type { Command } from "@commander-js/extra-typings"
import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { makeCliUpdater } from "../features/update/updater"
import { Effect, Option } from "effect"
import { executeUpdate } from "../features/update/execute"
import { isDevelopmentBuild } from "../runtime/environment"
import { CLI_VERSION } from "../version"

const runExplicitUpdate = Effect.gen(function* () {
  if (isDevelopmentBuild()) {
    yield* Effect.sync(() => {
      process.stderr.write("`magnitude update` is not available in development builds.\n")
      process.exitCode = 1
    })
    return
  }

  const updater = yield* makeCliUpdater({
    currentVersion: CLI_VERSION,
    developmentBuild: false,
  })
  if (Option.isNone(updater.updateAction)) {
    yield* Effect.sync(() => {
      process.stderr.write(
        "Could not detect how Magnitude was installed. Update manually with `npm install -g @magnitudedev/cli`.\n",
      )
      process.exitCode = 1
    })
    return
  }
  yield* executeUpdate(updater, updater.updateAction.value)
}).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))

export const registerUpdateCommand = (program: Command): void => {
  program
    .command("update")
    .description("Update Magnitude with the package manager that installed it")
    .action(() => Effect.runPromise(runExplicitUpdate))
}
