import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { updateActionFor } from "@magnitudedev/release"
import { makeCliUpdater } from "../features/update/updater"
import { Effect, Either, Option } from "effect"
import { executeUpdate } from "../features/update/execute"
import { isDevelopmentBuild } from "../runtime/environment"
import { CLI_VERSION } from "../version"

const runExplicitUpdate = Effect.gen(function* () {
  if (isDevelopmentBuild()) {
    yield* Effect.sync(() => {
      process.stderr.write("`magnitude update` is not available in development builds.\n")
    })
    return 1
  }

  const updater = yield* makeCliUpdater({
    currentVersion: CLI_VERSION,
    developmentBuild: false,
  })
  if (Option.isNone(updater.packageManager)) {
    yield* Effect.sync(() => {
      process.stderr.write(
        "Could not detect how Magnitude was installed. Update manually with `npm install -g @magnitudedev/cli`.\n",
      )
    })
    return 1
  }
  // The explicit command resolves its target the same way the prompt does —
  // channel-selected and readiness-verified — but ignores dismissals: asking
  // to update overrides having dismissed.
  const checked = yield* Effect.either(updater.updateTarget)
  if (Either.isLeft(checked)) {
    yield* Effect.sync(() => {
      process.stderr.write(`Could not check for updates: ${checked.left.reason}\n`)
    })
    return 1
  }
  if (Option.isNone(checked.right)) {
    yield* Effect.sync(() => {
      process.stdout.write("Magnitude is already up to date.\n")
    })
    return 0
  }
  return yield* executeUpdate(
    updater,
    updateActionFor(updater.packageManager.value, checked.right.value),
  )
}).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))

export const runUpdate = () => Effect.runPromise(runExplicitUpdate).then(
  (exitCode) => process.exit(exitCode),
)
