import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { runHostedSetup } from "../runtime/interactive"
import { explainInteractiveFailure } from "../startup/service-startup-error"
import { interactiveLaunchOptions } from "./interactive-runtime"

export const runPiSetup = () => Effect.runPromise(Effect.suspend(() => {
  if (!process.stdin.isTTY || !process.stdout.isTTY) {
    return Effect.sync(() => {
      process.stderr.write("Pi setup requires an interactive terminal\n")
      return 1
    })
  }
  return runHostedSetup({ ...interactiveLaunchOptions({}, true), setupHost: "pi" })
}).pipe(
  Effect.provide([BunContext.layer, FetchHttpClient.layer]),
  Effect.catchAll(error => Effect.sync(() => {
    process.stderr.write(`${explainInteractiveFailure(error)}\n`)
    return 1
  })),
)).then(exitCode => process.exit(exitCode))
