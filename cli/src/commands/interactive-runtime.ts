import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import type { AuthSource } from "../state/cli-atoms"
import {
  runInteractiveCommand,
  type InteractiveLaunchOptions,
} from "../runtime/interactive"
import { isDevelopmentBuild } from "../runtime/environment"
import { explainInteractiveFailure } from "../startup/service-startup-error"

const resolveEnvAuth = (): AuthSource => {
  const envKey = process.env.MAGNITUDE_API_KEY
  return envKey && envKey.trim()
    ? { source: "env", key: envKey, envVarName: "MAGNITUDE_API_KEY" }
    : { source: "none" }
}

export interface InteractiveCommandOptions {
  readonly resume?: true | string
  readonly prompt?: string
  readonly atif?: string
  readonly systemOverride?: string
}

export const runInteractive = (
  opts: InteractiveCommandOptions,
  setup: boolean,
) => {
  const developmentBuild = isDevelopmentBuild()

  const options: InteractiveLaunchOptions = {
    debug: developmentBuild,
    setup,
    developmentBuild,
    sessionStart: opts.resume === undefined
      ? { _tag: "new" }
      : opts.resume === true
        ? { _tag: "latest" }
        : { _tag: "resume", sessionId: opts.resume },
    initialPrompt: opts.prompt,
    envAuth: resolveEnvAuth(),
    sessionOptions: {
      disableShellSafeguards: false,
      disableCwdSafeguards: false,
      atifPath: opts.atif,
      solo: false,
      headless: false,
      systemPromptOverride: opts.systemOverride,
    },
  }

  return Effect.runPromise(runInteractiveCommand(options).pipe(
    Effect.provide([BunContext.layer, FetchHttpClient.layer]),
    Effect.catchAll((error) => Effect.sync(() => {
      process.stderr.write(`${explainInteractiveFailure(error)}\n`)
      return 1
    })),
  )).then((exitCode) => process.exit(exitCode))
}
