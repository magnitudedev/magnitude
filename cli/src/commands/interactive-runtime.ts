import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import { dirname, isAbsolute } from "node:path"
import { writeFileAtomic } from "@magnitudedev/utils/atomic-file"
import { HostedSetupResult, hostedSetupFailure } from "@magnitudedev/client-common/harness-connections/hosted-setup"
import type { AuthSource } from "../state/cli-atoms"
import {
  runInteractiveCommand,
  runHostedSetup,
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

class HostedSetupInvalid extends Schema.TaggedError<HostedSetupInvalid>()("HostedSetupInvalid", {
  message: Schema.String,
}) {}

export interface InteractiveCommandOptions {
  readonly resume?: true | string
  readonly prompt?: string
  readonly atif?: string
  readonly systemOverride?: string
}

export const runInteractive = (
  opts: InteractiveCommandOptions,
  setup: boolean,
  hostedResultFile?: string,
) => {
  const developmentBuild = isDevelopmentBuild()

  const options: InteractiveLaunchOptions = {
    debug: developmentBuild,
    setup,
    setupHost: hostedResultFile === undefined ? undefined : "pi",
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

  const program = Effect.gen(function* () {
    if (hostedResultFile === undefined) return yield* runInteractiveCommand(options)
    const fs = yield* FileSystem.FileSystem
    if (!isAbsolute(hostedResultFile)) return yield* new HostedSetupInvalid({ message: "Hosted setup result path must be absolute" })
    const parent = yield* fs.stat(dirname(hostedResultFile))
    if (parent.type !== "Directory"
      || (process.platform !== "win32" && (parent.mode & 0o077) !== 0)
      || (yield* fs.exists(hostedResultFile))) {
      return yield* new HostedSetupInvalid({ message: "Hosted setup requires a new result file inside an existing private directory" })
    }
    // Verify writability without truncating an existing target.
    yield* fs.writeFileString(hostedResultFile, "", { flag: "wx", mode: 0o600 })
    const result = !process.stdin.isTTY || !process.stdout.isTTY
      ? hostedSetupFailure("Hosted setup requires an interactive terminal")
      : yield* runHostedSetup(options).pipe(Effect.catchAll(error =>
        Effect.succeed(hostedSetupFailure(explainInteractiveFailure(error)))))
    yield* writeFileAtomic(hostedResultFile, yield* Schema.encode(Schema.parseJson(HostedSetupResult))(result))
    return result._tag === "Failed" ? 1 : 0
  })
  return Effect.runPromise(program.pipe(
    Effect.provide([BunContext.layer, FetchHttpClient.layer]),
    Effect.catchAll((error) => Effect.sync(() => {
      process.stderr.write(`${explainInteractiveFailure(error)}\n`)
      return 1
    })),
  )).then((exitCode) => process.exit(exitCode))
}
