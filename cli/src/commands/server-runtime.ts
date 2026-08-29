import { FetchHttpClient } from "@effect/platform"
import * as FileSystem from "@effect/platform/FileSystem"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import * as Terminal from "@effect/platform/Terminal"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import {
  confirmServicePublicReady,
  startServiceManager,
  stopService,
} from "../server/service"
import { probeTerminalAppearance } from "../platform/terminal-appearance"
import {
  makeInlineServiceStartupPresenter,
} from "../startup/inline-service-lifecycle"
import { resolveCliTheme } from "../utils/theme"
import { makeCliUpdater, updateReleaseNotesUrl } from "../features/update/updater"
import { CLI_VERSION } from "../version"
import { isDevelopmentBuild } from "../runtime/environment"
import { startingAcnConnection } from "../server/acn-connection"
import {
  explainError,
  explainServiceStartupFailure,
} from "../startup/service-startup-error"

const runServiceStartEffect = Effect.scoped(Effect.gen(function* () {
  const appearance = yield* probeTerminalAppearance()
  const theme = resolveCliTheme(appearance)
  const startup = yield* makeInlineServiceStartupPresenter(theme, {
    showReadyWhenNoWork: true,
  })
  const updater = yield* makeCliUpdater({
    currentVersion: CLI_VERSION,
    developmentBuild: isDevelopmentBuild(),
  })
  const discovery = yield* updater.discover

  const started = yield* Effect.exit(startServiceManager(Option.some(startup.acquisitionObserver)))
  if (started._tag === "Failure") {
    return yield* Effect.failCause(started.cause)
  }
  yield* startup.acquisitionSucceeded

  const connection = yield* startingAcnConnection
  yield* startup.run(connection.startup)
  yield* confirmServicePublicReady

  const latest = yield* discovery.fresh
  if (Option.isSome(latest)) {
    yield* Effect.sync(() => {
      process.stdout.write([
        `Update available! ${CLI_VERSION} → ${latest.value}`,
        `Release notes: ${updateReleaseNotesUrl(latest.value)}`,
        "Run `magnitude update` to install it.",
        "",
      ].join("\n"))
    })
  }
}))

const runServiceStopEffect = stopService.pipe(
  Effect.tap(() => Effect.sync(() => {
    process.stdout.write("Magnitude service stopped.\n")
  })),
)

type ServiceRequirements =
  | FileSystem.FileSystem
  | CommandExecutor.CommandExecutor
  | Path.Path
  | HttpClient.HttpClient
  | Terminal.Terminal

const run = (
  effect: Effect.Effect<void, unknown, ServiceRequirements>,
  explain: (error: unknown) => string,
) => Effect.runPromise(effect.pipe(
  Effect.provide([BunContext.layer, FetchHttpClient.layer]),
  Effect.catchAll((error) => Effect.sync(() => {
    process.stderr.write(`${explain(error)}\n`)
    process.exitCode = 1
  })),
))

export const runServiceStart = () => run(runServiceStartEffect, explainServiceStartupFailure)

export const runServiceStop = () => run(runServiceStopEffect, explainError)
