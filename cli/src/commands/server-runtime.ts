import { FetchHttpClient } from "@effect/platform"
import * as FileSystem from "@effect/platform/FileSystem"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import * as Terminal from "@effect/platform/Terminal"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option, Schema } from "effect"
import {
  confirmServicePublicReady,
  installService,
  serviceStatus,
  startServiceManager,
  stopService,
  uninstallService,
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
import { explainServiceStartupFailure } from "../startup/service-startup-error"
import { runCommand } from "./output"

const ServiceActionSchema = Schema.Struct({
  action: Schema.Literal("install", "uninstall", "stop"),
})
const ServiceStatusSchema = Schema.Union(
  Schema.Struct({
    installed: Schema.Boolean,
    enabled: Schema.Boolean,
    managed: Schema.Boolean,
    running: Schema.Literal(false),
    state: Schema.Literal("Stopped"),
  }),
  Schema.Struct({
    installed: Schema.Boolean,
    enabled: Schema.Boolean,
    managed: Schema.Boolean,
    running: Schema.Literal(true),
    revision: Schema.Number,
    state: Schema.Literal("Starting", "Ready", "Stopping"),
  }),
)

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

type ServiceActionRequirements =
  | FileSystem.FileSystem
  | CommandExecutor.CommandExecutor
  | Path.Path
  | HttpClient.HttpClient

const live = <A, E>(effect: Effect.Effect<A, E, ServiceActionRequirements>) => effect.pipe(
  Effect.provide([BunContext.layer, FetchHttpClient.layer]),
)

const action = (
  name: "install" | "uninstall" | "stop",
  effect: Effect.Effect<unknown, unknown, ServiceActionRequirements>,
) => runCommand({
  effect: live(effect).pipe(Effect.as({ action: name })),
  schema: ServiceActionSchema,
  render: () => `Magnitude service ${{
    install: "installed",
    uninstall: "uninstalled",
    stop: "stopped",
  }[name]}.\n`,
})

export const runServiceInstall = () => action("install", installService)
export const runServiceUninstall = () => action("uninstall", uninstallService)
export const runServiceStart = () => run(runServiceStartEffect, explainServiceStartupFailure)
export const runServiceStop = () => action("stop", stopService)
export const runServiceStatus = () => runCommand({
  effect: live(serviceStatus),
  schema: ServiceStatusSchema,
  render: (status) => [
    `Installed: ${status.installed ? "yes" : "no"}`,
    `Enabled: ${status.enabled ? "yes" : "no"}`,
    `Managed runtime: ${status.managed ? "yes" : "no"}`,
    `Running: ${status.running ? "yes" : "no"}`,
    `State: ${status.state}`,
    ...("revision" in status ? [`Revision: ${status.revision}`] : []),
    "",
  ].join("\n"),
})
