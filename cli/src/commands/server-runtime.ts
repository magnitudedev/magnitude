import { Atom, Registry } from "@effect-atom/atom"
import { FetchHttpClient } from "@effect/platform"
import * as FileSystem from "@effect/platform/FileSystem"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import * as Terminal from "@effect/platform/Terminal"
import { BunContext } from "@effect/platform-bun"
import { formatLocalModelDisplayName } from "@magnitudedev/client-common"
import { Client } from "@magnitudedev/effect-query"
import {
  MAGNITUDE_SERVICE_ORIGIN,
  MagnitudeBoundary,
  magnitudeImplementationsLayer,
} from "@magnitudedev/sdk"
import { Effect, Layer, Option } from "effect"
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
import { existingAcnConnection, startingAcnConnection } from "../server/acn-connection"
import { explainServiceStartupFailure } from "../startup/service-startup-error"
import { runCommand } from "./output"

interface ActiveModel {
  readonly displayName: string
  readonly status: "Loading" | "Ready" | "Stopping"
}

interface ServiceStatusPresentation {
  readonly status: "Stopped" | "Starting" | "Ready" | "Stopping"
  readonly address: string
  readonly version: Option.Option<string>
  readonly startsAutomaticallyOnLogin: boolean
  readonly activeModel: Option.Option<ActiveModel>
}

const serviceAddress = new URL(MAGNITUDE_SERVICE_ORIGIN).host

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

export const runServiceInstall = () => runCommand({
  effect: live(installService),
  render: () => "Magnitude will start automatically when you log in.\n",
})
export const runServiceUninstall = () => runCommand({
  effect: live(uninstallService),
  render: () => "Magnitude service stopped and was removed from login startup.\nModels, settings, and sessions were kept.\n",
})
export const runServiceStart = () => run(runServiceStartEffect, explainServiceStartupFailure)
export const runServiceStop = () => runCommand({
  effect: live(stopService),
  render: () => "Magnitude service stopped.\n",
})

const readActiveModel = Effect.scoped(Effect.gen(function* () {
  const connection = yield* existingAcnConnection
  const registry = Registry.make()
  yield* Effect.addFinalizer(() => Effect.sync(() => registry.dispose()))
  const client = Client.make(
    MagnitudeBoundary,
    magnitudeImplementationsLayer(connection.protocolLayer.pipe(
      Layer.provide(FetchHttpClient.layer),
    )),
  )
  const catalog = yield* Registry.getResult(
    registry,
    Atom.make((get) => get(client.Models.GetCatalog({})).result),
  )
  if (catalog._tag === "Initializing") return Option.none<ActiveModel>()
  for (const entry of catalog.models) {
    if (entry._tag !== "Local") continue
    const model = entry.product
    const residency = model._tag === "Discovered"
      ? model.state._tag === "Ready" ? model.state.residencyState : undefined
      : "residencyState" in model.acquisitionState
        ? model.acquisitionState.residencyState
        : undefined
    if (residency === undefined) continue
    if (residency._tag !== "Requested"
      && residency._tag !== "Loading"
      && residency._tag !== "Ready"
      && residency._tag !== "Stopping") continue
    return Option.some({
      displayName: formatLocalModelDisplayName(model),
      status: residency._tag === "Requested" ? "Loading" as const : residency._tag,
    })
  }
  return Option.none<ActiveModel>()
}))

const publicServiceStatus = live(serviceStatus).pipe(
  Effect.flatMap((status) => {
    const activeModel = status.running && status.state === "Ready"
      ? readActiveModel.pipe(
          Effect.timeoutOption("2 seconds"),
          Effect.map(Option.flatten),
          Effect.orElseSucceed(() => Option.none<ActiveModel>()),
        )
      : Effect.succeed(Option.none<ActiveModel>())
    return activeModel.pipe(Effect.map((model) => ({
      status: status.state,
      address: serviceAddress,
      version: status.running ? Option.some(String(status.version)) : Option.none<string>(),
      startsAutomaticallyOnLogin: status.enabled,
      activeModel: model,
    })))
  }),
)

export const renderServiceStatus = (status: ServiceStatusPresentation): string => [
  "Magnitude service",
  `  Runtime         ${status.status}`,
  `  Starts at login ${status.startsAutomaticallyOnLogin ? "Yes" : "No"}`,
  ...(Option.isSome(status.version) ? [
    `  Version         ${status.version.value}`,
    `  Address         ${status.address}`,
  ] : []),
  ...(status.status === "Ready" ? [`  Active model    ${Option.match(status.activeModel, {
    onNone: () => "None",
    onSome: (model) => model.status === "Ready"
      ? model.displayName
      : `${model.displayName} - ${model.status}`,
  })}`] : []),
  "",
].join("\n")

export const runServiceStatus = () => runCommand({
  effect: publicServiceStatus,
  render: renderServiceStatus,
})
