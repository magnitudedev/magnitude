import { desktopApplication, desktopServiceOrigin, startDesktopApplication, stopDesktopApplication, readDesktopLoginStartup, setDesktopLoginStartup } from "../server/application"
import { FetchHttpClient } from "@effect/platform"
import * as FileSystem from "@effect/platform/FileSystem"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import { BunContext } from "@effect/platform-bun"
import { formatLocalModelDisplayName } from "@magnitudedev/client-common"
import type { TrayRegistration } from "@magnitudedev/sdk/desktop-host"
import { Effect, Option } from "effect"
import { existingAcnConnection } from "../server/acn-connection"
import { explainServiceStartupFailure } from "../startup/service-startup-error"
import { runCommand } from "./output"

interface ActiveModel {
  readonly displayName: string
  readonly status: "Loading" | "Ready" | "Stopping"
}

interface ServiceStatusPresentation {
  readonly status: "Stopped" | "Starting" | "Ready" | "Stopping" | "Failed" | "CleanupFailed"
  readonly address: string
  readonly version: Option.Option<string>
  readonly startsAutomaticallyOnLogin: Option.Option<boolean>
  readonly activeModel: { readonly _tag: "Unavailable" } | { readonly _tag: "Observed"; readonly model: Option.Option<ActiveModel> }
  readonly tray: Option.Option<TrayRegistration>
}

const serviceAddress = new URL(desktopServiceOrigin).host

const runServiceStartEffect = startDesktopApplication.pipe(
  Effect.tap(() => Effect.sync(() => process.stdout.write("Magnitude service is running.\n"))),
  Effect.asVoid,
)

type ServiceRequirements =
  | FileSystem.FileSystem
  | CommandExecutor.CommandExecutor
  | Path.Path
  | HttpClient.HttpClient

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

export const runServiceInstall = () => runCommand({
  effect: setDesktopLoginStartup(true),
  render: state => state._tag === "RequiresApproval"
    ? "Allow Magnitude in your system login settings to finish enabling launch at login.\n"
    : state._tag === "Enabled" ? "Magnitude will start in the background when you log in.\n"
    : "Login startup could not be enabled. Check the installed desktop app's Settings.\n",
})
export const runServiceUninstall = () => runCommand({
  effect: setDesktopLoginStartup(false).pipe(Effect.zipRight(stopDesktopApplication)),
  render: () => "Magnitude was removed from login startup and quit.\nModels and settings were kept.\n",
})
export const runServiceStart = () => run(runServiceStartEffect, explainServiceStartupFailure)
export const runServiceStop = () => runCommand({
  effect: stopDesktopApplication,
  render: () => "Magnitude service stopped.\n",
})

const readActiveModel = Effect.scoped(Effect.gen(function* () {
  const connection = yield* existingAcnConnection
  const catalog = yield* connection.client.models.getCatalog({})
  if (catalog._tag === "Initializing") return { _tag: "Unavailable" } as const
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
    return { _tag: "Observed", model: Option.some({
      displayName: formatLocalModelDisplayName(model),
      status: residency._tag === "Requested" ? "Loading" as const : residency._tag,
    }) } as const
  }
  return { _tag: "Observed", model: Option.none<ActiveModel>() } as const
}))

const publicServiceStatus = desktopApplication.observe.pipe(
  Effect.map(Option.some),
  Effect.catchTag("ApplicationControlUnavailable", () => Effect.succeed(Option.none())),
  Effect.flatMap((snapshot) => {
    const state = Option.isSome(snapshot) ? snapshot.value.service : undefined
    const activeModel = state?._tag === "Ready" ? readActiveModel.pipe(
      Effect.timeout("2 seconds"),
      Effect.orElseSucceed(() => ({ _tag: "Unavailable" } as const)),
    ) : Effect.succeed({ _tag: "Unavailable" } as const)
    return Effect.all({ model: activeModel, login: readDesktopLoginStartup.pipe(Effect.map(state => state._tag === "Enabled" ? Option.some(true) : state._tag === "Disabled" ? Option.some(false) : Option.none<boolean>()), Effect.orElseSucceed(() => Option.none<boolean>())) }, { concurrency: "unbounded" }).pipe(Effect.map(({ model, login }): ServiceStatusPresentation => ({
      status: state?._tag ?? "Stopped", address: serviceAddress,
      version: state?._tag === "Ready" ? Option.some(String(state.health.version)) : Option.none(),
      startsAutomaticallyOnLogin: login, activeModel: model,
      tray: Option.map(snapshot, value => value.tray),
    })))
  }),
)

export const renderServiceStatus = (status: ServiceStatusPresentation): string => [
  "Magnitude service",
  `  Runtime         ${status.status}`,
  `  Tray            ${Option.match(status.tray, { onNone: () => "Not running", onSome: tray => tray._tag === "Unavailable" ? `Unavailable · ${tray.message}` : tray._tag })}`,
  `  Starts at login ${Option.match(status.startsAutomaticallyOnLogin, { onNone: () => "Unavailable", onSome: value => value ? "Yes" : "No" })}`,
  ...(Option.isSome(status.version) ? [
    `  Version         ${status.version.value}`,
    `  Address         ${status.address}`,
  ] : []),
  ...(status.status === "Ready" ? [`  Active model    ${status.activeModel._tag === "Unavailable" ? "Unavailable" : Option.match(status.activeModel.model, {
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
