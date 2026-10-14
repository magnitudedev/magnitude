import { applicationDataDirectory, desktopApplication, desktopServiceOrigin, readDesktopLoginStartup } from "../server/application"
import { isServerSetUp } from "../server/serve-owner"
import { serverLogs, serverNotRunning, serviceFailureLine } from "../server/server-setup"
import { readServerReach, renderServerReach, type ServerReach } from "../server/server-reach"
import { formatLocalModelDisplayName } from "@magnitudedev/client-common"
import type { ApplicationOwner } from "@magnitudedev/sdk/desktop-host"
import { Command, CommandExecutor, FetchHttpClient, FileSystem, HttpClient, HttpClientResponse } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import { MAC_SERVER_LABEL, MAC_SERVER_PLIST } from "@magnitudedev/daemon-management/desktop-native"
import { AcnHealthResponseSchema } from "@magnitudedev/sdk"
import { isIP } from "node:net"
import { userInfo } from "node:os"
import { existingAcnConnection } from "../server/acn-connection"
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
  readonly owner: Option.Option<ApplicationOwner>
  readonly reach: Option.Option<ServerReach>
  readonly server: boolean
  readonly failing: Option.Option<ServerFailing>
  readonly loaded: Option.Option<boolean>
  readonly elsewhere: Option.Option<OtherAccount>
}

/** A service this account can't observe or control, answering on the port for another account. */
interface OtherAccount {
  readonly account: Option.Option<string>
}

/** A system service that systemd keeps restarting, with the service's own last error when the journal is readable. */
interface ServerFailing {
  readonly restarts: number
  readonly lastError: Option.Option<string>
}

const readServerFailing = Effect.gen(function* () {
  const executor = yield* CommandExecutor.CommandExecutor
  const output = (command: string, ...args: string[]) => executor.string(Command.make(command, ...args)).pipe(
    Effect.timeout("3 seconds"), Effect.orElseSucceed(() => ""))
  const unit = new Map((yield* output("/usr/bin/systemctl", "show", "magnitude.service", "--property=NRestarts,Result,ActiveState"))
    .split("\n").map(line => line.split("=", 2) as [string, string | undefined]))
  const restarts = Number(unit.get("NRestarts"))
  // An administrator's stop leaves the restart count but a successful result.
  const failed = (unit.get("Result") ?? "success") !== "success" || (restarts > 0 && unit.get("ActiveState") !== "inactive")
  if (!Number.isInteger(restarts) || !failed) return Option.none<ServerFailing>()
  const journal = yield* output("/usr/bin/journalctl", "-u", "magnitude.service", "-n", "300", "-o", "cat", "--no-pager", "--quiet")
  return Option.some({ restarts, lastError: serviceFailureLine(journal.split("\n")) })
}).pipe(Effect.provide(BunContext.layer))

/**
 * The service answering on this computer's port when this account has no application of its own:
 * another account's app, `serve` or server. On macOS the server's plist names the account it runs as.
 */
const readOtherAccount = Effect.gen(function* () {
  const response = yield* HttpClient.get(`${desktopServiceOrigin}/health`)
  const health = yield* HttpClientResponse.schemaBodyJson(AcnHealthResponseSchema)(response)
  const plist = process.platform === "darwin"
    ? yield* Effect.flatMap(FileSystem.FileSystem, fs => fs.readFileString(MAC_SERVER_PLIST)).pipe(Effect.orElseSucceed(() => ""))
    : ""
  const account = Option.fromNullable(/<key>UserName<\/key>\s*<string>([^<]+)<\/string>/.exec(plist)?.[1])
  return { health, account }
}).pipe(Effect.scoped, Effect.timeout("2 seconds"), Effect.option, Effect.provide([FetchHttpClient.layer, BunContext.layer]))

/** Whether launchd has the macOS server daemon loaded, which decides how it is started again. */
const readMacServerLoaded = Effect.flatMap(CommandExecutor.CommandExecutor, executor =>
  executor.exitCode(Command.make("/bin/launchctl", "print", `system/${MAC_SERVER_LABEL}`).pipe(Command.stdout("pipe"), Command.stderr("pipe")))).pipe(
  Effect.timeout("3 seconds"), Effect.map(code => Option.some(code === 0)), Effect.orElseSucceed(() => Option.none<boolean>()),
  Effect.provide(BunContext.layer))

const serviceAddress = new URL(desktopServiceOrigin).host

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

const ownStatus = desktopApplication.observe.pipe(
  Effect.map(Option.some),
  Effect.catchTag("ApplicationControlUnavailable", () => Effect.succeed(Option.none())),
  Effect.flatMap((snapshot) => {
    const state = Option.isSome(snapshot) ? snapshot.value.service : undefined
    const activeModel = state?._tag === "Ready" ? readActiveModel.pipe(
      Effect.timeout("2 seconds"),
      Effect.orElseSucceed(() => ({ _tag: "Unavailable" } as const)),
    ) : Effect.succeed({ _tag: "Unavailable" } as const)
    const reach = state?._tag === "Ready" ? Effect.all([applicationDataDirectory, isServerSetUp]).pipe(
      Effect.flatMap(([dataDirectory, service]) => readServerReach({ endpoint: desktopServiceOrigin, dataDirectory, service })), Effect.option)
      : Effect.succeed(Option.none<ServerReach>())
    const failing = (server: boolean) => server && process.platform === "linux" && state?._tag !== "Ready" ? readServerFailing : Effect.succeed(Option.none<ServerFailing>())
    const loaded = (server: boolean) => server && process.platform === "darwin" && Option.isNone(snapshot) ? readMacServerLoaded : Effect.succeed(Option.none<boolean>())
    return Effect.all({ model: activeModel, reach, server: isServerSetUp.pipe(Effect.flatMap(server => Effect.all({ server: Effect.succeed(server), failing: failing(server), loaded: loaded(server) }))), login: Option.isSome(snapshot) && snapshot.value.owner._tag === "Desktop" ? readDesktopLoginStartup.pipe(Effect.map(state => state._tag === "Enabled" ? Option.some(true) : state._tag === "Disabled" ? Option.some(false) : Option.none<boolean>()), Effect.orElseSucceed(() => Option.none<boolean>())) : Effect.succeed(Option.none<boolean>()) }, { concurrency: "unbounded" }).pipe(Effect.map(({ model, login, reach, server: { server, failing, loaded } }): ServiceStatusPresentation => ({
      status: state?._tag ?? "Stopped", address: serviceAddress,
      version: state?._tag === "Ready" ? Option.some(String(state.health.version)) : Option.none(),
      startsAutomaticallyOnLogin: login, activeModel: model,
      owner: Option.map(snapshot, value => value.owner), reach, server, failing, loaded, elsewhere: Option.none(),
    })))
  }),
)

const publicServiceStatus = ownStatus.pipe(Effect.flatMap(status => Option.isSome(status.owner) ? Effect.succeed(status)
  : readOtherAccount.pipe(Effect.map(Option.match({
    onNone: () => status,
    onSome: ({ health, account }): ServiceStatusPresentation => ({
      ...status, status: health.state._tag, version: Option.some(String(health.version)), failing: Option.none(),
      elsewhere: Option.some({ account: Option.filter(account, name => name !== userInfo().username) }),
    }),
  })))))

/** Where the system service is set up, a Headless owner is that service rather than someone's `serve`. */
const ownerName = (status: ServiceStatusPresentation) => Option.match(status.owner, {
  onNone: () => Option.match(status.elsewhere, { onNone: () => "None", onSome: other => Option.match(other.account, {
    onNone: () => "Another account", onSome: account => `Another account (${account})` }) }), onSome: owner => owner._tag === "Headless" && status.server ? "Server" : owner._tag })

/** With network access active the service also listens on all interfaces or its configured address, as ACN resolves it. */
const listeningAddress = (status: ServiceStatusPresentation) => {
  const network = Option.getOrUndefined(Option.flatMap(status.reach, reach => reach.network))
  if (!network?.enabled || network.pending) return status.address
  const configured = Option.getOrUndefined(network.bind)?.trim()
  if (configured === undefined || isIP(configured) === 0 || configured === "0.0.0.0") return `0.0.0.0:${network.port} (all interfaces)`
  return `${status.address}, ${isIP(configured) === 6 ? `[${configured}]` : configured}:${network.port}`
}

export const renderStatus =(status: ServiceStatusPresentation, platform: NodeJS.Platform = process.platform): string => [
  "Magnitude service",
  `  Runtime         ${status.status}`,
  `  Owner           ${ownerName(status)}`,
  ...(Option.isSome(status.owner) && status.owner.value._tag === "Desktop" ? [
    `  Tray            ${status.owner.value.tray._tag === "Unavailable" ? `Unavailable · ${status.owner.value.tray.message}` : status.owner.value.tray._tag}`,
    `  Starts at login ${Option.match(status.startsAutomaticallyOnLogin, { onNone: () => "Unavailable", onSome: value => value ? "Yes" : "No" })}`,
  ] : []),
  ...(Option.isSome(status.version) ? [
    `  Version         ${status.version.value}`,
    `  Address         ${listeningAddress(status)}`,
  ] : []),
  ...(status.status === "Ready" ? [`  Active model    ${status.activeModel._tag === "Unavailable" ? "Unavailable" : Option.match(status.activeModel.model, {
    onNone: () => "None",
    onSome: (model) => model.status === "Ready"
      ? model.displayName
      : `${model.displayName} - ${model.status}`,
  })}`] : []),
  ...Option.match(status.reach, { onNone: () => [], onSome: reach => ["", ...renderServerReach(reach).map(line => `  ${line}`)] }),
  ...Option.match(status.elsewhere, { onNone: () => [], onSome: () => ["",
    "Magnitude is running on this computer for another account. This account can't see or control it;",
    status.server ? "the account that runs it can stop it, or remove the server with `magnitude server remove`." : "the account that runs it can quit it."] }),
  ...Option.match(status.failing, {
    onSome: failing => ["", "Keeps failing to start",
      `The Magnitude server keeps stopping as it starts (restarted ${failing.restarts} ${failing.restarts === 1 ? "time" : "times"}).`,
      ...Option.match(failing.lastError, { onNone: () => [], onSome: line => [`Its last error: ${line}`] }),
      `See its log: ${serverLogs(platform)}`],
    onNone: () => Option.isNone(status.owner) && Option.isNone(status.elsewhere) ? [
      "", "Not running", status.server ? serverNotRunning(platform, status.loaded) : "Open the Magnitude desktop app or run `magnitude serve`.",
    ] : [],
  }),
  "",
].join("\n")

export const runStatus = () => runCommand({
  effect: publicServiceStatus,
  render: status => renderStatus(status),
})
