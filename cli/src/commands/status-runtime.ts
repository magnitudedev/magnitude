import { applicationDataDirectory, desktopApplication, desktopIsolatedProfile, desktopServiceOrigin, readDesktopLoginStartup } from "../server/application"
import { isServerSetUp } from "../server/serve-owner"
import { serverLogs, serverNotRunning, serviceFailureLine } from "../server/server-setup"
import { readServerReach, renderServerReach, type ServerReach } from "../server/server-reach"
import { formatLocalModelDisplayName, localModelResidency } from "@magnitudedev/client-common"
import type { ApplicationOwner } from "@magnitudedev/sdk/desktop-host"
import { Command, CommandExecutor, FetchHttpClient, FileSystem, HttpClient, HttpClientResponse } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import { MAC_SERVER_LABEL, MAC_SERVER_PLIST } from "@magnitudedev/daemon-management/desktop-native"
import { AcnHealthResponseSchema, type AcnHealthResponse } from "@magnitudedev/sdk"
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
  readonly elsewhere: Option.Option<Elsewhere>
}

/**
 * A service answering on this profile's port that this account can't observe or control: another
 * account's, when the system says whose it is, or otherwise just one outside this profile.
 */
export type Elsewhere = { readonly _tag: "OtherAccount"; readonly account: string } | { readonly _tag: "Unknown" }

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

/** The uid that owns the socket listening on a loopback or wildcard address at `port`, from `/proc/net/tcp` or `tcp6`. */
export const listeningUid = (table: string, port: number): Option.Option<number> => {
  const hexPort = port.toString(16).toUpperCase().padStart(4, "0")
  const row = table.split("\n").map(line => line.trim().split(/\s+/))
    .find(fields => fields[1]?.endsWith(`:${hexPort}`) && fields[3] === "0A" && fields[7] !== undefined)
  return Option.fromNullable(row?.[7]).pipe(Option.map(Number), Option.filter(Number.isInteger))
}

/** Who owns the service answering on the port, where the system says: the macOS server plist, or Linux's socket table. */
const portAccount = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const read = (path: string) => fs.readFileString(path).pipe(Effect.orElseSucceed(() => ""))
  if (process.platform === "darwin") {
    return Option.fromNullable(/<key>UserName<\/key>\s*<string>([^<]+)<\/string>/.exec(yield* read(MAC_SERVER_PLIST))?.[1])
  }
  if (process.platform !== "linux") return Option.none<string>()
  const port = Number(new URL(desktopServiceOrigin).port)
  const tables = [yield* read("/proc/net/tcp"), yield* read("/proc/net/tcp6")]
  const owner = Option.firstSomeOf(tables.map(table => listeningUid(table, port)))
  if (Option.isNone(owner)) return Option.none<string>()
  if (owner.value === process.getuid?.()) return Option.some(userInfo().username)
  const executor = yield* CommandExecutor.CommandExecutor
  const name = (yield* executor.string(Command.make("/usr/bin/id", "-un", String(owner.value))).pipe(Effect.orElseSucceed(() => ""))).trim()
  return Option.some(name || `uid ${owner.value}`)
})

/**
 * The service answering on this computer's own port when this account has no application of its
 * own: another account's app, `serve` or server. Isolated profiles use their own ports, so they
 * never mistake another Magnitude on the default port for theirs. One this account runs isn't elsewhere.
 */
const readElsewhere: Effect.Effect<Option.Option<{ readonly health: AcnHealthResponse; readonly elsewhere: Elsewhere }>> = desktopIsolatedProfile ? Effect.succeed(Option.none()) : Effect.gen(function* () {
  const response = yield* HttpClient.get(`${desktopServiceOrigin}/health`)
  const health = yield* HttpClientResponse.schemaBodyJson(AcnHealthResponseSchema)(response)
  const account = yield* portAccount
  if (Option.contains(account, userInfo().username)) return Option.none<{ readonly health: AcnHealthResponse; readonly elsewhere: Elsewhere }>()
  const elsewhere: Elsewhere = Option.match(account, { onNone: () => ({ _tag: "Unknown" }), onSome: account => ({ _tag: "OtherAccount", account }) })
  return Option.some({ health, elsewhere })
}).pipe(Effect.scoped, Effect.timeout("2 seconds"), Effect.orElseSucceed(() => Option.none()), Effect.provide([FetchHttpClient.layer, BunContext.layer]))

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
    const residency = Option.getOrUndefined(localModelResidency(model))
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
  : readElsewhere.pipe(Effect.map(Option.match({
    onNone: () => status,
    onSome: ({ health, elsewhere }): ServiceStatusPresentation => ({
      ...status, status: health.state._tag, version: Option.some(String(health.version)), failing: Option.none(),
      elsewhere: Option.some(elsewhere),
    }),
  })))))

/** Where the system service is set up, a Headless owner is that service rather than someone's `serve`. */
const ownerName = (status: ServiceStatusPresentation) => Option.match(status.owner, {
  onNone: () => Option.match(status.elsewhere, { onNone: () => "None",
    onSome: other => other._tag === "OtherAccount" ? `Another account (${other.account})` : "Not this profile" }), onSome: owner => owner._tag === "Headless" && status.server ? "Server" : owner._tag })

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
  ...Option.match(status.elsewhere, { onNone: () => [], onSome: other => other._tag === "OtherAccount" ? ["",
    `Magnitude is running on this computer for another account (${other.account}). This account can't see or control it;`,
    status.server ? "that account can stop it, or remove the server with `magnitude server remove`." : "that account can quit it."]
    : ["", "Another Magnitude is answering on this computer's port, outside this profile. This account can't see or control it."] }),
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
