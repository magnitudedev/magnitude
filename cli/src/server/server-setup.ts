import { Context, Duration, Effect, Option, Schema } from "effect"
import { FOREIGN_SERVER_ACCOUNT_MESSAGE, MAC_SERVER_LABEL, MAC_SERVER_PLIST, SERVER_DATA_DIRECTORY, SERVER_PORT } from "@magnitudedev/daemon-management/desktop-native"

export const WINDOWS_SERVER_MESSAGE = "Server mode isn't available on Windows. To run Magnitude as a service, see https://docs.magnitude.dev/remote-server#windows"

export class ServerSetupFailed extends Schema.TaggedError<ServerSetupFailed>()("ServerSetupFailed", { message: Schema.String }) {}
const failed = (message: string) => new ServerSetupFailed({ message })

/** How the people and agents on other computers reach the server once network access is on. */
export interface ServerAccess {
  readonly addresses: readonly string[]
  readonly port: number
  readonly key: string
}

/**
 * Everything `server setup` and `server remove` touch, so the steps can be exercised without a machine.
 * `isSetUp` means the service is registered (the Linux marker or the macOS LaunchDaemon); `hasRemnants`
 * means anything a complete or partial setup leaves, which `server remove` cleans up.
 */
export interface ServerSetupHost {
  readonly platform: NodeJS.Platform
  readonly user: string
  readonly isRoot: boolean
  /** Running under sudo rather than from a root login. */
  readonly underSudo: boolean
  /** systemd on Linux, launchd on macOS. */
  readonly hasServiceManager: Effect.Effect<boolean>
  /** The owner of the person's own profile, if the desktop app or `magnitude serve` is running. */
  readonly personalOwner: Effect.Effect<Option.Option<"Desktop" | "Headless">>
  readonly isSetUp: Effect.Effect<boolean>
  readonly hasRemnants: Effect.Effect<boolean>
  readonly hasForeignAccount: Effect.Effect<boolean>
  readonly hasTerminal: Effect.Effect<boolean>
  /** Whether sudo can run without asking: a cached password or a NOPASSWD rule. */
  readonly sudoWithoutPrompt: Effect.Effect<boolean>
  readonly confirm: (question: string) => Effect.Effect<boolean, ServerSetupFailed>
  /** Runs the installed CLI's hidden root command, through sudo unless this already is root, prompting at most once. */
  readonly runRootStep: (args: readonly string[]) => Effect.Effect<void, ServerSetupFailed>
  /**
   * Who answers on the server's port: nobody, the registered service (the instance its owner reports
   * is the one answering), or anything else, such as another account's Magnitude.
   */
  readonly servicePort: Effect.Effect<ServicePort>
  /** How many times the service manager has started the service, where it says. */
  readonly serviceRuns: Effect.Effect<Option.Option<number>>
  /** The service log's recent lines, since setup started when the log can say. */
  readonly serviceLog: Effect.Effect<readonly string[]>
  /** Turns network access on with a key once the service answers, and restarts it if that is needed. */
  readonly enableNetworkAccess: Effect.Effect<ServerAccess, ServerSetupFailed>
  readonly turnOffDesktopLoginStartup: Effect.Effect<boolean>
  readonly write: (text: string) => Effect.Effect<void>
}
export type ServicePort = "Free" | "Service" | "Other"
export const ServerSetupHost = Context.GenericTag<ServerSetupHost>("@magnitudedev/cli/ServerSetupHost")

/** Where the service's output goes: the journal on Linux, the person's log file on macOS. */
export const serverLogs = (platform: NodeJS.Platform) => platform === "darwin" ? "~/.magnitude/logs/server.log" : "journalctl -u magnitude"

const macStart = (loaded: Option.Option<boolean>) => Option.match(loaded, {
  onSome: loaded => loaded ? `\`sudo launchctl kickstart system/${MAC_SERVER_LABEL}\`` : `\`sudo launchctl bootstrap system ${MAC_SERVER_PLIST}\``,
  onNone: () => `\`sudo launchctl bootstrap system ${MAC_SERVER_PLIST}\` (or, if launchd says it's already loaded, \`sudo launchctl kickstart system/${MAC_SERVER_LABEL}\`)`,
})

/**
 * What to do when the registered service isn't answering; a server has no desktop app and refuses `serve`.
 * launchd only kickstarts a loaded daemon, and one booted out must be bootstrapped from its plist again.
 */
export const serverNotRunning = (platform: NodeJS.Platform, loaded: Option.Option<boolean> = Option.none()) => platform === "darwin"
  ? `The Magnitude server isn't running. Start it with ${macStart(loaded)}, and see its log: ${serverLogs(platform)}`
  : `The Magnitude server isn't running. Start it with \`sudo systemctl start magnitude\`, and see its log: ${serverLogs(platform)}`

export const SERVER_START_TIMEOUT = Duration.minutes(2)
/** Restarts by the service manager during the wait that mean the service fails to start rather than is slow. */
const FAILED_STARTS = 3
/** How long another program may answer on the port after setup started the service. */
const FOREIGN_PORT_GRACE = 15

export const portTaken = `Port ${SERVER_PORT} is already in use, by Magnitude running for another account on this computer or by another program, `
  + "so the server can't use it. Stop that, then run `magnitude server setup` again."

/** The service manager's own lines about the unit, which say that it stopped but never why. */
const SERVICE_MANAGER_LINE = /^(magnitude\.service: |(Started|Starting|Stopped|Stopping) magnitude\.service\b)/
/** A logger entry, or its continuation: what the service did, rather than its final word. */
const LOG_ENTRY = /^(\[\d[\d:.]*\] [A-Z]+ \(#\d+\):|timestamp=|\s|")/
const ERROR_ENTRY = /^\[\d[\d:.]*\] (?:ERROR|FATAL) \(#\d+\):\s*(\S.*)$/

const serviceLines = (lines: readonly string[]) => lines
  .map(line => line.trimEnd())
  .filter(line => line.trim() !== "" && !SERVICE_MANAGER_LINE.test(line) && !/^\s*at\s/.test(line) && !line.includes("/$bunfs/"))
const clip = (line: string) => line.length > 240 ? `${line.slice(0, 239)}…` : line

/** The service's own account of why it stopped: its final plain line, or else its last logged error. */
export const serviceFailureLine = (lines: readonly string[]): Option.Option<string> => {
  const service = serviceLines(lines)
  return Option.orElse(Option.fromNullable(service.findLast(line => !LOG_ENTRY.test(line))),
    () => Option.fromNullable(service.map(line => ERROR_ENTRY.exec(line)?.[1]).findLast(message => message !== undefined))).pipe(Option.map(clip))
}

/**
 * The lines of a service log that say what went wrong: the service's own account of why it stopped
 * or, without one, the tail of its log, never the service manager's lines, stack frames, blank lines or repeats.
 */
export const explainServiceLog = (lines: readonly string[]) => Option.match(serviceFailureLine(lines), {
  onSome: line => [line],
  onNone: () => serviceLines(lines).filter((line, index, all) => all.lastIndexOf(line) === index).slice(-6).map(clip),
})

const startFailed = (host: ServerSetupHost, problem: string) => host.serviceLog.pipe(Effect.flatMap(log => {
  const lines = explainServiceLog(log)
  return Effect.fail(failed([
    problem,
    ...(lines.length > 0 ? ["Its log ends with:", ...lines.map(line => `  ${line}`)] : []),
    `See its log: ${serverLogs(host.platform)}`,
    "Once it's fixed, run `magnitude server setup` again to finish, or `magnitude server remove` to undo setup.",
  ].join("\n")))
}))

/**
 * Waits until the service setup started answers on the server's port. A service the service manager
 * keeps restarting, or another program holding the port, fails at once with the service's own account
 * of what went wrong, rather than after the full start timeout.
 */
const awaitService = (host: ServerSetupHost) => Effect.gen(function* () {
  const initialRuns = yield* host.serviceRuns
  for (let foreign = 0; ; yield* Effect.sleep(Duration.seconds(1))) {
    const port = yield* host.servicePort
    if (port === "Service") return
    foreign = port === "Other" ? foreign + 1 : 0
    if (foreign >= FOREIGN_PORT_GRACE) return yield* startFailed(host, portTaken)
    const runs = yield* host.serviceRuns
    if (Option.isSome(initialRuns) && Option.isSome(runs) && runs.value - initialRuns.value >= FAILED_STARTS) {
      return yield* startFailed(host, "The Magnitude server keeps stopping as it starts.")
    }
  }
})

/**
 * Bounds the wait for the service to answer, even when a request in flight cannot be interrupted:
 * a server that never becomes ready fails setup with what its log says, and setup can be run again.
 */
export const withinServerStart = <A>(host: ServerSetupHost, wait: Effect.Effect<A, ServerSetupFailed>) => awaitService(host).pipe(
  Effect.zipRight(wait),
  Effect.disconnect,
  Effect.timeoutOption(SERVER_START_TIMEOUT),
  Effect.flatMap(Option.match({
    onNone: () => startFailed(host, "The Magnitude server did not become ready within 2 minutes."),
    onSome: Effect.succeed,
  })))

export const renderServerReady = (platform: NodeJS.Platform, access: ServerAccess) => [
  "Magnitude is running as a server on this machine.",
  "",
  "Open it in a browser on another computer:",
  ...(access.addresses.length > 0 ? access.addresses : ["<this machine>"]).map(address => `  http://${address}:${access.port}`),
  "",
  `Network access key: ${access.key}`,
  "  The browser app asks for it once. Save it; later, find it in Settings → Network access",
  "  in the browser app on this machine.",
  "",
  "Apps and agents use the same address with /inference/v1.",
  "",
  `Status:  magnitude status`,
  `Logs:    ${serverLogs(platform)}`,
  ...(platform === "linux" ? ["Stop:    sudo systemctl stop magnitude", "Remove:  magnitude server remove"] : ["Stop:    magnitude server remove"]),
  "",
].join("\n")

/** A Mac's desktop app opens at login by default, where it could only offer to stop the server. */
export const renderDesktopLoginStartup = (turnedOff: boolean) => turnedOff
  ? "The Magnitude app no longer opens when you log in; it can't run while the server does.\n"
    + "After `magnitude server remove`, you can turn Launch at login back on in its Settings.\n"
  : "If the Magnitude app opens when you log in, it can only offer to stop the server.\n"
    + "Turn that off in System Settings → General → Login Items & Extensions.\n"

/** A registered service that is still starting can answer on the port before its owner reports it, so it gets a few seconds. */
const settledPort = (host: ServerSetupHost) => Effect.gen(function* () {
  const registered = yield* host.isSetUp
  for (let attempt = 1; ; attempt++) {
    const port = yield* host.servicePort
    if (port !== "Other" || !registered || attempt >= 5) return port
    yield* Effect.sleep(Duration.seconds(1))
  }
})

/** Does everything it can as the person, and only the root step through sudo. */
export const serverSetup = Effect.gen(function* () {
  const host = yield* ServerSetupHost
  if (host.platform === "win32") return yield* failed(WINDOWS_SERVER_MESSAGE)
  if (host.platform !== "linux" && host.platform !== "darwin") return yield* failed("Server mode is available on Linux and macOS.")
  if (host.isRoot && (host.underSudo || host.platform !== "linux")) return yield* failed("Run `magnitude server setup` as yourself, without sudo. It asks for your password when it needs it.")
  if (!(yield* host.hasServiceManager)) return yield* failed("Server setup needs systemd, which this machine doesn't run. Run `magnitude serve` instead to serve until you stop it.")
  // Readable without sudo, so the refusal is the whole answer rather than a failed root step.
  if (yield* host.hasForeignAccount) return yield* failed(FOREIGN_SERVER_ACCOUNT_MESSAGE)
  const owner = yield* host.personalOwner
  // On macOS the service serves in the person's own profile, so after a setup that did not finish it is the Headless owner.
  const service = Option.isSome(owner) && owner.value === "Headless" && (yield* host.isSetUp)
  if (Option.isSome(owner) && !service) return yield* failed(owner.value === "Desktop"
    ? "Quit the Magnitude desktop app first. The app and the server can't run at the same time."
    : "Stop `magnitude serve` first. It and the server can't run at the same time.")
  // Whatever answers on the port receives setup's requests: another account's Magnitude would be configured and its key printed.
  if ((yield* settledPort(host)) === "Other") return yield* failed(`${portTaken} Nothing was changed.`)
  if (!host.isRoot && !(yield* host.sudoWithoutPrompt) && !(yield* host.hasTerminal)) return yield* failed("Server setup needs an administrator account (one that can use sudo) and its password; run it in a terminal.")
  // Through sudo the root step names the person who ran it, whom the macOS server serves as; a root login names no one.
  yield* host.runRootStep(host.isRoot ? ["_server-install"] : ["_server-install", host.user])
  yield* host.write("Waiting for the Magnitude server to start…\n")
  const access = yield* withinServerStart(host, host.enableNetworkAccess)
  yield* host.write(renderServerReady(host.platform, access))
  if (host.platform === "darwin") yield* host.write(renderDesktopLoginStartup(yield* host.turnOffDesktopLoginStartup))
})

export const serverRemove = Effect.gen(function* () {
  const host = yield* ServerSetupHost
  if (host.platform === "win32") return yield* failed(WINDOWS_SERVER_MESSAGE)
  if (host.platform !== "linux" && host.platform !== "darwin") return yield* failed("Server mode is available on Linux and macOS.")
  if (host.isRoot && (host.underSudo || host.platform !== "linux")) return yield* failed("Run `magnitude server remove` as yourself, without sudo. It asks for your password when it needs it.")
  if (!(yield* host.hasRemnants)) return yield* host.write("Magnitude isn't set up as a server on this machine.\n")
  if (!(yield* host.hasTerminal)) return yield* failed("Run `magnitude server remove` in a terminal to confirm it.")
  const kept = host.platform === "linux" ? SERVER_DATA_DIRECTORY : "~/.magnitude"
  const confirmed = yield* host.confirm(`Stop the Magnitude server and remove its service? Browsers and agents connected to it will be disconnected. Its data stays in ${kept}. [y/N] `)
  if (!confirmed) return yield* host.write("Nothing was changed.\n")
  yield* host.runRootStep(["_server-remove"])
  yield* host.write(host.platform === "linux"
    ? `Removed the Magnitude server. Its data is still in ${kept}, now owned by root.\n`
    : `Removed the Magnitude server. Its data is still in ${kept}.\n`)
})
