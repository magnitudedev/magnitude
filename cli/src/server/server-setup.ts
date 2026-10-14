import { Context, Duration, Effect, Option, Schema } from "effect"
import { MAC_SERVER_LABEL, SERVER_DATA_DIRECTORY } from "@magnitudedev/daemon-management/desktop-native"

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
  readonly hasTerminal: Effect.Effect<boolean>
  /** Whether sudo can run without asking: a cached password or a NOPASSWD rule. */
  readonly sudoWithoutPrompt: Effect.Effect<boolean>
  readonly confirm: (question: string) => Effect.Effect<boolean, ServerSetupFailed>
  /** Runs the installed CLI's hidden root command, through sudo unless this already is root, prompting at most once. */
  readonly runRootStep: (args: readonly string[]) => Effect.Effect<void, ServerSetupFailed>
  /** Waits for the service, turns network access on with a key, and restarts it if that is needed. */
  readonly enableNetworkAccess: Effect.Effect<ServerAccess, ServerSetupFailed>
  readonly write: (text: string) => Effect.Effect<void>
}
export const ServerSetupHost = Context.GenericTag<ServerSetupHost>("@magnitudedev/cli/ServerSetupHost")

/** Where the service's output goes: the journal on Linux, the person's log file on macOS. */
export const serverLogs = (platform: NodeJS.Platform) => platform === "darwin" ? "~/.magnitude/logs/server.log" : "journalctl -u magnitude"

/** What to do when the registered service isn't answering; a server has no desktop app and refuses `serve`. */
export const serverNotRunning = (platform: NodeJS.Platform) => platform === "darwin"
  ? `The Magnitude server isn't running. Start it with \`sudo launchctl kickstart system/${MAC_SERVER_LABEL}\`, and see its log: ${serverLogs(platform)}`
  : `The Magnitude server isn't running. Start it with \`sudo systemctl start magnitude\`, and see its log: ${serverLogs(platform)}`

export const SERVER_START_TIMEOUT = Duration.minutes(2)

/**
 * Bounds the wait for the service to answer, even when a request in flight cannot be interrupted:
 * a server that never becomes ready fails setup with where to look, and setup can be run again.
 */
export const withinServerStart = <A>(platform: NodeJS.Platform, wait: Effect.Effect<A, ServerSetupFailed>) => wait.pipe(
  Effect.disconnect,
  Effect.timeoutFail({ duration: SERVER_START_TIMEOUT, onTimeout: () => failed([
    `The Magnitude server did not become ready within 2 minutes. See its log: ${serverLogs(platform)}`,
    "Once it's fixed, run `magnitude server setup` again to finish, or `magnitude server remove` to undo setup.",
  ].join("\n")) }))

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
  `Stop:    magnitude server remove`,
  "",
].join("\n")

/** Does everything it can as the person, and only the root step through sudo. */
export const serverSetup = Effect.gen(function* () {
  const host = yield* ServerSetupHost
  if (host.platform === "win32") return yield* host.write(`${WINDOWS_SERVER_MESSAGE}\n`)
  if (host.platform !== "linux" && host.platform !== "darwin") return yield* failed("Server mode is available on Linux and macOS.")
  if (host.isRoot && (host.underSudo || host.platform !== "linux")) return yield* failed("Run `magnitude server setup` as yourself, without sudo. It asks for your password when it needs it.")
  if (!(yield* host.hasServiceManager)) return yield* failed("Server setup needs systemd, which this machine doesn't run. Run `magnitude serve` instead to serve until you stop it.")
  const owner = yield* host.personalOwner
  // On macOS the service serves in the person's own profile, so after a setup that did not finish it is the Headless owner.
  const service = Option.isSome(owner) && owner.value === "Headless" && (yield* host.isSetUp)
  if (Option.isSome(owner) && !service) return yield* failed(owner.value === "Desktop"
    ? "Quit the Magnitude desktop app first. The app and the server can't run at the same time."
    : "Stop `magnitude serve` first. It and the server can't run at the same time.")
  if (!host.isRoot && !(yield* host.sudoWithoutPrompt) && !(yield* host.hasTerminal)) return yield* failed("Server setup needs your password; run it in a terminal.")
  // Through sudo the root step names the person who ran it, whom the macOS server serves as; a root login names no one.
  yield* host.runRootStep(host.isRoot ? ["_server-install"] : ["_server-install", host.user])
  yield* host.write("Waiting for the Magnitude server to start…\n")
  const access = yield* withinServerStart(host.platform, host.enableNetworkAccess)
  yield* host.write(renderServerReady(host.platform, access))
})

export const serverRemove = Effect.gen(function* () {
  const host = yield* ServerSetupHost
  if (host.platform === "win32") return yield* host.write(`${WINDOWS_SERVER_MESSAGE}\n`)
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
