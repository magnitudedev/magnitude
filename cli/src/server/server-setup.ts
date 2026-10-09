import { Context, Effect, Option, Schema } from "effect"
import { SERVER_DATA_DIRECTORY } from "@magnitudedev/daemon-management/desktop-native"

export const WINDOWS_SERVER_MESSAGE = "Server mode isn't available on Windows. To run Magnitude as a service, see https://docs.magnitude.dev/remote-server#windows"

export class ServerSetupFailed extends Schema.TaggedError<ServerSetupFailed>()("ServerSetupFailed", { message: Schema.String }) {}
const failed = (message: string) => new ServerSetupFailed({ message })

/** How the people and agents on other computers reach the server once network access is on. */
export interface ServerAccess {
  readonly addresses: readonly string[]
  readonly port: number
  readonly key: string
}

/** Everything `server setup` and `server remove` touch, so the steps can be exercised without a machine. */
export interface ServerSetupHost {
  readonly platform: NodeJS.Platform
  readonly user: string
  readonly isRoot: boolean
  /** systemd on Linux, launchd on macOS. */
  readonly hasServiceManager: Effect.Effect<boolean>
  /** The owner of the person's own profile, if the desktop app or `magnitude serve` is running. */
  readonly personalOwner: Effect.Effect<Option.Option<"Desktop" | "Headless">>
  readonly isSetUp: Effect.Effect<boolean>
  readonly hasTerminal: Effect.Effect<boolean>
  /** Whether sudo can run without asking: a cached password or a NOPASSWD rule. */
  readonly sudoWithoutPrompt: Effect.Effect<boolean>
  readonly confirm: (question: string) => Effect.Effect<boolean, ServerSetupFailed>
  /** Runs the installed CLI's hidden root command through sudo, prompting at most once. */
  readonly runRootStep: (args: readonly string[]) => Effect.Effect<void, ServerSetupFailed>
  /** Waits for the service, turns network access on with a key, and restarts it if that is needed. */
  readonly enableNetworkAccess: Effect.Effect<ServerAccess, ServerSetupFailed>
  readonly write: (text: string) => Effect.Effect<void>
}
export const ServerSetupHost = Context.GenericTag<ServerSetupHost>("@magnitudedev/cli/ServerSetupHost")

const logsCommand = (platform: NodeJS.Platform) => platform === "darwin" ? "~/.magnitude/logs/service.log" : "journalctl -u magnitude"

export const renderServerReady = (platform: NodeJS.Platform, access: ServerAccess) => [
  "Magnitude is running as a server on this machine.",
  "",
  "Open it in a browser on another computer:",
  ...(access.addresses.length > 0 ? access.addresses : ["<this machine>"]).map(address => `  http://${address}:${access.port}`),
  "",
  `Network access key: ${access.key}`,
  "  The browser app asks for it once. This is the only time it is printed; later, find it in",
  "  Settings → Network access in the browser app on this machine.",
  "",
  "Apps and agents use the same address with /inference/v1.",
  "",
  `Status:  magnitude status`,
  `Logs:    ${logsCommand(platform)}`,
  `Stop:    magnitude server remove`,
  ...(platform === "linux" ? ["", "You were added to the magnitude group. Log out and back in before running `magnitude status` or `magnitude update`."] : []),
  "",
].join("\n")

/** Does everything it can as the person, and only the root step through sudo. */
export const serverSetup = Effect.gen(function* () {
  const host = yield* ServerSetupHost
  if (host.platform === "win32") return yield* host.write(`${WINDOWS_SERVER_MESSAGE}\n`)
  if (host.platform !== "linux" && host.platform !== "darwin") return yield* failed("Server mode is available on Linux and macOS.")
  if (host.isRoot) return yield* failed("Run `magnitude server setup` as yourself, without sudo. It asks for your password when it needs it.")
  if (!(yield* host.hasServiceManager)) return yield* failed("Server setup needs systemd, which this machine doesn't run. Run `magnitude serve` instead to serve until you stop it.")
  const owner = yield* host.personalOwner
  if (Option.isSome(owner)) return yield* failed(owner.value === "Desktop"
    ? "Quit the Magnitude desktop app first. The app and the server can't run at the same time."
    : "Stop `magnitude serve` first. It and the server can't run at the same time.")
  if (!(yield* host.sudoWithoutPrompt) && !(yield* host.hasTerminal)) return yield* failed("Server setup needs your password; run it in a terminal.")
  yield* host.runRootStep(["_server-install", host.user])
  const access = yield* host.enableNetworkAccess
  yield* host.write(renderServerReady(host.platform, access))
})

export const serverRemove = Effect.gen(function* () {
  const host = yield* ServerSetupHost
  if (host.platform === "win32") return yield* host.write(`${WINDOWS_SERVER_MESSAGE}\n`)
  if (host.platform !== "linux" && host.platform !== "darwin") return yield* failed("Server mode is available on Linux and macOS.")
  if (host.isRoot) return yield* failed("Run `magnitude server remove` as yourself, without sudo. It asks for your password when it needs it.")
  if (!(yield* host.isSetUp)) return yield* host.write("Magnitude isn't set up as a server on this machine.\n")
  if (!(yield* host.hasTerminal)) return yield* failed("Run `magnitude server remove` in a terminal to confirm it.")
  const kept = host.platform === "linux" ? SERVER_DATA_DIRECTORY : "~/.magnitude"
  const confirmed = yield* host.confirm(`Stop the Magnitude server and remove its service? Browsers and agents connected to it will be disconnected. Its data stays in ${kept}. [y/N] `)
  if (!confirmed) return yield* host.write("Nothing was changed.\n")
  yield* host.runRootStep(["_server-remove"])
  yield* host.write(`Removed the Magnitude server. Its data is still in ${kept}.\n`)
})
