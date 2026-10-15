import { Command, CommandExecutor, FetchHttpClient, FileSystem, HttpClient, HttpClientResponse } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Duration, Effect, Layer, Option, Schedule, Schema } from "effect"
import { open } from "node:fs/promises"
import { connect } from "node:net"
import { join } from "node:path"
import { userInfo } from "node:os"
import { MagnitudeClient } from "@magnitudedev/sdk"
import { applicationStateDirectory, hasForeignServerAccount, hasLinuxServerRemnants, installedServerCli, MAC_SERVER_LABEL, requestApplication, TerminalCommand,
  nodeTerminalCommand } from "@magnitudedev/daemon-management/desktop-native"
import { isServerSetUp } from "./serve-owner"
import { desktopApplication, desktopDataDirectory, desktopServiceOrigin } from "./application"
import { ServerSetupFailed, ServerSetupHost, type ServerAccess, type ServicePort } from "./server-setup"
import { withLocalService } from "./server-reach"

const failed = (message: string) => new ServerSetupFailed({ message })

/** Reads one answer from the controlling terminal, so a piped stdin (curl | sh) cannot answer it. */
const askTerminal = (question: string) => Effect.acquireUseRelease(
  Effect.tryPromise({ try: () => open("/dev/tty", "r+"), catch: () => failed("Run this command in a terminal.") }),
  tty => Effect.tryPromise({ try: async () => {
    await tty.write(question)
    const buffer = Buffer.alloc(256)
    const { bytesRead } = await tty.read(buffer, 0, buffer.length, null)
    return buffer.subarray(0, bytesRead).toString("utf8").trim()
  }, catch: () => failed("Could not read your answer from the terminal.") }),
  tty => Effect.promise(() => tty.close()),
)

const withService = <A, E>(use: (client: MagnitudeClient) => Effect.Effect<A, E>) => withLocalService(desktopServiceOrigin, use)

/** The service may still be starting, or restarting after a settings change; `serverSetup` bounds the whole wait. */
const readNetworkAccess = (settled: boolean) => withService(client => client.configuration.getNetworkAccess({})).pipe(
  Effect.filterOrFail(settings => !settled || !settings.pending, () => failed("The server is still restarting.")),
  Effect.retry(Schedule.spaced(Duration.seconds(1))),
  Effect.mapError(() => failed("The Magnitude server did not start.")))

const serviceUrl = new URL(desktopServiceOrigin)

const portAccepts = Effect.async<boolean>(resume => {
  const socket = connect({ host: serviceUrl.hostname, port: Number(serviceUrl.port) })
  const settle = (accepted: boolean) => { socket.destroy(); resume(Effect.succeed(accepted)) }
  socket.setTimeout(2000, () => settle(true))
  socket.once("connect", () => settle(true))
  socket.once("error", () => settle(false))
})

const answeringInstance = HttpClient.get(`${desktopServiceOrigin}/health`).pipe(
  Effect.flatMap(HttpClientResponse.schemaBodyJson(Schema.Struct({ id: Schema.String }))),
  Effect.map(health => health.id), Effect.scoped, Effect.timeout("3 seconds"), Effect.provide(FetchHttpClient.layer), Effect.option)

/** The instance the registered service's owner runs: the server profile's on Linux, the person's own on macOS. */
const serviceInstance = desktopApplication.observe.pipe(
  Effect.map(snapshot => snapshot.service._tag === "Ready" ? Option.some(snapshot.service.health.id)
    : snapshot.service._tag === "Starting" ? Option.map(snapshot.service.health, health => health.id) : Option.none<string>()),
  Effect.orElseSucceed(() => Option.none<string>()))

const servicePort: Effect.Effect<ServicePort> = Effect.gen(function* () {
  if (!(yield* portAccepts)) return "Free"
  const [answering, service] = yield* Effect.all([answeringInstance, serviceInstance], { concurrency: "unbounded" })
  return Option.isSome(answering) && Option.isSome(service) && answering.value === service.value ? "Service" : "Other"
})

const enableNetworkAccess: Effect.Effect<ServerAccess, ServerSetupFailed> = Effect.gen(function* () {
  const current = yield* readNetworkAccess(false)
  if (!current.enabled || !current.requireApiKey) {
    yield* withService(client => client.configuration.setNetworkAccess({ enabled: Option.some(true), requireApiKey: Option.some(true), bind: Option.none() })).pipe(
      Effect.mapError(() => failed("Could not turn on network access.")))
  }
  const configured = yield* readNetworkAccess(false)
  if (configured.pending) {
    yield* withService(client => client.application.restartApplication({})).pipe(Effect.mapError(() => failed("Could not restart the server to apply network access.")))
  }
  const settings = yield* readNetworkAccess(true)
  if (!settings.enabled || Option.isNone(settings.apiKey)) return yield* failed("Network access did not turn on. Check Settings → Network access in the browser app on this machine.")
  return {
    addresses: settings.interfaces.filter(entry => entry.kind !== "virtual").map(entry => entry.address.includes(":") ? `[${entry.address}]` : entry.address),
    port: settings.port, key: settings.apiKey.value,
  }
})

export const ServerSetupHostLive = Layer.effect(ServerSetupHost, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const executor = yield* CommandExecutor.CommandExecutor
  const terminal = yield* TerminalCommand
  const platform = process.platform
  const started = Math.floor(Date.now() / 1000)
  const output = (command: string, ...args: string[]) => executor.string(Command.make(command, ...args)).pipe(
    Effect.timeout("5 seconds"), Effect.orElseSucceed(() => ""))
  const journal = ["-u", "magnitude.service", `--since=@${started}`, "-n", "300", "-o", "cat", "--no-pager", "--quiet"]
  return ServerSetupHost.of({
    platform,
    user: userInfo().username,
    isRoot: process.getuid?.() === 0,
    underSudo: process.env.SUDO_USER !== undefined,
    hasServiceManager: platform === "linux" ? fs.exists("/run/systemd/system").pipe(Effect.orElseSucceed(() => false)) : Effect.succeed(platform === "darwin"),
    personalOwner: applicationStateDirectory({ platform, dataDirectory: desktopDataDirectory, override: Option.none() }).pipe(
      Effect.flatMap(directory => requestApplication(join(directory, "application.sock"), "Observe")),
      Effect.map(snapshot => Option.some(snapshot.owner._tag)), Effect.orElseSucceed(() => Option.none())),
    isSetUp: isServerSetUp,
    hasRemnants: platform === "linux"
      ? Effect.all([isServerSetUp, hasLinuxServerRemnants.pipe(
        Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(CommandExecutor.CommandExecutor, executor), Effect.orElseSucceed(() => false))]).pipe(Effect.map(found => found.some(Boolean)))
      : isServerSetUp,
    hasForeignAccount: platform === "linux"
      ? hasForeignServerAccount.pipe(Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(CommandExecutor.CommandExecutor, executor), Effect.orElseSucceed(() => false))
      : Effect.succeed(false),
    hasTerminal: Effect.tryPromise(() => open("/dev/tty", "r+").then(tty => tty.close())).pipe(Effect.as(true), Effect.orElseSucceed(() => false)),
    sudoWithoutPrompt: executor.exitCode(Command.make("/usr/bin/sudo", "-n", "true")).pipe(Effect.map(code => code === 0), Effect.orElseSucceed(() => false)),
    confirm: question => askTerminal(question).pipe(Effect.map(answer => /^y(es)?$/i.test(answer))),
    runRootStep: args => Effect.gen(function* () {
      const cli = installedServerCli(platform)
      if (Option.isNone(cli) || !(yield* fs.exists(cli.value).pipe(Effect.orElseSucceed(() => false)))) {
        return yield* failed("Server mode needs the installed Magnitude app. Install it from https://magnitude.dev/download.")
      }
      const root = process.getuid?.() === 0
      // In this terminal's session, so sudo reuses its ticket (the installer's Yes) or asks here once.
      const code = yield* terminal.run(root ? cli.value : "/usr/bin/sudo", root ? args : ["--", cli.value, ...args], { stdin: "inherit" }).pipe(
        Effect.orElseSucceed(() => -1))
      // The root step has printed what went wrong, and what it undid; sudo has said if it refused this account.
      const administrator = root ? "" : "\nIt needs an administrator account, one that can use sudo. If sudo refused this account above, run it from an administrator account."
      if (code !== 0) return yield* failed(args[0] === "_server-remove"
        ? `Server removal did not finish. Run \`magnitude server remove\` again once the problem above is fixed.${administrator}`
        : `Server setup did not finish.${administrator}`)
    }),
    servicePort,
    serviceRuns: (platform === "darwin"
      ? output("/bin/launchctl", "print", `system/${MAC_SERVER_LABEL}`).pipe(Effect.map(text => /^\s*runs = (\d+)$/m.exec(text)?.[1]))
      : output("/usr/bin/systemctl", "show", "magnitude.service", "--property=NRestarts", "--value").pipe(Effect.map(text => text.trim()))).pipe(
      Effect.map(value => value !== undefined && /^\d+$/.test(value) ? Option.some(Number(value)) : Option.none())),
    // Reading the system journal needs the adm group or root; sudo still holds setup's authorization.
    serviceLog: (platform === "darwin"
      ? fs.readFileString(join(desktopDataDirectory, "logs", "server.log")).pipe(Effect.map(text => text.split("\n").slice(-300)), Effect.orElseSucceed(() => []))
      : output("/usr/bin/journalctl", ...journal).pipe(
        Effect.flatMap(text => text.trim() !== "" || process.getuid?.() === 0 ? Effect.succeed(text) : output("/usr/bin/sudo", "-n", "--", "/usr/bin/journalctl", ...journal)),
        Effect.map(text => text.split("\n")))),
    enableNetworkAccess,
    write: text => Effect.sync(() => { process.stdout.write(text) }),
  })
})).pipe(Layer.provide([BunContext.layer, nodeTerminalCommand]))
