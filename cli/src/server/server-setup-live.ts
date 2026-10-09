import { Command, CommandExecutor, FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Duration, Effect, Layer, Option, Schedule } from "effect"
import { open } from "node:fs/promises"
import { join } from "node:path"
import { userInfo } from "node:os"
import { MagnitudeClient } from "@magnitudedev/sdk"
import { applicationStateDirectory, installedServerCli, requestApplication } from "@magnitudedev/daemon-management/desktop-native"
import { isServerSetUp } from "./serve-owner"
import { desktopDataDirectory, desktopServiceOrigin } from "./application"
import { ServerSetupFailed, ServerSetupHost, type ServerAccess } from "./server-setup"
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

/** The service may still be starting, or restarting after a settings change. */
const readNetworkAccess = (settled: boolean) => withService(client => client.configuration.getNetworkAccess({})).pipe(
  Effect.filterOrFail(settings => !settled || !settings.pending, () => failed("The server is still restarting.")),
  Effect.retry(Schedule.spaced(Duration.seconds(1)).pipe(Schedule.upTo(Duration.minutes(3)))),
  Effect.mapError(() => failed("The Magnitude server did not start. See `journalctl -u magnitude` for details.")))

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
  const platform = process.platform
  return ServerSetupHost.of({
    platform,
    user: userInfo().username,
    isRoot: process.getuid?.() === 0,
    hasServiceManager: platform === "linux" ? fs.exists("/run/systemd/system").pipe(Effect.orElseSucceed(() => false)) : Effect.succeed(platform === "darwin"),
    personalOwner: applicationStateDirectory({ platform, dataDirectory: desktopDataDirectory, override: Option.none() }).pipe(
      Effect.flatMap(directory => requestApplication(join(directory, "application.sock"), "Observe")),
      Effect.map(snapshot => Option.some(snapshot.owner._tag)), Effect.orElseSucceed(() => Option.none())),
    isSetUp: isServerSetUp,
    hasTerminal: Effect.tryPromise(() => open("/dev/tty", "r+").then(tty => tty.close())).pipe(Effect.as(true), Effect.orElseSucceed(() => false)),
    sudoWithoutPrompt: executor.exitCode(Command.make("/usr/bin/sudo", "-n", "true")).pipe(Effect.map(code => code === 0), Effect.orElseSucceed(() => false)),
    confirm: question => askTerminal(question).pipe(Effect.map(answer => /^y(es)?$/i.test(answer))),
    runRootStep: args => Effect.gen(function* () {
      const cli = installedServerCli(platform)
      if (Option.isNone(cli) || !(yield* fs.exists(cli.value).pipe(Effect.orElseSucceed(() => false)))) {
        return yield* failed("Server mode needs the installed Magnitude app. Install it from https://magnitude.dev/download.")
      }
      const code = yield* executor.exitCode(Command.make("/usr/bin/sudo", "--", cli.value, ...args).pipe(
        Command.stdin("inherit"), Command.stdout("inherit"), Command.stderr("inherit"))).pipe(Effect.orElseSucceed(() => -1))
      if (code !== 0) return yield* failed("The root step did not finish; nothing else was changed.")
    }),
    enableNetworkAccess,
    write: text => Effect.sync(() => { process.stdout.write(text) }),
  })
})).pipe(Layer.provide(BunContext.layer))
