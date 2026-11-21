import { Command, CommandExecutor } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Data, Effect, Option } from "effect"
import { NativeHost, resolveInstalledApplicationRuntime } from "@magnitudedev/daemon-management/desktop-native"
import { bundledWindowsNative } from "@magnitudedev/daemon-management/bun"
import { desktopApplication, desktopServiceOrigin } from "../server/application"
import { isDevelopmentBuild } from "../runtime/environment"
import { runCommand } from "./output"

class AppNotServing extends Data.TaggedError("AppNotServing")<{ readonly message: string }> {}

/** Windows reports a URL handler as run even without an interactive desktop, as over SSH, so that is checked first. */
const interactiveWindowsDesktop = Effect.flatMap(NativeHost, native => native.requireInteractiveDesktop).pipe(
  Effect.provide(bundledWindowsNative.host), Effect.as(true), Effect.orElseSucceed(() => false))

const browserCommand = (url: string): Effect.Effect<Option.Option<Command.Command>> =>
  process.platform === "darwin" ? Effect.succeed(Option.some(Command.make("/usr/bin/open", url)))
    : process.platform === "win32" ? interactiveWindowsDesktop.pipe(Effect.map(desktop =>
      desktop ? Option.some(Command.make("rundll32.exe", "url.dll,FileProtocolHandler", url)) : Option.none()))
      : Effect.succeed(process.env.DISPLAY || process.env.WAYLAND_DISPLAY ? Option.some(Command.make("xdg-open", url)) : Option.none())

/**
 * A CLI-only installation has no desktop window to show: its app is the one `magnitude serve` serves to
 * browsers, so it is opened in this computer's browser, or its address is printed where there is none.
 */
const openInBrowser = Effect.gen(function* () {
  const running = yield* desktopApplication.observe.pipe(
    Effect.map(snapshot => snapshot.service._tag !== "Stopping" && snapshot.service._tag !== "Stopped"),
    Effect.orElseSucceed(() => false))
  if (!running) return yield* new AppNotServing({ message: `Magnitude isn't running. Start \`magnitude serve\`, then open ${desktopServiceOrigin} in a browser or run \`magnitude app open\` again.` })
  const executor = yield* CommandExecutor.CommandExecutor
  const opened = yield* Option.match(yield* browserCommand(desktopServiceOrigin), {
    onNone: () => Effect.succeed(false),
    onSome: command => executor.exitCode(command).pipe(Effect.map(code => code === 0), Effect.orElseSucceed(() => false)),
  })
  return opened ? `Opened Magnitude in your browser at ${desktopServiceOrigin}.\n`
    : `Open ${desktopServiceOrigin} in a browser on this computer. To open it from another computer, see \`magnitude docs remote-access\`.\n`
}).pipe(Effect.provide(BunContext.layer))

const standalone = isDevelopmentBuild() ? Effect.succeed(false)
  : resolveInstalledApplicationRuntime(process.execPath, process.platform).pipe(
    Effect.map(runtime => runtime.layout === "Standalone"), Effect.orElseSucceed(() => false), Effect.provide(BunContext.layer))

const showDesktopWindow = desktopApplication.ensure("ShowWindow").pipe(Effect.as("Opened Magnitude.\n"))

export const openApplication = () => runCommand({
  effect: standalone.pipe(Effect.flatMap((cliOnly): Effect.Effect<string, unknown> => cliOnly ? openInBrowser : showDesktopWindow)),
  render: message => message,
})
