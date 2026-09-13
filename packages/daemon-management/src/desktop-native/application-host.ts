import { ApplicationControlFailed, ApplicationControlUnavailable, requestApplication, requestLoginStartup, requestApplicationUpdate } from "./application-control"
import { access } from "node:fs/promises"
import { join } from "node:path"
import { homedir } from "node:os"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"
import { Effect, Layer, Option, Schema } from "effect"
import { MAGNITUDE_RPC_VERSION } from "@magnitudedev/sdk"
import { ApplicationLaunchFailed, launchApplicationProcess, makeApplicationClient } from "./application-client"
import { NativeHost, NativeHostUnavailable, nativeHostLayer } from "./index"
import { WindowsProcessObserver, WindowsProcessObserverUnavailable, nativeWindowsProcessObserverLayer } from "./windows-process-observer"
import { WindowsProcessId } from "@magnitudedev/utils/windows-native"
import { applicationStateDirectory } from "./application-state-directory"
import { LINUX_DESKTOP_EXECUTABLE_PATH } from "@magnitudedev/release/executables"
import { NativeMacApplicationInstallation, waitForMacApplicationInstallation } from "./mac-update-installation"

export const makeDesktopApplicationHost = (developmentRepository: Option.Option<string>, windowsNative?: {
  readonly host: Layer.Layer<NativeHost, NativeHostUnavailable>
  readonly observer: Layer.Layer<WindowsProcessObserver, WindowsProcessObserverUnavailable>
}) => {
  const development = Option.isSome(developmentRepository)
  const desktopIsolatedProfile = development || process.env.MAGNITUDE_DEV_DATA_DIR !== undefined
  const desktopDataDirectory = process.env.MAGNITUDE_DEV_DATA_DIR ?? join(homedir(), development ? ".magnitude-desktop-dev" : ".magnitude")
  const desktopServiceOrigin = `http://127.0.0.1:${desktopIsolatedProfile ? process.env.MAGNITUDE_DEV_PORT ?? "11101" : "10100"}`
  const developmentAddon = Option.map(developmentRepository, repository => join(repository, `packages/daemon-management/dist/native/win32-${process.arch}/desktop-host.node`))
  const hostNative = windowsNative?.host ?? Option.match(developmentAddon, {
    onNone: () => Layer.effect(NativeHost, Effect.fail(new NativeHostUnavailable({ message: "The CLI installation is missing its native Windows adapter. Reinstall Magnitude CLI." }))),
    onSome: nativeHostLayer,
  })
  const processObserver = windowsNative?.observer ?? Option.match(developmentAddon, {
    onNone: () => Layer.effect(WindowsProcessObserver, Effect.fail(new WindowsProcessObserverUnavailable({ message: "The CLI installation is missing its native Windows process observer." }))),
    onSome: nativeWindowsProcessObserverLayer,
  })
  const localAppDataDirectory = Effect.flatMap(NativeHost, native => native.localAppDataDirectory).pipe(Effect.provide(hostNative))
  const windowsExecutable = process.env.MAGNITUDE_DESKTOP_PATH ? Effect.succeed(process.env.MAGNITUDE_DESKTOP_PATH)
    : localAppDataDirectory.pipe(Effect.map(directory => join(directory, "Programs/Magnitude/Magnitude.exe")),
      Effect.mapError(error => new ApplicationLaunchFailed({ message: error.message })))
  const stateDirectory = applicationStateDirectory({ platform: process.platform, dataDirectory: desktopDataDirectory, development, override: Option.fromNullable(process.env.MAGNITUDE_DESKTOP_STATE_DIR), localAppDataDirectory })
  const endpoint = process.platform === "win32" ? stateDirectory.pipe(Effect.flatMap(directory => Effect.flatMap(NativeHost, native => native.inspectEndpoint(directory))),
    Effect.provide(hostNative),
    Effect.mapError(error => new ApplicationControlFailed({ message: error.message })),
    Effect.flatMap(Option.match({ onNone: () => Effect.fail(new ApplicationControlUnavailable({ message: "Magnitude desktop is not running" })), onSome: Effect.succeed })),
  ) : stateDirectory.pipe(Effect.map(directory => join(directory, "application.sock")), Effect.mapError(error => new ApplicationControlFailed({ message: error.message })))
  const exists = (path: string) => Effect.tryPromise({ try: () => access(path), catch: () => new ApplicationLaunchFailed({ message: `Magnitude desktop is not installed at ${path}. Install the Magnitude desktop app before starting inference.` }) })

  const desktopApplication = makeApplicationClient({
    request: intent => endpoint.pipe(Effect.flatMap(path => requestApplication(path, intent))),
    launch: (intent, observe) => Effect.gen(function* () {
      if (process.platform === "linux" && !process.env.DISPLAY && !process.env.WAYLAND_DISPLAY) {
        return yield* new ApplicationLaunchFailed({ message: "Magnitude requires a graphical user session. Start the desktop app in your desktop session." })
      }
      const requireLaunchSession = process.platform === "win32" ? Effect.flatMap(NativeHost, native => native.requireInteractiveDesktop).pipe(
        Effect.provide(hostNative), Effect.mapError(error => new ApplicationLaunchFailed({ message: error.message })),
      ) : Effect.void
      const argumentsForIntent = intent === "EnsureRunning" ? ["--background"] : []
      const environment = { ...process.env, MAGNITUDE_SHELL_ENV_INHERITED: "1" }
      if (Option.isSome(developmentRepository)) {
        const repository = developmentRepository.value
        const executable = join(repository, "node_modules/electron/dist", process.platform === "darwin" ? "Electron.app/Contents/MacOS/Electron" : process.platform === "win32" ? "electron.exe" : "electron")
        yield* exists(executable)
        yield* exists(join(repository, "desktop/out/main/main.js"))
        yield* requireLaunchSession
        return yield* launchApplicationProcess({ executable, arguments: [join(repository, "desktop"), ...argumentsForIntent], environment }, observe)
      }
      if (process.platform === "darwin") {
        const bundle = process.env.MAGNITUDE_DESKTOP_PATH ?? "/Applications/Magnitude.app"
        // The old service-only Magnitude.app is not a desktop installation.
        yield* exists(join(bundle, "Contents/Frameworks/Electron Framework.framework"))
        yield* waitForMacApplicationInstallation(bundle).pipe(Effect.provide(NativeMacApplicationInstallation),
          Effect.mapError(error => new ApplicationLaunchFailed({ message: error.message })))
        return yield* launchApplicationProcess({ executable: "/usr/bin/open", arguments: ["-n", ...(intent === "EnsureRunning" ? ["-g", "-j"] : []), "-a", bundle, "--args", ...argumentsForIntent], environment }, observe)
      }
      const executable = process.env.MAGNITUDE_DESKTOP_PATH ?? (process.platform === "win32"
        ? yield* windowsExecutable
        : LINUX_DESKTOP_EXECUTABLE_PATH)
      yield* exists(executable)
      yield* requireLaunchSession
      return yield* launchApplicationProcess({ executable, arguments: argumentsForIntent, environment, installationGuarded: process.platform === "linux" && executable === LINUX_DESKTOP_EXECUTABLE_PATH }, observe)
    }),
  })
  const startDesktopApplication = Effect.flatMap(desktopApplication.ensure(), owner => desktopApplication.awaitReady(owner, MAGNITUDE_RPC_VERSION))

  const stopDesktopApplication = Effect.scoped(Effect.gen(function* () {
    const snapshot = yield* desktopApplication.observe.pipe(Effect.map(Option.some), Effect.catchTag("ApplicationControlUnavailable", () => Effect.succeed(Option.none())))
    if (Option.isNone(snapshot)) return
    if (process.platform === "win32") return yield* Effect.gen(function* () {
      const observer = yield* WindowsProcessObserver
      const pid = yield* Schema.decodeUnknown(WindowsProcessId)(snapshot.value.pid).pipe(Effect.mapError(() => new ApplicationLaunchFailed({ message: "Magnitude returned an invalid Windows process identity." })))
      const initial = yield* observer.observe(pid)
      if (Option.isNone(initial)) return
      if (yield* initial.value.exited) return
      const reply = yield* desktopApplication.quit
      if (reply.pid !== pid) return yield* new ApplicationLaunchFailed({ message: "The Magnitude application changed while quitting. Check its current status." })
      yield* initial.value.awaitExit
    }).pipe(Effect.provide(processObserver), Effect.mapError(error => new ApplicationLaunchFailed({ message: error.message })))
    const initial = yield* ProcessGroupControllerLive.inspect(snapshot.value.pid)
    yield* desktopApplication.quit
    if (Option.isNone(initial)) return
    for (;;) {
      const current = yield* ProcessGroupControllerLive.inspect(initial.value.pid)
      if (Option.isNone(current) || current.value.processStartIdentity !== initial.value.processStartIdentity) return
      yield* Effect.sleep("100 millis")
    }
  })).pipe(Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => new ApplicationLaunchFailed({ message: "Magnitude has not finished quitting. Open the app for cleanup details; the CLI will not kill its service independently." }) }))

  const readDesktopLoginStartup = endpoint.pipe(Effect.flatMap(path => requestLoginStartup(path, "read")))
  const setDesktopLoginStartup = (enabled: boolean) => desktopApplication.ensure().pipe(
    Effect.zipRight(endpoint.pipe(Effect.flatMap(path => requestLoginStartup(path, enabled ? "enable" : "disable")))),
  )

  const updateDesktopApplication = (action: import("@magnitudedev/sdk/desktop-host").ApplicationUpdateAction) => (action === "status" ? Effect.void : desktopApplication.ensure().pipe(Effect.asVoid)).pipe(
    Effect.zipRight(endpoint.pipe(Effect.flatMap(path => requestApplicationUpdate(path, action)))),
  )

  return { updateDesktopApplication, desktopIsolatedProfile, desktopDataDirectory, desktopServiceOrigin, desktopApplication, startDesktopApplication, stopDesktopApplication, readDesktopLoginStartup, setDesktopLoginStartup }

}
