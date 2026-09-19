import { FileSystem } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { join } from "node:path"
import { desktopEnvironment, DesktopEnvironment, DisposableDesktopUser } from "./desktop-environment"
import { InfrastructureFailure } from "./domain"

export const ApplicationContext = Schema.Struct({ ...DesktopEnvironment.fields, harnessHome: Schema.String })

/** One application context is shared by desktop, CLI, endpoint and harness acceptance. */
export const prepareApplicationContext = (root: string, port: number, inherited: Readonly<Record<string, string>>) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const user = yield* Effect.serviceOption(DisposableDesktopUser)
  if (Option.isSome(user)) {
    const profile = join(user.value.home, ".magnitude")
    if ((yield* fs.readDirectory(user.value.home)).includes(".magnitude")) return yield* new InfrastructureFailure({ operation: "application-context", message: "Disposable user's application data already exists; a fresh installed context is required" })
    const context = ApplicationContext.make({ mode: "installed-user", profile, port: 10100, harnessHome: user.value.home, environment: { ...inherited } })
    return { ...context, environment: yield* desktopEnvironment(context) }
  }
  const profile = join(root, "profile"), home = join(root, "home")
  const state = process.platform === "win32" ? join(profile, "state")
    : yield* fs.makeTempDirectoryScoped({ directory: "/tmp", prefix: "ml-state-" })
  yield* fs.makeDirectory(home, { recursive: true, mode: 0o700 })
  const context = ApplicationContext.make({ mode: "isolated", profile, port, harnessHome: join(profile, "harness-home"), environment: {
    ...inherited, HOME: home, USERPROFILE: home, APPDATA: join(home, "AppData", "Roaming"), LOCALAPPDATA: join(home, "AppData", "Local"),
    XDG_CONFIG_HOME: join(home, ".config"), XDG_DATA_HOME: join(home, ".local", "share"), MAGNITUDE_DESKTOP_STATE_DIR: state,
    MAGNITUDE_DEV_DATA_DIR: profile, MAGNITUDE_DEV_PORT: String(port),
  } })
  return { ...context, environment: yield* desktopEnvironment(context) }
})
