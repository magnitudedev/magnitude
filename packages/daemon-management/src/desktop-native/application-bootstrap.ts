import { Effect, Layer, Schema } from "effect"
import { FileSystem } from "@effect/platform"
import { posix, win32 } from "node:path"
import { MAGNITUDE_RPC_VERSION } from "@magnitudedev/sdk"
import { nativeWindowsJobOwnerLayer, nativeWindowsPrivatePipesLayer } from "@magnitudedev/utils/windows-native"
import { makeUnixOwnedChildSpawner, OwnedChildSpawner, OwnedChildSpawnFailed, type OwnedChildCommand } from "./owned-child"
import { makeWindowsOwnedChildSpawner } from "./windows-owned-child"
import { makeOwnedService } from "./owned-service"
import { previousInstallationUpgrade } from "./previous-installation-live"
import { requireServicePort } from "./service-port"
import type { ChildOutputMode } from "./child-output"

export const ApplicationRuntime = Schema.Union(
  Schema.TaggedStruct("Installed", { resourcesDirectory: Schema.String }),
  Schema.TaggedStruct("Development", { repository: Schema.String }),
)
export type ApplicationRuntime = typeof ApplicationRuntime.Type

export class ApplicationRuntimeUnavailable extends Schema.TaggedError<ApplicationRuntimeUnavailable>()("ApplicationRuntimeUnavailable", {
  message: Schema.String,
}) {}

/** Resolve the executing payload, including shell symlinks, rather than a PATH installation guess. */
export const resolveInstalledApplicationRuntime = (executable: string, platform: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const resolved = yield* fs.realPath(executable).pipe(Effect.mapError(error => new ApplicationRuntimeUnavailable({ message: error.message })))
  const path = paths(platform)
  const resourcesDirectory = path.dirname(resolved)
  const name = path.basename(resolved)
  const resources = path.basename(resourcesDirectory)
  const matched = platform === "darwin"
    ? name === "magnitude" && resources === "Resources" && path.basename(path.dirname(resourcesDirectory)) === "Contents"
      && path.dirname(path.dirname(resourcesDirectory)).endsWith(".app")
    : name === (platform === "win32" ? "magnitude.exe" : "magnitude") && resources === "resources"
  if (!matched) return yield* new ApplicationRuntimeUnavailable({ message: "This command is not inside a complete Magnitude installation. Reinstall Magnitude." })
  return { _tag: "Installed", resourcesDirectory } satisfies ApplicationRuntime
})
export const ApplicationProfile = Schema.Struct({
  dataDirectory: Schema.String, isolated: Schema.Boolean, port: Schema.Number, endpoint: Schema.String,
})
export type ApplicationProfile = typeof ApplicationProfile.Type

type Environment = Readonly<Record<string, string | undefined>>
const paths = (platform: string) => platform === "win32" ? win32 : posix

/** Selection is observational; only native owner admission creates protected state. */
export const resolveApplicationProfile = (options: {
  readonly runtime: ApplicationRuntime; readonly home: string; readonly platform: string
  readonly acceptance: boolean; readonly environment: Environment
}): ApplicationProfile => {
  const { runtime, environment, acceptance } = options
  const isolated = acceptance || runtime._tag === "Development" || environment.MAGNITUDE_DEV_DATA_DIR !== undefined
  const dataDirectory = environment.MAGNITUDE_DEV_DATA_DIR ?? paths(options.platform).join(options.home,
    acceptance ? ".magnitude-update-acceptance" : runtime._tag === "Installed" ? ".magnitude" : ".magnitude-desktop-dev")
  const port = isolated ? Number(environment.MAGNITUDE_DEV_PORT ?? (acceptance ? 11143 : 11101)) : 10100
  return { dataDirectory, isolated, port, endpoint: `http://127.0.0.1:${port}` }
}

export const applicationNativeHostPath = (runtime: ApplicationRuntime, platform: string, architecture: string) =>
  runtime._tag === "Installed" ? paths(platform).join(runtime.resourcesDirectory, "desktop-host.node")
    : paths(platform).join(runtime.repository, `packages/daemon-management/dist/native/${platform}-${architecture}/desktop-host.node`)

export const applicationServiceCommand = (options: {
  readonly runtime: ApplicationRuntime; readonly profile: ApplicationProfile
  readonly platform: string; readonly architecture: string; readonly environment: Environment
  readonly output: ChildOutputMode
}): OwnedChildCommand => {
  const { runtime, profile, environment, platform, architecture } = options
  const path = paths(platform)
  return {
    output: options.output,
    executable: runtime._tag === "Installed"
      ? path.join(runtime.resourcesDirectory, platform === "win32" ? "magnitude-service.exe" : "magnitude-service")
      : environment.MAGNITUDE_BUN_PATH ?? "bun",
    arguments: [...(runtime._tag === "Installed" ? [] : [path.join(runtime.repository, "packages/acn/src/binary.ts")]),
      "serve", "--data-dir", profile.dataDirectory, "--port", String(profile.port)],
    environment: { ...environment, MAGNITUDE_NATIVE_HOST: applicationNativeHostPath(runtime, platform, architecture),
      ...(runtime._tag === "Installed" || environment.MAGNITUDE_ICN_PATH ? {} : {
        MAGNITUDE_ICN_PATH: path.join(runtime.repository, "inference/target/development/installation.json"),
      }) },
  }
}

/** Called within the retained application owner's scope, after installation admission. */
export const makeApplicationService = (options: {
  readonly runtime: ApplicationRuntime; readonly profile: ApplicationProfile
  readonly stateDirectory: string; readonly home: string; readonly environment: Environment
  readonly output: ChildOutputMode
}) => Effect.gen(function* () {
  const addon = applicationNativeHostPath(options.runtime, process.platform, process.arch)
  const spawner = process.platform === "win32" ? yield* Effect.gen(function* () {
    const pipes = yield* Layer.build(nativeWindowsPrivatePipesLayer(addon))
    const jobs = yield* Layer.build(nativeWindowsJobOwnerLayer(addon))
    return yield* makeWindowsOwnedChildSpawner.pipe(Effect.provide(pipes), Effect.provide(jobs))
  }) : yield* makeUnixOwnedChildSpawner
  const upgrade: Effect.Effect<void, { readonly message: string }> = options.runtime._tag === "Installed" && !options.profile.isolated && process.platform !== "win32"
    ? yield* previousInstallationUpgrade({ home: options.home, dataDirectory: options.profile.dataDirectory, stateDirectory: options.stateDirectory })
    : Effect.void
  const checked = yield* requireServicePort(options.profile.port).pipe(Effect.provideService(OwnedChildSpawner, spawner))
  const admitted = OwnedChildSpawner.of({ spawn: command => upgrade.pipe(
    Effect.mapError(error => new OwnedChildSpawnFailed({ executable: command.executable, message: error.message })),
    Effect.zipRight(checked.spawn(command)),
  ) })
  return yield* makeOwnedService(applicationServiceCommand({ ...options, platform: process.platform, architecture: process.arch }), MAGNITUDE_RPC_VERSION)
    .pipe(Effect.provideService(OwnedChildSpawner, admitted))
})
