import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import {
  acquireRelease,
  findReleaseUpdate,
  RELEASE_DIST_TAGS_URL,
  UpdateDiscoveryFailed,
  currentHost,
  installMethodFromEnvironment,
  releaseBaseUrl,
  selectArtifact,
  updateCommandString,
  type PackageManager,
  type UpdateAction,
} from "@magnitudedev/release"
import { Context, Effect, Option, Schema } from "effect"

export class UpdateCommandFailed extends Schema.TaggedError<UpdateCommandFailed>()(
  "UpdateCommandFailed",
  {
    command: Schema.String,
    reason: Schema.String,
  },
) {}

export interface CliUpdaterOptions {
  readonly currentVersion: string
  readonly dataDir: string
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly npmPackageUrl?: string
  readonly releaseBaseUrl?: string
}

export interface CliUpdaterShape {
  /** The package manager that owns this installation, when one was declared. */
  readonly packageManager: Option.Option<PackageManager>
  /** A fresh, channel-selected and release-verified explicit update target. */
  readonly updateTarget: Effect.Effect<Option.Option<string>, UpdateDiscoveryFailed>
  readonly runUpdate: (action: UpdateAction) => Effect.Effect<void, UpdateCommandFailed>
}

export class CliUpdater extends Context.Tag("CliUpdater")<
  CliUpdater,
  CliUpdaterShape
>() {}

export const isDevelopmentVersion = (version: string): boolean =>
  version.includes("+dev.") || version === "0.0.0"

export const makeCliUpdater = (
  options: CliUpdaterOptions,
): Effect.Effect<
  CliUpdaterShape,
  never,
  | CommandExecutor.CommandExecutor
  | FileSystem.FileSystem
  | HttpClient.HttpClient
  | Path.Path
> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const http = yield* HttpClient.HttpClient
  const path = yield* Path.Path
  const commandExecutor = yield* CommandExecutor.CommandExecutor
  const dataDir = options.dataDir
  const environment = options.environment ?? process.env
  const installMethod = installMethodFromEnvironment(environment)
  const packageManager: Option.Option<PackageManager> = installMethod === "other"
    ? Option.none()
    : Option.some(installMethod)
  // Environment override mirrors MAGNITUDE_RELEASE_BASE_URL: it lets the
  // distribution simulator point discovery at a local registry.
  const npmPackageUrl = options.npmPackageUrl
    ?? environment.MAGNITUDE_NPM_PACKAGE_URL
    ?? RELEASE_DIST_TAGS_URL
  const verifyNativeRelease = (version: string) =>
    acquireRelease(
      options.releaseBaseUrl ?? releaseBaseUrl(),
      version,
      path.join(dataDir, "releases", "manifests", version),
    ).pipe(
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.provideService(HttpClient.HttpClient, http),
      Effect.provideService(Path.Path, path),
      Effect.flatMap(({ manifest }) => selectArtifact(
        manifest,
        "cli",
        currentHost(),
      )),
      Effect.as(version),
      Effect.mapError((error) => new UpdateDiscoveryFailed({
        stage: "release",
        reason: error.message,
      })),
    )

  const updateTarget = findReleaseUpdate({
    currentVersion: options.currentVersion,
    registryUrl: npmPackageUrl,
    verify: verifyNativeRelease,
  }).pipe(Effect.provideService(HttpClient.HttpClient, http))

  const runUpdate = (action: UpdateAction) => {
    const commandString = updateCommandString(action)
    return Command.make(action.command, ...action.args).pipe(
      Command.stdin("inherit"),
      Command.stdout("inherit"),
      Command.stderr("inherit"),
      Command.exitCode,
      Effect.provideService(CommandExecutor.CommandExecutor, commandExecutor),
      Effect.mapError((error) => new UpdateCommandFailed({
        command: commandString,
        reason: String(error),
      })),
      Effect.flatMap((exitCode) => Number(exitCode) === 0
        ? Effect.void
        : Effect.fail(new UpdateCommandFailed({
            command: commandString,
            reason: `exited with status ${Number(exitCode)}`,
          }))),
    )
  }

  return CliUpdater.of({
    packageManager,
    updateTarget,
    runUpdate,
  })
})
