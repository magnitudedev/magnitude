import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as Path from "@effect/platform/Path"
import {
  defaultGlobalStorageRoot,
  makeConfigStorage,
  makeGlobalStorage,
  GlobalStorage,
  readStructuredFile,
  writeStructuredFileAtomic,
} from "@magnitudedev/storage"
import {
  acquireRelease,
  currentHost,
  installMethodFromEnvironment,
  releaseBaseUrl,
  releaseTag,
  selectArtifact,
  updateActionFor,
  updateCommandString,
  type InstallMethod,
  type UpdateAction,
} from "@magnitudedev/release"
import { Clock, Context, Effect, Option, Schema, Scope, Stream } from "effect"
import semver from "semver"

const NPM_PACKAGE_URL =
  "https://registry.npmjs.org/-/package/@magnitudedev%2Fcli/dist-tags"
const NPM_RESPONSE_LIMIT = 64 * 1_024
const UPDATE_CACHE_TTL_MS = 20 * 60 * 60 * 1_000

export const UpdateVersionInfoSchema = Schema.Struct({
  latestVersion: Schema.String,
  lastCheckedAt: Schema.String,
})
export type UpdateVersionInfo = typeof UpdateVersionInfoSchema.Type

export const UpdateDismissalSchema = Schema.Struct({
  version: Schema.String,
})
export type UpdateDismissal = typeof UpdateDismissalSchema.Type

const NpmDistTagsSchema = Schema.Struct({
  latest: Schema.String,
})

export class UpdateCommandFailed extends Schema.TaggedError<UpdateCommandFailed>()(
  "UpdateCommandFailed",
  {
    command: Schema.String,
    reason: Schema.String,
  },
) {}

class UpdateDiscoveryFailed extends Schema.TaggedError<UpdateDiscoveryFailed>()(
  "UpdateDiscoveryFailed",
  {
    stage: Schema.Literal("registry", "release"),
    reason: Schema.String,
  },
) {}

export interface CliUpdaterOptions {
  readonly currentVersion: string
  readonly dataDir?: string
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly developmentBuild?: boolean
  readonly npmPackageUrl?: string
  readonly releaseBaseUrl?: string
}

export interface CliUpdaterShape {
  readonly installMethod: InstallMethod
  readonly updateAction: Option.Option<UpdateAction>
  readonly getUpgradeVersion: Effect.Effect<Option.Option<string>, never, Scope.Scope>
  readonly dismissVersion: (version: string) => Effect.Effect<void>
  readonly runUpdate: (action: UpdateAction) => Effect.Effect<void, UpdateCommandFailed>
}

export class CliUpdater extends Context.Tag("CliUpdater")<
  CliUpdater,
  CliUpdaterShape
>() {}

export const isDevelopmentVersion = (version: string): boolean =>
  version.includes("+dev.") || version === "0.0.0"

export const isNewerVersion = (
  candidate: string,
  current: string,
): boolean => semver.valid(candidate) !== null
  && semver.valid(current) !== null
  && semver.gt(candidate, current)

export const updateReleaseNotesUrl = (version: string): string =>
  `https://github.com/magnitudedev/magnitude/releases/tag/${encodeURIComponent(releaseTag(version))}`

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
  const dataDir = options.dataDir ?? defaultGlobalStorageRoot()
  const cachePath = path.join(dataDir, "version.json")
  const dismissalPath = path.join(dataDir, "version-dismissal.json")
  const environment = options.environment ?? process.env
  const installMethod = installMethodFromEnvironment(environment)
  const updateAction = updateActionFor(installMethod)
  // Environment override mirrors MAGNITUDE_RELEASE_BASE_URL: it lets the
  // distribution simulator point discovery at a local registry.
  const npmPackageUrl = options.npmPackageUrl
    ?? environment.MAGNITUDE_NPM_PACKAGE_URL
    ?? NPM_PACKAGE_URL
  const configStorage = yield* makeConfigStorage().pipe(
    Effect.provideService(GlobalStorage, makeGlobalStorage({ root: dataDir })),
  )

  const readVersionInfo = readStructuredFile(
    cachePath,
    UpdateVersionInfoSchema,
  ).pipe(
    Effect.provideService(FileSystem.FileSystem, fs),
    Effect.map((result) => result._tag === "Present"
      ? Option.some(result.value)
      : Option.none()),
    Effect.catchAll(() => Effect.succeed(Option.none())),
  )

  const writeVersionInfo = (info: UpdateVersionInfo) =>
    writeStructuredFileAtomic(
      cachePath,
      UpdateVersionInfoSchema,
      info,
      { mode: 0o600 },
    ).pipe(
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.catchAll(() => Effect.void),
    )

  const readDismissal = readStructuredFile(
    dismissalPath,
    UpdateDismissalSchema,
  ).pipe(
    Effect.provideService(FileSystem.FileSystem, fs),
    Effect.map((result) => result._tag === "Present"
      ? Option.some(result.value)
      : Option.none()),
    Effect.catchAll(() => Effect.succeed(Option.none())),
  )

  const writeDismissal = (dismissal: UpdateDismissal) =>
    writeStructuredFileAtomic(
      dismissalPath,
      UpdateDismissalSchema,
      dismissal,
      { mode: 0o600 },
    ).pipe(
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.catchAll(() => Effect.void),
    )

  const fetchLatestVersion = Effect.gen(function* () {
    const response = yield* http.execute(HttpClientRequest.get(
      npmPackageUrl,
    )).pipe(Effect.mapError((error) => new UpdateDiscoveryFailed({
      stage: "registry",
      reason: String(error),
    })))
    if (response.status < 200 || response.status >= 300) {
      return yield* new UpdateDiscoveryFailed({
        stage: "registry",
        reason: `npm registry returned HTTP ${response.status}`,
      })
    }
    const bytes = yield* response.stream.pipe(
      Stream.runFoldEffect(
        { chunks: [] as Uint8Array[], size: 0 },
        (state, chunk) => {
          const size = state.size + chunk.byteLength
          return size > NPM_RESPONSE_LIMIT
            ? new UpdateDiscoveryFailed({
                stage: "registry",
                reason: "npm registry response exceeds its size bound",
              })
            : Effect.succeed({ chunks: [...state.chunks, chunk], size })
        },
      ),
      Effect.mapError((error) => error instanceof UpdateDiscoveryFailed
        ? error
        : new UpdateDiscoveryFailed({
            stage: "registry",
            reason: String(error),
          })),
    )
    const body = new Uint8Array(bytes.size)
    let offset = 0
    for (const chunk of bytes.chunks) {
      body.set(chunk, offset)
      offset += chunk.byteLength
    }
    const distTags = yield* Schema.decodeUnknown(
      Schema.parseJson(NpmDistTagsSchema),
    )(new TextDecoder().decode(body)).pipe(
      Effect.mapError((error) => new UpdateDiscoveryFailed({
        stage: "registry",
        reason: String(error),
      })),
    )
    const version = distTags.latest
    if (semver.valid(version) === null) {
      return yield* new UpdateDiscoveryFailed({
        stage: "registry",
        reason: "npm latest is not a valid semantic version",
      })
    }
    return version
  }).pipe(Effect.timeoutFail({
    duration: "30 seconds",
    onTimeout: () => new UpdateDiscoveryFailed({
      stage: "registry",
      reason: "npm registry request timed out",
    }),
  }))

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
      Effect.asVoid,
      Effect.mapError((error) => new UpdateDiscoveryFailed({
        stage: "release",
        reason: error.message,
      })),
    )

  const refresh = Effect.gen(function* () {
    const latestVersion = yield* fetchLatestVersion
    yield* verifyNativeRelease(latestVersion)
    const now = yield* Clock.currentTimeMillis
    yield* writeVersionInfo({
      latestVersion,
      lastCheckedAt: new Date(now).toISOString(),
    })
  }).pipe(
    Effect.catchAll((error) => Effect.logDebug(
      `Failed to refresh Magnitude update information: ${String(error)}`,
    )),
  )

  const getUpgradeVersion = Effect.gen(function* () {
    if (
      options.developmentBuild === true
      || isDevelopmentVersion(options.currentVersion)
      || Option.isNone(updateAction)
    ) return Option.none()

    const config = yield* configStorage.load().pipe(
      Effect.catchAll(() => Effect.succeed({
        checkForUpdateOnStartup: Option.some(true),
      })),
    )
    if (Option.contains(config.checkForUpdateOnStartup, false)) {
      return Option.none()
    }

    const info = yield* readVersionInfo
    const now = yield* Clock.currentTimeMillis
    const stale = Option.match(info, {
      onNone: () => true,
      onSome: ({ lastCheckedAt }) => {
        const checkedAt = Date.parse(lastCheckedAt)
        return !Number.isFinite(checkedAt)
          || checkedAt < now - UPDATE_CACHE_TTL_MS
      },
    })
    if (stale) yield* Effect.forkScoped(refresh)

    const dismissal = yield* readDismissal
    return Option.filter(info, ({ latestVersion }) =>
      isNewerVersion(latestVersion, options.currentVersion)
      && !Option.exists(dismissal, ({ version }) => version === latestVersion),
    ).pipe(Option.map(({ latestVersion }) => latestVersion))
  })

  const dismissVersion = (version: string) => writeDismissal({ version })

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
    installMethod,
    updateAction,
    getUpgradeVersion,
    dismissVersion,
    runUpdate,
  })
})
