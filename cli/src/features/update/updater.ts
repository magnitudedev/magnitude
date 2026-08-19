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
  updateCommandString,
  type PackageManager,
  type UpdateAction,
} from "@magnitudedev/release"
import { Context, Deferred, Effect, Option, Schema, Scope, Stream } from "effect"
import {
  admittedChannels,
  isNewerVersion,
  isValidVersion,
  newestFirst,
  releaseChannelOf,
} from "./release-channels"

const NPM_PACKAGE_URL =
  "https://registry.npmjs.org/-/package/@magnitudedev%2Fcli/dist-tags"
const NPM_RESPONSE_LIMIT = 64 * 1_024

const CHANNEL_TAGS = ["latest", "beta", "alpha"] as const

const NpmDistTagsSchema = Schema.Struct({
  latest: Schema.optional(Schema.String),
  beta: Schema.optional(Schema.String),
  alpha: Schema.optional(Schema.String),
})
type NpmDistTags = typeof NpmDistTagsSchema.Type

/**
 * The selected update candidate of the last completed check — admissible for
 * the writing client's channel and readiness-verified. Admissibility, newness,
 * and the dismissal floor are re-applied at read: the running binary (and so
 * its channel) may have changed since the write.
 */
export const UpdateCandidateSchema = Schema.Struct({
  version: Schema.String,
})
export type UpdateCandidate = typeof UpdateCandidateSchema.Type

/**
 * The dismissal floor. Dismissing an offer means "don't interrupt me again
 * until something strictly newer than this exists" — the floor suppresses
 * every candidate at or below it, so declining the best available option
 * never surfaces a strictly older one. A completed check whose best
 * candidate falls below the floor clears it: the registry retreated past
 * what was dismissed, so the slate is clean.
 */
export const UpdateDismissalSchema = Schema.Struct({
  version: Schema.String,
})
export type UpdateDismissal = typeof UpdateDismissalSchema.Type

export class UpdateCommandFailed extends Schema.TaggedError<UpdateCommandFailed>()(
  "UpdateCommandFailed",
  {
    command: Schema.String,
    reason: Schema.String,
  },
) {}

export class UpdateDiscoveryFailed extends Schema.TaggedError<UpdateDiscoveryFailed>()(
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

export interface UpdateDiscovery {
  /** The last known answer, available without the network. */
  readonly known: Option.Option<string>
  /**
   * Resolves when this launch's check completes, with the freshest available
   * answer. A completed check is authoritative — its selection stands even
   * when it retracts a cached offer; the known answer stands in only when the
   * check itself failed. Never fails; hangs only if the discovery scope
   * closes first, so consumers race or fork it.
   */
  readonly fresh: Effect.Effect<Option.Option<string>>
}

export interface CliUpdaterShape {
  /** The package manager that owns this installation, when one was declared. */
  readonly packageManager: Option.Option<PackageManager>
  readonly discover: Effect.Effect<UpdateDiscovery, never, Scope.Scope>
  /**
   * The version an explicit update should install: this launch's check,
   * selected for this client's channel, ignoring dismissals. None means the
   * client is up to date.
   */
  readonly updateTarget: Effect.Effect<Option.Option<string>, UpdateDiscoveryFailed>
  readonly dismissVersion: (version: string) => Effect.Effect<void>
  readonly runUpdate: (action: UpdateAction) => Effect.Effect<void, UpdateCommandFailed>
}

export class CliUpdater extends Context.Tag("CliUpdater")<
  CliUpdater,
  CliUpdaterShape
>() {}

export const isDevelopmentVersion = (version: string): boolean =>
  version.includes("+dev.") || version === "0.0.0"

export const updateReleaseNotesUrl = (version: string): string =>
  `https://github.com/magnitudedev/magnitude/releases/tag/${releaseTag(version)}`

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
  const cachePath = path.join(dataDir, "state", "version.json")
  const dismissalPath = path.join(dataDir, "state", "version-dismissal.json")
  const environment = options.environment ?? process.env
  const installMethod = installMethodFromEnvironment(environment)
  const packageManager: Option.Option<PackageManager> = installMethod === "other"
    ? Option.none()
    : Option.some(installMethod)
  const admitted = admittedChannels(releaseChannelOf(options.currentVersion))
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
    UpdateCandidateSchema,
  ).pipe(
    Effect.provideService(FileSystem.FileSystem, fs),
    Effect.map((result) => result._tag === "Present"
      ? Option.some(result.value)
      : Option.none()),
    Effect.catchAll(() => Effect.succeed(Option.none())),
  )

  const writeVersionInfo = (candidate: UpdateCandidate) =>
    writeStructuredFileAtomic(
      cachePath,
      UpdateCandidateSchema,
      candidate,
      { mode: 0o600 },
    ).pipe(
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.catchAll(() => Effect.void),
    )

  const clearVersionInfo = fs.remove(cachePath, { force: true }).pipe(
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

  const clearDismissal = fs.remove(dismissalPath, { force: true }).pipe(
    Effect.catchAll(() => Effect.void),
  )

  const fetchDistTags = Effect.gen(function* () {
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
    return yield* Schema.decodeUnknown(
      Schema.parseJson(NpmDistTagsSchema),
    )(new TextDecoder().decode(body)).pipe(
      Effect.mapError((error) => new UpdateDiscoveryFailed({
        stage: "registry",
        reason: String(error),
      })),
    )
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

  const admissibleUpgrade = (version: string): boolean =>
    admitted.has(releaseChannelOf(version))
    && isNewerVersion(version, options.currentVersion)

  const admissibleUpgrades = (tags: NpmDistTags): ReadonlyArray<string> =>
    CHANNEL_TAGS.flatMap((tag) => {
      const version = tags[tag]
      return version !== undefined && admissibleUpgrade(version) ? [version] : []
    })

  /**
   * One completed check: fetch the dist-tags, then walk the admissible
   * upgrades newest-first until one passes readiness — the normal case
   * verifies exactly one manifest; a candidate published before its release
   * assets is skipped, not fatal. The result is authoritative and the cache
   * records it, including the empty result (registry rollback).
   */
  const performCheck = Effect.gen(function* () {
    const tags = yield* fetchDistTags
    for (const version of newestFirst([...new Set(admissibleUpgrades(tags))])) {
      const ready = yield* verifyNativeRelease(version).pipe(
        Effect.as(true),
        Effect.catchAll((error) => Effect.logDebug(
          `Update candidate ${version} is not ready: ${String(error)}`,
        ).pipe(Effect.as(false))),
      )
      if (ready) {
        yield* writeVersionInfo({ version })
        return Option.some(version)
      }
    }
    yield* clearVersionInfo
    return Option.none<string>()
  })

  const refresh = performCheck.pipe(
    Effect.map(Option.some),
    Effect.catchAll((error) => Effect.logDebug(
      `Failed to refresh Magnitude update information: ${String(error)}`,
    ).pipe(Effect.as(Option.none<Option.Option<string>>()))),
  )

  const exceedsFloor = (
    version: string,
    dismissal: Option.Option<UpdateDismissal>,
  ): boolean => Option.match(dismissal, {
    onNone: () => true,
    onSome: (dismissed) => !isValidVersion(dismissed.version)
      || isNewerVersion(version, dismissed.version),
  })

  const rolledBackBelowFloor = (
    selection: Option.Option<string>,
    floor: string,
  ): boolean => !isValidVersion(floor) || Option.match(selection, {
    onNone: () => true,
    onSome: (best) => isNewerVersion(floor, best),
  })

  const discover: Effect.Effect<UpdateDiscovery, never, Scope.Scope> =
    Effect.gen(function* () {
      if (
        options.developmentBuild === true
        || isDevelopmentVersion(options.currentVersion)
        || Option.isNone(packageManager)
      ) {
        return {
          known: Option.none<string>(),
          fresh: Effect.succeed(Option.none<string>()),
        }
      }

      const config = yield* configStorage.load().pipe(
        Effect.catchAll(() => Effect.succeed({
          checkForUpdateOnStartup: Option.some(true),
        })),
      )
      if (Option.contains(config.checkForUpdateOnStartup, false)) {
        return {
          known: Option.none<string>(),
          fresh: Effect.succeed(Option.none<string>()),
        }
      }

      const dismissal = yield* readDismissal
      const known = Option.map(yield* readVersionInfo, (cached) => cached.version).pipe(
        Option.filter((version) =>
          admissibleUpgrade(version) && exceedsFloor(version, dismissal)),
      )
      const outcome = yield* Deferred.make<Option.Option<string>>()
      // A completed check is authoritative: its selection is the answer, even
      // when that retracts a cached offer (registry rollback). The known
      // answer stands in only when the check itself failed.
      yield* Effect.forkScoped(refresh.pipe(
        Effect.flatMap(Option.match({
          onNone: () => Effect.succeed(known),
          onSome: (selection) => Option.match(dismissal, {
            onNone: () => Effect.succeed(selection),
            onSome: (dismissed) =>
              rolledBackBelowFloor(selection, dismissed.version)
                ? clearDismissal.pipe(Effect.as(selection))
                : Effect.succeed(Option.filter(selection, (version) =>
                    exceedsFloor(version, dismissal))),
          }),
        })),
        Effect.flatMap((answer) => Deferred.succeed(outcome, answer)),
      ))
      return { known, fresh: Deferred.await(outcome) }
    })

  const updateTarget = performCheck

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
    packageManager,
    discover,
    updateTarget,
    dismissVersion,
    runUpdate,
  })
})
