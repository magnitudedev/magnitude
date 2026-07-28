import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import { Effect, Option } from "effect"
import { homedir } from "node:os"
import { join } from "node:path"
import {
  acquireRelease,
  currentHost,
  embeddedTrustedReleaseKeys,
  installArtifact,
  NodeArchiveExtractor,
  selectArtifact,
} from "@magnitudedev/release"
import {
  BinaryNotFound,
  BinaryVersionMismatch,
  DaemonSpawnFailed,
  DownloadFailed,
} from "./errors"

export interface ResolveBinaryOptions {
  readonly binaryPath?: string
  readonly version?: string
  readonly dataDir?: string
}

export interface ResolvedBinaryCommand {
  readonly command: string[]
  readonly needsDownload: boolean
}

export const defaultDataDir = (): string => join(homedir(), ".magnitude")
export const defaultBinaryPath = (dataDir: string = defaultDataDir()): string =>
  join(dataDir, "bin", "magnitude-acn")

const releaseRoot = (dataDir: string) => join(dataDir, "releases")
const acnRoot = (dataDir: string, version: string) =>
  join(releaseRoot(dataDir), "acn", version, currentHost())
const pointerPath = (dataDir: string, version: string) =>
  join(acnRoot(dataDir, version), "current.txt")
const executableName = () => process.platform === "win32"
  ? "magnitude-acn.exe"
  : "magnitude-acn"

export function releaseTag(version: string): string {
  return `@magnitudedev/cli@${version}`
}

export function releaseBaseUrl(): string {
  return (
    process.env.MAGNITUDE_RELEASE_BASE_URL ??
    "https://github.com/magnitudedev/magnitude/releases/download"
  ).replace(/\/+$/, "")
}

const validateBinaryVersion = (
  binaryPath: string,
  expectedVersion: string,
): Effect.Effect<
  void,
  BinaryVersionMismatch | DaemonSpawnFailed,
  CommandExecutor.CommandExecutor
> =>
  Effect.gen(function* () {
    const actual = yield* Command.make(binaryPath, "version").pipe(
      Command.string,
      Effect.map((value) => value.trim()),
      Effect.mapError((cause) => new DaemonSpawnFailed({ reason: String(cause) })),
    )
    if (actual !== expectedVersion) {
      return yield* new BinaryVersionMismatch({
        path: binaryPath,
        expected: expectedVersion,
        actual,
      })
    }
  })

const cachedAcn = (
  dataDir: string,
  version: string,
): Effect.Effect<
  Option.Option<string>,
  never,
  CommandExecutor.CommandExecutor | FileSystem.FileSystem | Path.Path
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const digest = yield* fs.readFileString(pointerPath(dataDir, version)).pipe(
      Effect.map((value) => value.trim()),
      Effect.orElseSucceed(() => ""),
    )
    if (!/^[a-f0-9]{64}$/.test(digest)) return Option.none()
    const executable = path.join(acnRoot(dataDir, version), digest, "bin", executableName())
    if (!(yield* fs.exists(executable).pipe(Effect.orElseSucceed(() => false)))) {
      return Option.none()
    }
    const valid = yield* validateBinaryVersion(executable, version).pipe(
      Effect.as(true),
      Effect.catchAll(() => Effect.succeed(false)),
    )
    return valid ? Option.some(executable) : Option.none()
  })

const publishPointer = (
  dataDir: string,
  version: string,
  digest: string,
): Effect.Effect<void, DownloadFailed, FileSystem.FileSystem | Path.Path> =>
  Effect.scoped(
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const pointer = pointerPath(dataDir, version)
      const parent = path.dirname(pointer)
      yield* fs.makeDirectory(parent, { recursive: true, mode: 0o700 }).pipe(
        Effect.mapError(acquisitionFailure(version)),
      )
      const temporary = yield* fs.makeTempFileScoped({
        directory: parent,
        prefix: ".current-",
      }).pipe(Effect.mapError(acquisitionFailure(version)))
      yield* fs.writeFileString(temporary, `${digest}\n`, { mode: 0o600 }).pipe(
        Effect.mapError(acquisitionFailure(version)),
      )
      yield* fs.rename(temporary, pointer).pipe(
        Effect.catchAll(() =>
          fs.remove(pointer, { force: true }).pipe(
            Effect.zipRight(fs.rename(temporary, pointer)),
          )
        ),
        Effect.mapError(acquisitionFailure(version)),
      )
    }),
  )

const acquisitionFailure = (version: string) => (cause: unknown) =>
  new DownloadFailed({
    url: `${releaseBaseUrl()}/${encodeURIComponent(releaseTag(version))}`,
    status: 0,
    reason: cause instanceof Error ? cause.message : String(cause),
  })

const ensureAcn = (
  version: string,
  dataDir: string,
): Effect.Effect<
  { readonly path: string; readonly acquired: boolean },
  DownloadFailed | BinaryVersionMismatch | DaemonSpawnFailed,
  | CommandExecutor.CommandExecutor
  | FileSystem.FileSystem
  | Path.Path
  | HttpClient.HttpClient
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const cached = yield* cachedAcn(dataDir, version)
    if (Option.isSome(cached)) return { path: cached.value, acquired: false }

    const authenticated = yield* acquireRelease(
      releaseBaseUrl(),
      version,
      embeddedTrustedReleaseKeys(),
      path.join(releaseRoot(dataDir), "manifests", version),
    ).pipe(Effect.mapError(acquisitionFailure(version)))
    const artifact = yield* selectArtifact(
      authenticated.manifest,
      "acn",
      currentHost(),
    ).pipe(Effect.mapError(acquisitionFailure(version)))
    const destination = path.join(acnRoot(dataDir, version), artifact.sha256)
    const executable = path.join(destination, "bin", executableName())

    if (yield* fs.exists(destination).pipe(Effect.orElseSucceed(() => false))) {
      const valid = yield* validateBinaryVersion(executable, version).pipe(
        Effect.as(true),
        Effect.catchAll(() => Effect.succeed(false)),
      )
      if (!valid) {
        yield* fs.remove(destination, { recursive: true, force: true }).pipe(
          Effect.mapError(acquisitionFailure(version)),
        )
      }
    }
    let acquired = false
    if (!(yield* fs.exists(destination).pipe(Effect.orElseSucceed(() => false)))) {
      yield* installArtifact(
        releaseBaseUrl(),
        version,
        artifact,
        destination,
      ).pipe(
        Effect.provide(NodeArchiveExtractor),
        Effect.mapError(acquisitionFailure(version)),
      )
      acquired = true
    }
    yield* validateBinaryVersion(executable, version)
    yield* publishPointer(dataDir, version, artifact.sha256)
    return { path: executable, acquired }
  })

export const downloadAcn = (
  version: string,
  dataDir: string,
): Effect.Effect<
  string,
  DownloadFailed | BinaryVersionMismatch | DaemonSpawnFailed,
  | CommandExecutor.CommandExecutor
  | FileSystem.FileSystem
  | Path.Path
  | HttpClient.HttpClient
> => ensureAcn(version, dataDir).pipe(Effect.map(({ path }) => path))

export const resolveBinaryCommand = (
  options?: ResolveBinaryOptions,
): Effect.Effect<
  ResolvedBinaryCommand,
  DownloadFailed | BinaryNotFound | BinaryVersionMismatch | DaemonSpawnFailed,
  | CommandExecutor.CommandExecutor
  | FileSystem.FileSystem
  | Path.Path
  | HttpClient.HttpClient
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const dataDir = options?.dataDir ?? defaultDataDir()
    const expectedVersion = options?.version

    if (options?.binaryPath) {
      if (!(yield* fs.exists(options.binaryPath).pipe(Effect.orElseSucceed(() => false)))) {
        return yield* new BinaryNotFound({ path: options.binaryPath })
      }
      if (expectedVersion) yield* validateBinaryVersion(options.binaryPath, expectedVersion)
      return {
        command: [options.binaryPath, "serve", "--register", "--data-dir", dataDir],
        needsDownload: false,
      }
    }

    if (!expectedVersion) {
      const path = defaultBinaryPath(dataDir)
      if (!(yield* fs.exists(path).pipe(Effect.orElseSucceed(() => false)))) {
        return yield* new BinaryNotFound({ path })
      }
      return {
        command: [path, "serve", "--register", "--data-dir", dataDir],
        needsDownload: false,
      }
    }

    const resolved = yield* ensureAcn(expectedVersion, dataDir)
    return {
      command: [resolved.path, "serve", "--register", "--data-dir", dataDir],
      needsDownload: resolved.acquired,
    }
  })
