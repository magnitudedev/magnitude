import { Effect, Schedule } from "effect"
import { FileSystem } from "@effect/platform/FileSystem"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import { dirname, join } from "node:path"
import { homedir } from "node:os"
import {
  BinaryNotFound,
  BinaryVersionMismatch,
  DownloadFailed,
  DaemonSpawnFailed
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
  `${dataDir}/bin/magnitude-acn`

export const immutableBinaryPath = (dataDir: string, version: string): string =>
  join(dataDir, "bin", "acn", encodeURIComponent(version), platformArchTriple(), acnExecutableName())

export const cachedBinaryPath = (dataDir: string, version?: string): string =>
  version === undefined ? defaultBinaryPath(dataDir) : immutableBinaryPath(dataDir, version)

function isWindows(): boolean {
  return process.platform === "win32"
}

function acnExecutableName(): string {
  return isWindows() ? "magnitude-acn.exe" : "magnitude-acn"
}

function platformArchTriple(): string {
  const platform = process.platform
  const arch = process.arch

  switch (platform) {
    case "darwin":
      if (arch === "arm64") return "darwin-arm64"
      if (arch === "x64") return "darwin-x64"
      break
    case "linux":
      if (arch === "x64") return "linux-x64"
      if (arch === "arm64") return "linux-arm64"
      break
    case "win32":
      if (arch === "x64") return "windows-x64"
      break
  }

  throw new Error(`Unsupported platform/arch: ${platform} ${arch}`)
}

const RELEASE_REPO = "magnitudedev/magnitude"

export function releaseTag(version: string): string {
  return `@magnitudedev/cli@${version}`
}

export function releaseBaseUrl(): string {
  return (process.env.MAGNITUDE_RELEASE_BASE_URL ?? `https://github.com/${RELEASE_REPO}/releases/download`)
    .replace(/\/+$/, "")
}

export function acnAssetName(platformKey: string = platformArchTriple()): string {
  return `magnitude-acn-${platformKey}.tar.gz`
}

export function acnDownloadUrl(version: string, platformKey: string = platformArchTriple()): string {
  return `${releaseBaseUrl()}/${encodeURIComponent(releaseTag(version))}/${acnAssetName(platformKey)}`
}

function validateBinaryVersion(
  binaryPath: string,
  expectedVersion: string
): Effect.Effect<void, BinaryVersionMismatch | DaemonSpawnFailed, CommandExecutor.CommandExecutor> {
  return Effect.gen(function* () {
    const actual = yield* Command.make(binaryPath, "version").pipe(
      Command.string,
      Effect.map((value) => value.trim()),
      Effect.mapError((error) => new DaemonSpawnFailed({ reason: String(error) }))
    )
    if (actual !== expectedVersion) {
      return yield* new BinaryVersionMismatch({
        path: binaryPath,
        expected: expectedVersion,
        actual,
      })
    }
  })
}

const downloadBinary = (url: string): Effect.Effect<Uint8Array, DownloadFailed, HttpClient.HttpClient> =>
  Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    const request = HttpClientRequest.get(url)
    const response = yield* Effect.retry({
      schedule: Schedule.exponential("1 second").pipe(Schedule.intersect(Schedule.recurs(2)))
    })(client.execute(request)).pipe(
      Effect.timeout("30 seconds"),
      Effect.mapError((error) => new DownloadFailed({ url, status: 0, reason: String(error) }))
    )

    if (response.status < 200 || response.status >= 300) {
      return yield* new DownloadFailed({
        url,
        status: response.status,
        reason: `HTTP ${response.status}`
      })
    }

    const buffer = yield* response.arrayBuffer.pipe(
      Effect.mapError((error) => new DownloadFailed({ url, status: 0, reason: String(error) }))
    )
    return new Uint8Array(buffer)
  })

export const downloadAcn = (
  version: string,
  dataDir: string
): Effect.Effect<string, DownloadFailed | BinaryNotFound | BinaryVersionMismatch | DaemonSpawnFailed, FileSystem | HttpClient.HttpClient | CommandExecutor.CommandExecutor> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem
    const url = acnDownloadUrl(version)
    const binDir = join(dataDir, "bin")
    const tmpDir = join(dataDir, "downloads")
    const publicationId = yield* Effect.sync(() => crypto.randomUUID())
    const tmpFile = join(tmpDir, `magnitude-acn-${publicationId}.tar.gz.tmp`)
    const extractDir = join(binDir, `.tmp-${publicationId}`)
    const finalPath = immutableBinaryPath(dataDir, version)

    const existing = yield* fs.exists(finalPath).pipe(Effect.mapError(fsError(url)))
    if (existing) {
      yield* validateBinaryVersion(finalPath, version)
      return finalPath
    }

    yield* fs.makeDirectory(tmpDir, { recursive: true }).pipe(Effect.mapError(fsError(url)))
    yield* fs.makeDirectory(extractDir, { recursive: true }).pipe(Effect.mapError(fsError(url)))

    try {
      const bytes = yield* downloadBinary(url)
      yield* fs.writeFile(tmpFile, bytes).pipe(Effect.mapError(fsError(url)))

      const tarFlag = isWindows() ? "-xf" : "-xzf"
      const tarExit = yield* Command.make("tar", tarFlag, tmpFile, "-C", extractDir).pipe(
        Command.exitCode,
        Effect.mapError((error) => new DaemonSpawnFailed({ reason: String(error) }))
      )

      if (tarExit !== 0) {
        return yield* new DaemonSpawnFailed({ reason: `tar extraction failed (${tarExit})` })
      }

      const extractedPath = join(extractDir, acnExecutableName())
      const extractedExists = yield* fs.exists(extractedPath).pipe(Effect.mapError(fsError(url)))
      if (!extractedExists) {
        return yield* new BinaryNotFound({ path: extractedPath })
      }

      if (!isWindows()) {
        yield* fs.chmod(extractedPath, 0o755).pipe(
          Effect.mapError((error) => new DaemonSpawnFailed({ reason: String(error) }))
        )
      }

      yield* validateBinaryVersion(extractedPath, version)

      yield* fs.makeDirectory(dirname(finalPath), { recursive: true }).pipe(
        Effect.mapError(fsError(url)),
      )
      const publication = yield* fs
        .link(extractedPath, finalPath)
        .pipe(Effect.mapError(fsError(url)), Effect.either)
      if (publication._tag === "Left") {
        const published = yield* fs.exists(finalPath).pipe(Effect.mapError(fsError(url)))
        if (!published) return yield* publication.left
        yield* validateBinaryVersion(finalPath, version)
      }
      yield* fs.remove(extractDir, { recursive: true, force: true }).pipe(Effect.mapError(fsError(url)))

      return finalPath
    } finally {
      yield* fs.remove(tmpFile, { force: true }).pipe(Effect.catchAll(() => Effect.void))
      yield* fs.remove(extractDir, { recursive: true, force: true }).pipe(Effect.catchAll(() => Effect.void))
    }
  })

const fsError = (url: string) => (error: { readonly message: string }): DownloadFailed =>
  new DownloadFailed({ url, status: 0, reason: error.message })

export const resolveBinaryCommand = (
  options?: ResolveBinaryOptions
): Effect.Effect<ResolvedBinaryCommand, DownloadFailed | BinaryNotFound | BinaryVersionMismatch | DaemonSpawnFailed, FileSystem | HttpClient.HttpClient | CommandExecutor.CommandExecutor> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem
    const dataDir = options?.dataDir ?? defaultDataDir()
    const expectedVersion = options?.version

    if (options?.binaryPath) {
      const explicitExists = yield* fs.exists(options.binaryPath).pipe(Effect.catchAll(() => Effect.succeed(false)))
      if (!explicitExists) {
        return yield* new BinaryNotFound({ path: options.binaryPath })
      }
      if (expectedVersion) {
        yield* validateBinaryVersion(options.binaryPath, expectedVersion)
      }
      return {
        command: [options.binaryPath, "serve", "--register", "--data-dir", dataDir],
        needsDownload: false
      }
    }

    const cachedPath = cachedBinaryPath(dataDir, expectedVersion)
    const cachedExists = yield* fs.exists(cachedPath).pipe(Effect.catchAll(() => Effect.succeed(false)))

    if (expectedVersion) {
      if (cachedExists) {
        const cacheValid = yield* validateBinaryVersion(cachedPath, expectedVersion).pipe(
          Effect.as(true),
          Effect.catchAll(() => Effect.succeed(false))
        )
        if (cacheValid) {
          return {
            command: [cachedPath, "serve", "--register", "--data-dir", dataDir],
            needsDownload: false
          }
        }
      }
    } else if (cachedExists) {
      return {
        command: [cachedPath, "serve", "--register", "--data-dir", dataDir],
        needsDownload: false
      }
    }

    if (!expectedVersion) {
      return yield* new BinaryNotFound({ path: cachedPath })
    }

    const downloadedPath = yield* downloadAcn(expectedVersion, dataDir)
    return {
      command: [downloadedPath, "serve", "--register", "--data-dir", dataDir],
      needsDownload: true
    }
  })
