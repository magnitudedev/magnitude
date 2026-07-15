import { Effect, Schedule, Stream, type Scope } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import type { PlatformError } from "@effect/platform/Error"
import { LlamaCppBinaryDownloadFailed } from "../errors"
import { downloadUrl, type PlatformAsset } from "../platform"
import { buildNumberToTag } from "../version"
import { cachedBinaryDir, downloadTmpDir, llamacppDataDir, versionMarkerPath } from "../paths"
import { validateBinary } from "./validate"
import { DownloadResult } from "./types"

const RETRY_POLICY = Schedule.recurs(2).pipe(Schedule.addDelay(() => "2 seconds"))

/**
 * Download a specific build/asset from GitHub releases, extract, validate,
 * and move into the Magnitude-managed cache.
 */
export function downloadBinary(
  buildNumber: number,
  asset: PlatformAsset,
): Effect.Effect<
  DownloadResult,
  LlamaCppBinaryDownloadFailed,
  FileSystem.FileSystem | Path.Path | HttpClient.HttpClient | CommandExecutor.CommandExecutor
> {
  return Effect.scoped(downloadBinaryScoped(buildNumber, asset)).pipe(
    Effect.mapError((err) =>
      err instanceof LlamaCppBinaryDownloadFailed
        ? err
        : new LlamaCppBinaryDownloadFailed({ url: "", reason: String(err) })
    ),
  )
}

function downloadBinaryScoped(
  buildNumber: number,
  asset: PlatformAsset,
): Effect.Effect<
  DownloadResult,
  LlamaCppBinaryDownloadFailed | PlatformError,
  FileSystem.FileSystem | Path.Path | HttpClient.HttpClient | CommandExecutor.CommandExecutor | Scope.Scope
> {
  return Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const pathSvc = yield* Path.Path
    const client = yield* HttpClient.HttpClient

    const tag = buildNumberToTag(buildNumber)
    const url = downloadUrl(tag, asset)
    const targetDir = cachedBinaryDir(buildNumber)
    const targetPath = pathSvc.join(targetDir, "llama-server")

    const tmpDir = pathSvc.join(downloadTmpDir(), `.tmp-${buildNumber}-${Date.now()}`)

    yield* Effect.addFinalizer(() =>
      fs.remove(tmpDir, { recursive: true, force: true }).pipe(Effect.ignore),
    )

    yield* fs.makeDirectory(tmpDir, { recursive: true })

    const tmpArchive = pathSvc.join(tmpDir, "archive.tar.gz")

    yield* downloadTarball(client, fs, url, tmpArchive).pipe(
      Effect.retry(RETRY_POLICY),
      Effect.mapError((cause) =>
        new LlamaCppBinaryDownloadFailed({
          url,
          reason: cause instanceof Error ? cause.message : String(cause),
        }),
      ),
    )

    // Extract
    yield* Command.make("tar", "-xzf", tmpArchive, "-C", tmpDir).pipe(
      Command.exitCode,
      Effect.flatMap((code) =>
        code !== 0
          ? Effect.fail(new LlamaCppBinaryDownloadFailed({ url, reason: `tar extraction failed (${code})` }))
          : Effect.void,
      ),
    )

    // Find extracted directory
    const entries = yield* fs.readDirectory(tmpDir)
    const extractedDir = entries.find((e) => e.startsWith("llama-"))
    if (!extractedDir) {
      return yield* new LlamaCppBinaryDownloadFailed({ url, reason: "No llama-* directory in archive" })
    }

    const extractedServerPath = pathSvc.join(tmpDir, extractedDir, "llama-server")
    const exists = yield* fs.exists(extractedServerPath)
    if (!exists) {
      return yield* new LlamaCppBinaryDownloadFailed({ url, reason: "llama-server not found in archive" })
    }

    // Make executable
    yield* fs.chmod(extractedServerPath, 0o755).pipe(Effect.ignore)

    // Validate version
    const actualBuild = yield* validateBinary(extractedServerPath).pipe(
      Effect.mapError((err) =>
        new LlamaCppBinaryDownloadFailed({ url, reason: err.reason }),
      ),
    )
    if (actualBuild !== buildNumber) {
      return yield* new LlamaCppBinaryDownloadFailed({
        url,
        reason: `Version mismatch: expected ${buildNumber}, got ${actualBuild}`,
      })
    }

    // Remove quarantine on macOS
    if (process.platform === "darwin") {
      yield* Command.make("xattr", "-dr", "com.apple.quarantine", pathSvc.join(tmpDir, extractedDir)).pipe(
        Command.exitCode,
        Effect.ignore,
      )
    }

    // Move to final location
    yield* fs.makeDirectory(llamacppDataDir(), { recursive: true })
    const finalExists = yield* fs.exists(targetDir)
    if (finalExists) {
      yield* fs.remove(targetDir, { recursive: true, force: true })
    }
    yield* fs.rename(pathSvc.join(tmpDir, extractedDir), targetDir)

    // Write version marker
    yield* fs.writeFileString(versionMarkerPath(), String(buildNumber)).pipe(Effect.ignore)

    return { path: targetPath, directory: targetDir, buildNumber: actualBuild }
  })
}

function downloadTarball(
  client: HttpClient.HttpClient,
  fs: FileSystem.FileSystem,
  url: string,
  destination: string,
): Effect.Effect<void, unknown, Scope.Scope> {
  return Effect.gen(function* () {
    const file = yield* fs.open(destination, { flag: "w" })
    const response = yield* client.execute(HttpClientRequest.get(url)).pipe(
      Effect.flatMap(HttpClientResponse.filterStatusOk),
    )
    yield* response.stream.pipe(
      Stream.runForEach((chunk) => file.write(chunk as Uint8Array)),
    )
  })
}
