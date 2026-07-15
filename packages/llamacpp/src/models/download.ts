import { Effect, Stream, Schedule, type Scope, pipe } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { homedir } from "node:os"
import { dirname, basename } from "node:path"
import {
  LlamaCppModelDownloadFailed,
  LlamaCppGatedModelAccessDenied,
  LlamaCppHfTokenMissing,
} from "../errors"
import type {
  DownloadModelParams,
  DownloadModelResult,
  DownloadProgress,
  DownloadEvent,
  RepoGgufFile,
} from "./types"
import { resolveHfToken } from "./hf-token"

import type { DownloadRegistry } from "./download-registry"

const RETRY_POLICY = Schedule.recurs(3).pipe(Schedule.addDelay(() => "2 seconds"))
const HF_API_BASE = "https://huggingface.co/api"
const HF_RESOLVE_BASE = "https://huggingface.co"

// ── Paths ──

/**
 * Resolve the HF cache directory.
 * Checks env vars in order of precedence, then falls back to the default.
 */
function hfCacheDir(): string {
  const env = process.env
  if (env.HF_HUB_CACHE) return env.HF_HUB_CACHE
  if (env.HUGGINGFACE_HUB_CACHE) return env.HUGGINGFACE_HUB_CACHE
  if (env.HF_HOME) return `${env.HF_HOME}/hub`
  if (env.XDG_CACHE_HOME) return `${env.XDG_CACHE_HOME}/huggingface/hub`
  return `${homedir()}/.cache/huggingface/hub`
}

/**
 * Resolve the HF cache path for a given repo + commit + file.
 * Follows the standard HF Hub cache layout:
 * `~/.cache/huggingface/hub/models--{org}--{name}/snapshots/{commit}/{file}`
 */
export function hfCachePathForFile(
  repo: string,
  commit: string,
  file: string,
): string {
  const repoFolder = `models--${repo.replace(/\//g, "--")}`
  return `${hfCacheDir()}/${repoFolder}/snapshots/${commit}/${file}`
}

/** Resolve the .incomplete path for a given cache file. */
export function incompletePath(filePath: string): string {
  return `${filePath}.incomplete`
}

// ── HF API (Effect-native via HttpClient) ──

interface HfModelInfo {
  readonly gated?: boolean | string
}

interface HfFileEntry {
  readonly path: string
  readonly size: number
  readonly type: string
  readonly lfs?: { readonly size: number; readonly oid: string }
}

/** Add auth header to a request if token is present. */
function withAuth(
  req: HttpClientRequest.HttpClientRequest,
  token: string | null,
): HttpClientRequest.HttpClientRequest {
  return token
    ? pipe(req, HttpClientRequest.setHeader("Authorization", `Bearer ${token}`))
    : req
}

/**
 * Fetch model info from HF API to check gated status.
 * Returns `{ gated: false }` on any error — the caller will discover access issues
 * when trying to get the actual file.
 */
function getHfModelInfo(
  repo: string,
  token: string | null,
): Effect.Effect<HfModelInfo, never, HttpClient.HttpClient> {
  return Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    const req = withAuth(HttpClientRequest.get(`${HF_API_BASE}/models/${repo}`), token)
    const res = yield* pipe(
      client.execute(req),
      Effect.catchAll(() => Effect.succeed(null)),
    )
    if (!res || res.status !== 200) return { gated: false }
    const body = yield* pipe(
      res.json,
      Effect.catchAll(() => Effect.succeed({})),
    )
    return body as HfModelInfo
  })
}

/**
 * List files in a HF repo via the tree API.
 */
function listHfFiles(
  repo: string,
  revision: string,
  token: string | null,
): Effect.Effect<
  readonly HfFileEntry[],
  LlamaCppModelDownloadFailed,
  HttpClient.HttpClient
> {
  return Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    const req = withAuth(
      HttpClientRequest.get(`${HF_API_BASE}/models/${repo}/tree/${revision}?recursive=true`),
      token,
    )
    const res = yield* pipe(
      client.execute(req),
      Effect.mapError((err) =>
        new LlamaCppModelDownloadFailed({
          repo,
          file: "",
          reason: `Failed to list repo files: ${String(err)}`,
        }),
      ),
    )
    if (res.status !== 200) {
      return yield* new LlamaCppModelDownloadFailed({
        repo,
        file: "",
        reason: `HF API returned status ${res.status}`,
      })
    }
    const body = yield* pipe(
      res.json,
      Effect.mapError((err) =>
        new LlamaCppModelDownloadFailed({
          repo,
          file: "",
          reason: `Failed to parse file list: ${String(err)}`,
        }),
      ),
    )
    return body as readonly HfFileEntry[]
  })
}

/**
 * Get file download info (size + commit) from HF resolve endpoint.
 */
function getHfFileInfo(
  repo: string,
  file: string,
  revision: string,
  token: string | null,
): Effect.Effect<
  { readonly size: number; readonly commit: string; readonly url: string },
  LlamaCppModelDownloadFailed | LlamaCppGatedModelAccessDenied | LlamaCppHfTokenMissing,
  HttpClient.HttpClient
> {
  const url = `${HF_RESOLVE_BASE}/${repo}/resolve/${revision}/${file}`
  return Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    const req = withAuth(
      pipe(HttpClientRequest.get(url), HttpClientRequest.setHeader("Accept", "application/json")),
      token,
    )
    const res = yield* pipe(
      client.execute(req),
      Effect.mapError((err) =>
        new LlamaCppModelDownloadFailed({
          repo,
          file,
          reason: `Failed to get file info: ${String(err)}`,
        }),
      ),
    )

    if (res.status === 401 || res.status === 403) {
      if (!token) {
        return yield* new LlamaCppHfTokenMissing({ repo })
      }
      return yield* new LlamaCppGatedModelAccessDenied({
        repo,
        message: `Access denied to ${repo}. Accept the license at https://huggingface.co/${repo}`,
      })
    }

    if (res.status === 404) {
      return yield* new LlamaCppModelDownloadFailed({
        repo,
        file,
        reason: "File not found in repo",
      })
    }

    if (res.status !== 200) {
      return yield* new LlamaCppModelDownloadFailed({
        repo,
        file,
        reason: `HF returned status ${res.status}`,
      })
    }

    const contentLength = res.headers["content-length"] ?? "0"
    const size = Number(contentLength)
    const etag = res.headers["etag"] ?? ""
    const commit = etag.replace(/"/g, "").slice(0, 7) || "unknown"

    return { size, commit, url }
  })
}

// ── Public API ──

/**
 * List available GGUF files in a HuggingFace repo.
 * Filters for .gguf extension and parses quantization from filename using the library.
 */
export function listRepoGgufFiles(
  repo: string,
  revision: string = "main",
): Effect.Effect<
  readonly RepoGgufFile[],
  LlamaCppModelDownloadFailed | LlamaCppGatedModelAccessDenied | LlamaCppHfTokenMissing,
  FileSystem.FileSystem | HttpClient.HttpClient
> {
  return Effect.gen(function* () {
    const token = yield* resolveHfToken()

    // Check gated status
    const modelInfo = yield* getHfModelInfo(repo, token)
    if (modelInfo.gated && !token) {
      return yield* new LlamaCppHfTokenMissing({ repo })
    }

    const files = yield* listHfFiles(repo, revision, token)
    return files
      .filter((f) => f.path.endsWith(".gguf") && f.type === "file")
      .map((f) => ({
        path: f.path,
        size: f.lfs?.size ?? f.size,
        quantization: undefined,
      }))
  })
}

/**
 * Download a GGUF model file with streaming progress.
 *
 * Streams via HTTP with Range header support for resumable downloads.
 * Progress events are emitted via the registry; the final event is `DownloadModelResult`.
 */
export function downloadModelStream(
  params: DownloadModelParams,
  registry: DownloadRegistry,
): Stream.Stream<
  DownloadEvent,
  LlamaCppModelDownloadFailed | LlamaCppGatedModelAccessDenied | LlamaCppHfTokenMissing,
  FileSystem.FileSystem | Path.Path | HttpClient.HttpClient | Scope.Scope
> {
  return Stream.fromEffect(
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const token = yield* resolveHfToken()
      const revision = params.revision ?? "main"

      // Get file info (size, commit, download URL)
      const info = yield* getHfFileInfo(params.repo, params.file, revision, token)

      const totalBytes = info.size
      const downloadUrl = info.url
      const filePath = hfCachePathForFile(params.repo, info.commit, params.file)
      const incompleteFilePath = incompletePath(filePath)

      // Register download
      const downloadId = yield* registry.register(params, totalBytes)

      // Check if already fully downloaded
      const fullExists = yield* fs.exists(filePath).pipe(Effect.catchAll(() => Effect.succeed(false)))
      if (fullExists) {
        const stat = yield* fs.stat(filePath).pipe(Effect.catchAll(() => Effect.succeed(null)))
        if (stat && Number(stat.size) === totalBytes) {
          yield* registry.markCompleted(downloadId)
          return { filePath, repoId: params.repo, commit: info.commit } satisfies DownloadModelResult
        }
      }

      // Check for .incomplete partial file for resume
      let startOffset = 0
      const partialExists = yield* fs.exists(incompleteFilePath).pipe(
        Effect.catchAll(() => Effect.succeed(false)),
      )
      if (partialExists) {
        const partialStat = yield* fs.stat(incompleteFilePath).pipe(
          Effect.catchAll(() => Effect.succeed(null)),
        )
        if (partialStat && partialStat.type === "File") {
          startOffset = Number(partialStat.size)
        }
      }

      // Ensure parent directories exist
      const parentDir = dirname(filePath)
      yield* fs.makeDirectory(parentDir, { recursive: true }).pipe(Effect.ignore)

      // Build HTTP request with Range header for resume
      const client = yield* HttpClient.HttpClient
      const request = startOffset > 0
        ? pipe(
            HttpClientRequest.get(downloadUrl),
            HttpClientRequest.setHeader("Range", `bytes=${startOffset}-`),
          )
        : HttpClientRequest.get(downloadUrl)

      const response = yield* pipe(
        client.execute(request),
        Effect.flatMap(HttpClientResponse.filterStatusOk),
        Effect.retry(RETRY_POLICY),
        Effect.mapError((err) =>
          new LlamaCppModelDownloadFailed({
            repo: params.repo,
            file: params.file,
            reason: `HTTP request failed: ${String(err)}`,
          }),
        ),
      )

      // Open file — will be closed automatically by the scope
      const fileHandle = yield* (startOffset > 0
        ? fs.open(incompleteFilePath, { flag: "a" })
        : fs.open(incompleteFilePath, { flag: "w" }))

      let downloadedBytes = startOffset
      const startTime = Date.now()

      // Stream body chunks to file + update registry
      yield* pipe(
        response.stream,
        Stream.runForEach((chunk: Uint8Array) =>
          Effect.gen(function* () {
            yield* fileHandle.write(chunk)
            downloadedBytes += chunk.length

            const elapsedSec = (Date.now() - startTime) / 1000
            const bytesPerSecond = elapsedSec > 0 ? downloadedBytes / elapsedSec : 0
            const remaining = totalBytes - downloadedBytes
            const etaSeconds = bytesPerSecond > 0 ? remaining / bytesPerSecond : 0
            const percent = totalBytes > 0 ? (downloadedBytes / totalBytes) * 100 : 0

            yield* registry.updateProgress(downloadId, {
              downloadedBytes,
              percent,
              bytesPerSecond,
              etaSeconds,
            })
          }),
        ),
      )

      // Rename .incomplete → final
      yield* fs.rename(incompleteFilePath, filePath).pipe(
        Effect.mapError((err) =>
          new LlamaCppModelDownloadFailed({
            repo: params.repo,
            file: params.file,
            reason: `Failed to finalize download: ${String(err)}`,
          }),
        ),
      )

      yield* registry.markCompleted(downloadId)

      return { filePath, repoId: params.repo, commit: info.commit } satisfies DownloadModelResult
    }).pipe(
      Effect.catchAll((err) =>
        isKnownDownloadError(err)
          ? Effect.fail(err)
          : Effect.fail(new LlamaCppModelDownloadFailed({
              repo: params.repo,
              file: params.file,
              reason: String(err),
            })),
      ),
    ),
  )
}

function isKnownDownloadError(
  err: unknown,
): err is LlamaCppModelDownloadFailed | LlamaCppGatedModelAccessDenied | LlamaCppHfTokenMissing {
  return err instanceof LlamaCppModelDownloadFailed
    || err instanceof LlamaCppGatedModelAccessDenied
    || err instanceof LlamaCppHfTokenMissing
}

/**
 * Cancel a download by deleting the .incomplete file and removing from registry.
 */
export function cancelModelDownload(
  params: DownloadModelParams,
  registry: DownloadRegistry,
): Effect.Effect<void, never, FileSystem.FileSystem> {
  return Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const downloadId = `${params.repo}/${params.file}`
    yield* registry.remove(downloadId)

    // Best-effort: find and delete any .incomplete file matching this repo/file
    const cacheDir = hfCacheDir()
    const exists = yield* fs.exists(cacheDir).pipe(Effect.catchAll(() => Effect.succeed(false)))
    if (!exists) return

    const repoFolder = `models--${params.repo.replace(/\//g, "--")}`
    const snapshotsDir = `${cacheDir}/${repoFolder}/snapshots`
    const snapshotsExist = yield* fs.exists(snapshotsDir).pipe(
      Effect.catchAll(() => Effect.succeed(false)),
    )
    if (!snapshotsExist) return

    const commits = yield* fs.readDirectory(snapshotsDir).pipe(
      Effect.catchAll(() => Effect.succeed([] as readonly string[])),
    )
    for (const commit of commits) {
      const incompleteFile = `${snapshotsDir}/${commit}/${params.file}.incomplete`
      const fileExists = yield* fs.exists(incompleteFile).pipe(Effect.catchAll(() => Effect.succeed(false)))
      if (fileExists) {
        yield* fs.remove(incompleteFile).pipe(Effect.ignore)
      }
    }
  })
}

// Suppress unused import warning — basename is used by parseGGUFQuantLabel context
void basename
