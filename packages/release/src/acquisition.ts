import { createHash } from "node:crypto"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as Path from "@effect/platform/Path"
import { Effect, Option, Schedule, Stream } from "effect"
import { ArchiveExtractor } from "./archive"
import { downloadArtifact as downloadReleaseArtifact } from "./artifact-download"
import { ReleaseAcquisitionError } from "./errors"
import {
  decodeReleaseManifest,
  type ReleaseArtifact,
  type ReleaseManifest,
} from "./contracts"

const MANIFEST_LIMIT = 16 * 1024 * 1024
const MAXIMUM_DOWNLOAD_CHUNK_SIZE = 16 * 1024 * 1024
const PARALLEL_DOWNLOAD_CONCURRENCY = 4

export interface AcquiredRelease {
  readonly manifest: ReleaseManifest
  readonly manifestSha256: string
}

const failure = (
  stage: ReleaseAcquisitionError["stage"],
  message: string,
  transient = false,
) => new ReleaseAcquisitionError({ stage, message, transient })

const releaseTag = (version: string) => `@magnitudedev/cli@${version}`

export const releaseUrl = (
  baseUrl: string,
  version: string,
  filename: string,
) => `${baseUrl.replace(/\/+$/, "")}/${releaseTag(version).split("/").map(encodeURIComponent).join("/")}/${encodeURIComponent(filename)}`

const transientStatus = (status: number) =>
  status === 408 || status === 429 || status >= 500

const collectBounded = (
  stream: Stream.Stream<Uint8Array, unknown>,
  maximumBytes: number,
): Effect.Effect<Uint8Array, ReleaseAcquisitionError> =>
  stream.pipe(
    Stream.runFoldEffect(
      { chunks: [] as Uint8Array[], bytes: 0 },
      (state, chunk) => {
        const bytes = state.bytes + chunk.byteLength
        return bytes > maximumBytes
          ? Effect.fail(failure("download", "release response exceeds its size bound"))
          : Effect.succeed({
            chunks: [...state.chunks, chunk],
            bytes,
          })
      },
    ),
    Effect.map(({ chunks, bytes }) => {
      const value = new Uint8Array(bytes)
      let offset = 0
      for (const chunk of chunks) {
        value.set(chunk, offset)
        offset += chunk.byteLength
      }
      return value
    }),
    Effect.mapError((cause) =>
      cause instanceof ReleaseAcquisitionError
        ? cause
        : failure("download", "release response stream failed", true)
    ),
  )

const getBounded = (
  url: string,
  maximumBytes: number,
): Effect.Effect<Uint8Array, ReleaseAcquisitionError, HttpClient.HttpClient> =>
  Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    const response = yield* client.execute(HttpClientRequest.get(url)).pipe(
      Effect.mapError(() => failure("download", "release request failed", true)),
    )
    if (response.status < 200 || response.status >= 300) {
      return yield* failure(
        "download",
        `release endpoint returned HTTP ${response.status}`,
        transientStatus(response.status),
      )
    }
    const declared = Number(response.headers["content-length"])
    if (Number.isFinite(declared) && declared > maximumBytes) {
      return yield* failure("download", "release response exceeds its size bound")
    }
    return yield* collectBounded(response.stream, maximumBytes)
  }).pipe(
    Effect.timeoutFail({
      duration: "30 seconds",
      onTimeout: () => failure("download", "release response timed out", true),
    }),
    Effect.retry({
      while: (error) => error.transient,
      schedule: Schedule.exponential("500 millis").pipe(
        Schedule.jittered,
        Schedule.intersect(Schedule.recurs(2)),
      ),
    }),
  )

export const validateReleaseManifestBytes = (
  manifestBytes: Uint8Array,
): Effect.Effect<AcquiredRelease, ReleaseAcquisitionError> =>
  Effect.gen(function* () {
    const manifest = yield* decodeReleaseManifest(manifestBytes).pipe(
      Effect.mapError((error) => failure("validate", error.message)),
    )
    return {
      manifest,
      manifestSha256: createHash("sha256").update(manifestBytes).digest("hex"),
    }
  })

const readCachedManifest = (
  directory: string,
): Effect.Effect<
  Option.Option<Uint8Array>,
  never,
  FileSystem.FileSystem | Path.Path
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const manifest = yield* fs.readFile(
      path.join(directory, "magnitude-release.json"),
    ).pipe(Effect.option)
    if (Option.isNone(manifest)) return Option.none()
    return manifest.value.byteLength <= MANIFEST_LIMIT
      ? manifest
      : Option.none()
  })

const publishCachedManifest = (
  directory: string,
  manifest: Uint8Array,
): Effect.Effect<void, never, FileSystem.FileSystem | Path.Path> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const parent = path.dirname(directory)
    yield* fs.makeDirectory(parent, { recursive: true, mode: 0o700 })
    const staging = yield* fs.makeTempDirectory({
      directory: parent,
      prefix: ".manifest-",
    })
    yield* Effect.gen(function* () {
      yield* fs.writeFile(path.join(staging, "magnitude-release.json"), manifest, {
        flag: "wx",
        mode: 0o600,
      })
      yield* fs.rename(staging, directory).pipe(
        Effect.catchAll(() => Effect.void),
      )
    }).pipe(
      Effect.ensuring(
        fs.remove(staging, { recursive: true, force: true }).pipe(
          Effect.catchAll(() => Effect.void),
        ),
      ),
    )
  }).pipe(Effect.catchAll(() => Effect.void))

export const acquireRelease = (
  baseUrl: string,
  version: string,
  cacheDirectory: string,
): Effect.Effect<
  AcquiredRelease,
  ReleaseAcquisitionError,
  FileSystem.FileSystem | Path.Path | HttpClient.HttpClient
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const cached = yield* readCachedManifest(cacheDirectory)
    if (Option.isSome(cached)) {
      const acquired = yield* validateReleaseManifestBytes(cached.value).pipe(
        Effect.option,
      )
      if (
        Option.isSome(acquired) &&
        acquired.value.manifest.version === version &&
        acquired.value.manifest.tag === releaseTag(version)
      ) {
        return acquired.value
      }
    }
    yield* fs.remove(cacheDirectory, { recursive: true, force: true }).pipe(
      Effect.catchAll(() => Effect.void),
    )
    const manifestUrl = releaseUrl(baseUrl, version, "magnitude-release.json")
    const manifestBytes = yield* getBounded(manifestUrl, MANIFEST_LIMIT)
    const acquired = yield* validateReleaseManifestBytes(manifestBytes)
    if (
      acquired.manifest.version !== version ||
      acquired.manifest.tag !== releaseTag(version)
    ) return yield* failure("validate", "release identity differs from requested version")
    yield* publishCachedManifest(cacheDirectory, manifestBytes)
    return acquired
  })

export const selectArtifact = (
  manifest: ReleaseManifest,
  kind: ReleaseArtifact["kind"],
  host?: string,
  backend?: string,
): Effect.Effect<ReleaseArtifact, ReleaseAcquisitionError> => {
  const matches = manifest.artifacts.filter((artifact) =>
    artifact.kind === kind &&
    (host === undefined || Option.getOrUndefined(artifact.host) === host) &&
    (backend === undefined || Option.getOrUndefined(artifact.backend) === backend)
  )
  return matches.length === 1
    ? Effect.succeed(matches[0]!)
    : Effect.fail(failure(
      "validate",
      `release has ${matches.length} matching ${kind} artifacts`,
    ))
}

export const installArtifact = (
  baseUrl: string,
  version: string,
  artifact: ReleaseArtifact,
  destination: string,
): Effect.Effect<
  string,
  ReleaseAcquisitionError,
  FileSystem.FileSystem | Path.Path | HttpClient.HttpClient | ArchiveExtractor
> =>
  Effect.scoped(
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const extractor = yield* ArchiveExtractor
      const existing = yield* fs.stat(destination).pipe(Effect.option)
      if (Option.isSome(existing) && existing.value.type === "Directory") return destination
      const parent = path.dirname(destination)
      yield* fs.makeDirectory(parent, { recursive: true, mode: 0o700 }).pipe(
        Effect.mapError(() => failure("install", "unable to create release directory")),
      )
      const downloadDirectory = yield* fs.makeTempDirectoryScoped({
        directory: parent,
        prefix: ".download-",
      }).pipe(
        Effect.mapError(() =>
          failure("install", "unable to create release download directory")
        ),
      )
      const archive = path.join(downloadDirectory, "artifact")
      yield* downloadReleaseArtifact({
        url: releaseUrl(baseUrl, version, artifact.filename),
        destination: archive,
        bytes: artifact.bytes,
        sha256: artifact.sha256,
        strategy: {
          _tag: "Segmented",
          concurrency: PARALLEL_DOWNLOAD_CONCURRENCY,
          chunkBytes: Math.min(
            MAXIMUM_DOWNLOAD_CHUNK_SIZE,
            Math.ceil(artifact.bytes / PARALLEL_DOWNLOAD_CONCURRENCY),
          ),
          fallbackToSequential: true,
        },
      }).pipe(
        Effect.mapError((error) =>
          failure("download", error.message, error.transient)
        ),
      )
      const staging = yield* fs.makeTempDirectory({
        directory: parent,
        prefix: ".release-",
      }).pipe(
        Effect.mapError(() => failure("install", "unable to create extraction staging directory")),
      )
      yield* Effect.gen(function* () {
        yield* extractor.extract(archive, staging, artifact)
        yield* fs.rename(staging, destination).pipe(
          Effect.catchAll((cause) =>
            fs.stat(destination).pipe(
              Effect.flatMap((info) =>
                info.type === "Directory"
                  ? Effect.void
                  : Effect.fail(cause)
              ),
              Effect.mapError(() => cause),
            )
          ),
          Effect.mapError(() =>
            failure("install", "unable to publish release installation")
          ),
        )
      }).pipe(
        Effect.ensuring(
          fs.remove(staging, { recursive: true, force: true }).pipe(
            Effect.catchAll(() => Effect.void),
          ),
        ),
      )
      return destination
    }),
  )
