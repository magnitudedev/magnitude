import { createHash } from "node:crypto"
import { statfs } from "node:fs/promises"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as Path from "@effect/platform/Path"
import { Effect, Option, Schedule, Stream } from "effect"
import { ArchiveExtractor } from "./archive"
import {
  defaultArtifactDownloadPolicy,
  downloadArtifact as downloadReleaseArtifact,
} from "./artifact-download"
import type { ArtifactInstallationObserver } from "./installation-progress"
import { ReleaseAcquisitionError, ReleaseDiskFullError } from "./errors"
import {
  decodeReleaseManifest,
  releaseTag,
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

export interface ReleaseBundleSizes {
  readonly daemonBytes: number
  readonly inferenceEngineBytes: number
  readonly inferenceEngineBytesExact: boolean
}

export interface InstallArtifactOptions {
  readonly observer: Option.Option<ArtifactInstallationObserver>
}

const NoArtifactInstallationObserver: InstallArtifactOptions = {
  observer: Option.none(),
}

const failure = (
  stage: ReleaseAcquisitionError["stage"],
  message: string,
  transient = false
) => new ReleaseAcquisitionError({ stage, message, transient })

export const releaseBaseUrl = (): string => (
  process.env.MAGNITUDE_RELEASE_BASE_URL ??
  "https://github.com/magnitudedev/magnitude/releases/download"
).replace(/\/+$/, "")

export const releaseUrl = (
  baseUrl: string,
  version: string,
  filename: string
) =>
  `${baseUrl.replace(/\/+$/, "")}/${releaseTag(version)
    .split("/")
    .map(encodeURIComponent)
    .join("/")}/${encodeURIComponent(filename)}`

const transientStatus = (status: number) =>
  status === 408 || status === 429 || status >= 500

const collectBounded = <E>(
  stream: Stream.Stream<Uint8Array, E>,
  maximumBytes: number
): Effect.Effect<Uint8Array, ReleaseAcquisitionError> =>
  stream.pipe(
    Stream.mapError(() =>
      failure("download", "release response stream failed", true)
    ),
    Stream.runFoldEffect(
      { chunks: [] as Uint8Array[], bytes: 0 },
      (state, chunk) => {
        const bytes = state.bytes + chunk.byteLength
        return bytes > maximumBytes
          ? Effect.fail(
              failure("download", "release response exceeds its size bound")
            )
          : Effect.succeed({
              chunks: [...state.chunks, chunk],
              bytes,
            })
      }
    ),
    Effect.map(({ chunks, bytes }) => {
      const value = new Uint8Array(bytes)
      let offset = 0
      for (const chunk of chunks) {
        value.set(chunk, offset)
        offset += chunk.byteLength
      }
      return value
    })
  )

const getBounded = (
  url: string,
  maximumBytes: number
): Effect.Effect<Uint8Array, ReleaseAcquisitionError, HttpClient.HttpClient> =>
  Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    const response = yield* client
      .execute(HttpClientRequest.get(url))
      .pipe(
        Effect.mapError(() =>
          failure("download", "release request failed", true)
        )
      )
    if (response.status < 200 || response.status >= 300) {
      return yield* failure(
        "download",
        `release endpoint returned HTTP ${response.status}`,
        transientStatus(response.status)
      )
    }
    const declared = Number(response.headers["content-length"])
    if (Number.isFinite(declared) && declared > maximumBytes) {
      return yield* failure(
        "download",
        "release response exceeds its size bound"
      )
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
        Schedule.intersect(Schedule.recurs(2))
      ),
    })
  )

export const validateReleaseManifestBytes = (
  manifestBytes: Uint8Array
): Effect.Effect<AcquiredRelease, ReleaseAcquisitionError> =>
  Effect.gen(function* () {
    const manifest = yield* decodeReleaseManifest(manifestBytes).pipe(
      Effect.mapError((error) => failure("validate", error.message))
    )
    return {
      manifest,
      manifestSha256: createHash("sha256").update(manifestBytes).digest("hex"),
    }
  })

const readCachedManifest = (
  directory: string
): Effect.Effect<
  Option.Option<Uint8Array>,
  never,
  FileSystem.FileSystem | Path.Path
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const manifest = yield* fs
      .readFile(path.join(directory, "magnitude-release.json"))
      .pipe(Effect.option)
    if (Option.isNone(manifest)) return Option.none()
    return manifest.value.byteLength <= MANIFEST_LIMIT
      ? manifest
      : Option.none()
  })

const publishCachedManifest = (
  directory: string,
  manifest: Uint8Array
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
      yield* fs.writeFile(
        path.join(staging, "magnitude-release.json"),
        manifest,
        {
          flag: "wx",
          mode: 0o600,
        }
      )
      yield* fs
        .rename(staging, directory)
        .pipe(Effect.catchAll(() => Effect.void))
    }).pipe(
      Effect.ensuring(
        fs
          .remove(staging, { recursive: true, force: true })
          .pipe(Effect.catchAll(() => Effect.void))
      )
    )
  }).pipe(Effect.catchAll(() => Effect.void))

export const acquireRelease = (
  baseUrl: string,
  version: string,
  cacheDirectory: string
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
        Effect.option
      )
      if (
        Option.isSome(acquired) &&
        acquired.value.manifest.version === version &&
        acquired.value.manifest.tag === releaseTag(version)
      ) {
        return acquired.value
      }
    }
    yield* fs
      .remove(cacheDirectory, { recursive: true, force: true })
      .pipe(Effect.catchAll(() => Effect.void))
    const manifestUrl = releaseUrl(baseUrl, version, "magnitude-release.json")
    const manifestBytes = yield* getBounded(manifestUrl, MANIFEST_LIMIT)
    const acquired = yield* validateReleaseManifestBytes(manifestBytes)
    if (
      acquired.manifest.version !== version ||
      acquired.manifest.tag !== releaseTag(version)
    )
      return yield* failure(
        "validate",
        "release identity differs from requested version"
      )
    yield* publishCachedManifest(cacheDirectory, manifestBytes)
    return acquired
  })

export const selectArtifact = (
  manifest: ReleaseManifest,
  kind: ReleaseArtifact["kind"],
  host?: string,
): Effect.Effect<ReleaseArtifact, ReleaseAcquisitionError> => {
  const matches = manifest.artifacts.filter(
    (artifact) =>
      artifact.kind === kind &&
      (host === undefined || Option.getOrUndefined(artifact.host) === host)
  )
  return matches.length === 1
    ? Effect.succeed(matches[0]!)
    : Effect.fail(
        failure(
          "validate",
          `release has ${matches.length} matching ${kind} artifacts`
        )
      )
}

/** Each host has one inference artifact with every backend, so its size is known before download. */
export const releaseBundleSizes = (
  manifest: ReleaseManifest,
  host: string,
): Effect.Effect<ReleaseBundleSizes, ReleaseAcquisitionError> =>
  Effect.gen(function* () {
    const daemon = yield* selectArtifact(manifest, "acn", host)
    const inference = yield* selectArtifact(manifest, "icn-base", host)
    return {
      daemonBytes: daemon.bytes,
      inferenceEngineBytes: inference.bytes,
      inferenceEngineBytesExact: true,
    }
  })

const freeBytes = (directory: string): Effect.Effect<Option.Option<number>> =>
  Effect.tryPromise(() => statfs(directory)).pipe(
    Effect.map((stats) => Option.some(stats.bavail * stats.bsize)),
    Effect.orElseSucceed(() => Option.none<number>())
  )

/** A gzip archive records its expanded size, modulo 2^32, in its last four bytes. */
const gzipTrailerSize = (trailer: Uint8Array): Option.Option<number> =>
  trailer.byteLength === 4
    ? Option.some(new DataView(trailer.buffer, trailer.byteOffset, 4).getUint32(0, true))
    : Option.none()

const localExpandedBytes = (
  archive: string,
  bytes: number
): Effect.Effect<Option.Option<number>, never, FileSystem.FileSystem> =>
  Effect.scoped(
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const file = yield* fs.open(archive, { flag: "r" })
      yield* file.seek(bytes - 4, "start")
      const trailer = new Uint8Array(4)
      const read = yield* file.read(trailer)
      return read === BigInt(4) ? gzipTrailerSize(trailer) : Option.none<number>()
    })
  ).pipe(Effect.orElseSucceed(() => Option.none<number>()))

/** Reads the archive's gzip trailer with one range request, so the space check precedes the download. */
const remoteExpandedBytes = (
  url: string,
  bytes: number
): Effect.Effect<Option.Option<number>, never, HttpClient.HttpClient> =>
  Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    const response = yield* client.execute(
      HttpClientRequest.get(url).pipe(
        HttpClientRequest.setHeader("range", `bytes=${bytes - 4}-${bytes - 1}`)
      )
    )
    if (response.status !== 206 || response.headers["content-range"] !== `bytes ${bytes - 4}-${bytes - 1}/${bytes}`)
      return Option.none<number>()
    return gzipTrailerSize(new Uint8Array(yield* response.arrayBuffer))
  }).pipe(
    Effect.scoped,
    Effect.timeout("15 seconds"),
    Effect.orElseSucceed(() => Option.none<number>())
  )

/**
 * Fails when the disk can't hold `neededBytes` more. `reclaimedBytes` is space this installation
 * already holds and releases on failure, so the reported free space is what the person will see.
 */
const requireFreeSpace = (
  directory: string,
  neededBytes: number,
  requiredBytes: Option.Option<number>,
  reclaimedBytes: number
): Effect.Effect<void, ReleaseDiskFullError> =>
  freeBytes(directory).pipe(
    Effect.flatMap((free) =>
      Option.isSome(free) && free.value < neededBytes
        ? Effect.fail(new ReleaseDiskFullError({
            directory,
            requiredBytes,
            availableBytes: Option.some(free.value + reclaimedBytes),
          }))
        : Effect.void
    )
  )

export const installArtifact = (
  baseUrl: string,
  version: string,
  artifact: ReleaseArtifact,
  destination: string,
  options: InstallArtifactOptions = NoArtifactInstallationObserver
): Effect.Effect<
  string,
  ReleaseAcquisitionError | ReleaseDiskFullError,
  FileSystem.FileSystem | Path.Path | HttpClient.HttpClient | ArchiveExtractor
> =>
  Effect.scoped(
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const extractor = yield* ArchiveExtractor
      const existing = yield* fs.stat(destination).pipe(Effect.option)
      if (Option.isSome(existing) && existing.value.type === "Directory")
        return destination
      const parent = path.dirname(destination)
      yield* fs
        .makeDirectory(parent, { recursive: true, mode: 0o700 })
        .pipe(
          Effect.mapError(() =>
            failure("install", "unable to create release directory")
          )
        )
      const downloadDirectory = yield* fs
        .makeTempDirectoryScoped({
          directory: parent,
          prefix: ".download-",
        })
        .pipe(
          Effect.mapError(() =>
            failure("install", "unable to create release download directory")
          )
        )
      const archive = path.join(downloadDirectory, "artifact")
      const url = releaseUrl(baseUrl, version, artifact.filename)
      const announced = artifact.bytes > 4
        ? yield* remoteExpandedBytes(url, artifact.bytes)
        : Option.none<number>()
      yield* requireFreeSpace(
        parent,
        artifact.bytes + Option.getOrElse(announced, () => 0),
        Option.map(announced, (expanded) => artifact.bytes + expanded),
        0
      )
      yield* downloadReleaseArtifact({
        url,
        destination: archive,
        bytes: artifact.bytes,
        sha256: artifact.sha256,
        strategy: {
          _tag: "Segmented",
          concurrency: PARALLEL_DOWNLOAD_CONCURRENCY,
          chunkBytes: Math.min(
            MAXIMUM_DOWNLOAD_CHUNK_SIZE,
            Math.ceil(artifact.bytes / PARALLEL_DOWNLOAD_CONCURRENCY)
          ),
          fallbackToSequential: true,
        },
        policy: defaultArtifactDownloadPolicy,
        onProgress: Option.map(
          options.observer,
          (observer) => (progress) =>
            observer.report({ _tag: "Downloading", progress })
        ),
        onVerificationProgress: Option.map(
          options.observer,
          (observer) => (progress) =>
            observer.report({ _tag: "Verifying", progress })
        ),
      }).pipe(
        Effect.mapError((error) =>
          failure("download", error.message, error.transient)
        )
      )
      const expanded = artifact.bytes > 4
        ? yield* localExpandedBytes(archive, artifact.bytes)
        : Option.none<number>()
      if (Option.isSome(expanded)) {
        yield* requireFreeSpace(parent, expanded.value, Option.some(artifact.bytes + expanded.value), artifact.bytes)
      }
      const staging = yield* fs
        .makeTempDirectory({
          directory: parent,
          prefix: ".release-",
        })
        .pipe(
          Effect.mapError(() =>
            failure("install", "unable to create extraction staging directory")
          )
        )
      yield* Effect.gen(function* () {
        yield* extractor.extract(
          archive,
          staging,
          artifact,
          Option.map(
            options.observer,
            (observer) => (progress) =>
              observer.report({ _tag: "Extracting", progress })
          )
        )
        yield* fs.rename(staging, destination).pipe(
          Effect.catchAll((cause) =>
            fs.stat(destination).pipe(
              Effect.flatMap((info) =>
                info.type === "Directory" ? Effect.void : Effect.fail(cause)
              ),
              Effect.mapError(() => cause)
            )
          ),
          Effect.mapError(() =>
            failure("install", "unable to publish release installation")
          )
        )
      }).pipe(
        Effect.ensuring(
          fs
            .remove(staging, { recursive: true, force: true })
            .pipe(Effect.catchAll(() => Effect.void))
        ),
        Effect.catchTag("ReleaseDiskFullError", () =>
          Effect.fail(new ReleaseDiskFullError({
            directory: parent,
            requiredBytes: Option.map(expanded, (bytes) => artifact.bytes + bytes),
            availableBytes: Option.none(),
          }))
        )
      )
      return destination
    })
  )
