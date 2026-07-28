import { createHash, createPublicKey, verify } from "node:crypto"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as Path from "@effect/platform/Path"
import { Effect, Option, Schedule, Schema, Stream } from "effect"
import { ArchiveExtractor } from "./archive"
import { ReleaseAcquisitionError } from "./errors"
import {
  decodeReleaseManifest,
  ReleaseSignatureSchema,
  type ReleaseArtifact,
  type ReleaseManifest,
  type TrustedReleaseKey,
} from "./contracts"

const MANIFEST_LIMIT = 16 * 1024 * 1024
const SIGNATURE_LIMIT = 16 * 1024
const ARCHIVE_LIMIT = 4 * 1024 * 1024 * 1024

export interface AuthenticatedRelease {
  readonly manifest: ReleaseManifest
  readonly manifestBytes: Uint8Array
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

export const authenticateRelease = (
  manifestBytes: Uint8Array,
  signatureBytes: Uint8Array,
  trustedKeys: readonly TrustedReleaseKey[],
): Effect.Effect<AuthenticatedRelease, ReleaseAcquisitionError> =>
  Effect.gen(function* () {
    const signature = yield* Schema.decodeUnknown(
      Schema.parseJson(ReleaseSignatureSchema),
    )(new TextDecoder().decode(signatureBytes)).pipe(
      Effect.mapError(() => failure("authenticate", "release signature is malformed")),
    )
    const trusted = trustedKeys.find((key) => key.keyId === signature.keyId)
    if (!trusted) return yield* failure("authenticate", "release signature uses an untrusted key")
    const valid = yield* Effect.sync(() => {
      try {
        return verify(
          null,
          manifestBytes,
          createPublicKey({
            key: Buffer.from(trusted.publicKeySpki, "base64"),
            format: "der",
            type: "spki",
          }),
          Buffer.from(signature.signature, "base64"),
        )
      } catch {
        return false
      }
    })
    if (!valid) return yield* failure("authenticate", "release signature is invalid")
    const manifest = yield* decodeReleaseManifest(manifestBytes).pipe(
      Effect.mapError((error) => failure("authenticate", error.message)),
    )
    return {
      manifest,
      manifestBytes,
      manifestSha256: createHash("sha256").update(manifestBytes).digest("hex"),
    }
  })

const readCachedPair = (
  directory: string,
): Effect.Effect<
  Option.Option<readonly [Uint8Array, Uint8Array]>,
  never,
  FileSystem.FileSystem | Path.Path
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const pair = yield* Effect.all([
      fs.readFile(path.join(directory, "magnitude-release.json")),
      fs.readFile(path.join(directory, "magnitude-release.json.sig")),
    ]).pipe(Effect.option)
    if (Option.isNone(pair)) return Option.none()
    const [manifest, signature] = pair.value
    return manifest.byteLength <= MANIFEST_LIMIT && signature.byteLength <= SIGNATURE_LIMIT
      ? Option.some([manifest, signature] as const)
      : Option.none()
  })

const publishCachedPair = (
  directory: string,
  manifest: Uint8Array,
  signature: Uint8Array,
): Effect.Effect<void, never, FileSystem.FileSystem | Path.Path> =>
  Effect.scoped(
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const parent = path.dirname(directory)
      yield* fs.makeDirectory(parent, { recursive: true, mode: 0o700 })
      const staging = yield* fs.makeTempDirectoryScoped({
        directory: parent,
        prefix: ".manifest-",
      })
      yield* Effect.all([
        fs.writeFile(path.join(staging, "magnitude-release.json"), manifest, {
          flag: "wx",
          mode: 0o600,
        }),
        fs.writeFile(path.join(staging, "magnitude-release.json.sig"), signature, {
          flag: "wx",
          mode: 0o600,
        }),
      ])
      yield* fs.rename(staging, directory).pipe(
        Effect.catchAll(() => Effect.void),
      )
    }),
  ).pipe(Effect.catchAll(() => Effect.void))

export const acquireRelease = (
  baseUrl: string,
  version: string,
  trustedKeys: readonly TrustedReleaseKey[],
  cacheDirectory: string,
): Effect.Effect<
  AuthenticatedRelease,
  ReleaseAcquisitionError,
  FileSystem.FileSystem | Path.Path | HttpClient.HttpClient
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const cached = yield* readCachedPair(cacheDirectory)
    if (Option.isSome(cached)) {
      const authenticated = yield* authenticateRelease(
        cached.value[0],
        cached.value[1],
        trustedKeys,
      ).pipe(Effect.option)
      if (Option.isSome(authenticated) && authenticated.value.manifest.version === version) {
        return authenticated.value
      }
    }
    yield* fs.remove(cacheDirectory, { recursive: true, force: true }).pipe(
      Effect.catchAll(() => Effect.void),
    )
    const manifestUrl = releaseUrl(baseUrl, version, "magnitude-release.json")
    const [manifestBytes, signatureBytes] = yield* Effect.all(
      [
        getBounded(manifestUrl, MANIFEST_LIMIT),
        getBounded(`${manifestUrl}.sig`, SIGNATURE_LIMIT),
      ],
      { concurrency: 2 },
    )
    const authenticated = yield* authenticateRelease(
      manifestBytes,
      signatureBytes,
      trustedKeys,
    )
    if (
      authenticated.manifest.version !== version ||
      authenticated.manifest.tag !== releaseTag(version)
    ) return yield* failure("authenticate", "release identity differs from requested version")
    yield* publishCachedPair(cacheDirectory, manifestBytes, signatureBytes)
    return authenticated
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
      "authenticate",
      `release has ${matches.length} matching ${kind} artifacts`,
    ))
}

const downloadArtifact = (
  url: string,
  artifact: ReleaseArtifact,
  destination: string,
): Effect.Effect<
  void,
  ReleaseAcquisitionError,
  FileSystem.FileSystem | HttpClient.HttpClient
> =>
  Effect.suspend(() => {
    const digest = createHash("sha256")
    let bytes = 0
    return Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const client = yield* HttpClient.HttpClient
      yield* fs.truncate(destination, 0).pipe(
        Effect.mapError(() => failure("download", "unable to reset release staging file")),
      )
      const response = yield* client.execute(HttpClientRequest.get(url)).pipe(
        Effect.mapError(() => failure("download", "release artifact request failed", true)),
      )
      if (response.status < 200 || response.status >= 300) {
        return yield* failure(
          "download",
          `release endpoint returned HTTP ${response.status}`,
          transientStatus(response.status),
        )
      }
      const declared = Number(response.headers["content-length"])
      if (Number.isFinite(declared) && declared !== artifact.bytes) {
        return yield* failure("download", "release artifact size differs from its manifest")
      }
      yield* response.stream.pipe(
        Stream.tap((chunk) =>
          bytes + chunk.byteLength > artifact.bytes ||
          bytes + chunk.byteLength > ARCHIVE_LIMIT
            ? Effect.fail(failure(
              "download",
              "release artifact exceeds its authenticated size",
            ))
            : Effect.sync(() => {
              bytes += chunk.byteLength
              digest.update(chunk)
            })
        ),
        Stream.run(fs.sink(destination, { flag: "w", mode: 0o600 })),
        Effect.mapError((cause) =>
          cause instanceof ReleaseAcquisitionError
            ? cause
            : failure("download", "release artifact stream failed", true)
        ),
      )
      if (bytes !== artifact.bytes || digest.digest("hex") !== artifact.sha256) {
        return yield* failure("download", "release artifact digest or size differs from manifest")
      }
    })
  }).pipe(
    Effect.timeoutFail({
      duration: "10 minutes",
      onTimeout: () => failure("download", "release artifact response timed out", true),
    }),
    Effect.retry({
      while: (error) => error.transient,
      schedule: Schedule.exponential("500 millis").pipe(
        Schedule.jittered,
        Schedule.intersect(Schedule.recurs(2)),
      ),
    }),
  )

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
      const archive = yield* fs.makeTempFileScoped({
        directory: parent,
        prefix: ".download-",
      }).pipe(
        Effect.mapError(() => failure("install", "unable to create release staging file")),
      )
      yield* downloadArtifact(
        releaseUrl(baseUrl, version, artifact.filename),
        artifact,
        archive,
      )
      const staging = yield* fs.makeTempDirectoryScoped({
        directory: parent,
        prefix: ".release-",
      }).pipe(
        Effect.mapError(() => failure("install", "unable to create extraction staging directory")),
      )
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
      return destination
    }),
  )
