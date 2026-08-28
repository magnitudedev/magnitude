import { createReadStream, createWriteStream } from "node:fs"
import { chmod, mkdir } from "node:fs/promises"
import { dirname, resolve, sep } from "node:path"
import { Transform } from "node:stream"
import { pipeline } from "node:stream/promises"
import { createGunzip } from "node:zlib"
import { extract } from "tar-stream"
import { Context, Effect, Either, Layer, Option, Runtime } from "effect"
import type { ReleaseArtifact } from "./contracts"
import { ReleaseAcquisitionError } from "./errors"
import type { ArtifactByteProgress } from "./installation-progress"
import {
  ACN_EXECUTABLE_NAME,
  ICN_EXECUTABLE_NAME,
} from "./executables"

const EXPANDED_LIMIT = 8 * 1024 * 1024 * 1024
const ENTRY_LIMIT = 65_536

export interface ArchiveExtractorService {
  readonly extract: (
    archive: string,
    destination: string,
    artifact: ReleaseArtifact,
    onProgress: Option.Option<
      (progress: ArtifactByteProgress) => Effect.Effect<void>
    >,
  ) => Effect.Effect<void, ReleaseAcquisitionError>
}

export class ArchiveExtractor extends Context.Tag("@magnitudedev/release/ArchiveExtractor")<
  ArchiveExtractor,
  ArchiveExtractorService
>() {}

const archiveError = (message: string) =>
  new ReleaseAcquisitionError({
    stage: "archive",
    message,
    transient: false,
  })

const safePath = (
  value: string,
): Either.Either<string, ReleaseAcquisitionError> => {
  if (
    value.length === 0 ||
    value.length > 1024 ||
    value.startsWith("/") ||
    value.startsWith("\\") ||
    value.includes("\\") ||
    value.includes("\0") ||
    /^[a-zA-Z]:/.test(value) ||
    value.split("/").some((part) => part === "" || part === "." || part === "..")
  ) return Either.left(archiveError(`unsafe archive path ${value}`))
  return Either.right(value)
}

const validateLayout = (
  artifact: ReleaseArtifact,
  paths: ReadonlySet<string>,
): Effect.Effect<void, ReleaseAcquisitionError> => {
  const extension = Option.exists(
    artifact.host,
    (host) => host === "windows-x64-msvc",
  ) ? ".exe" : ""
  if (artifact.kind === "cli" || artifact.kind === "acn") {
    const expected = artifact.kind === "cli"
      ? `bin/magnitude-cli${extension}`
      : `bin/${ACN_EXECUTABLE_NAME}${extension}`
    if (paths.size !== 1 || !paths.has(expected)) {
      return archiveError(
        `${artifact.id} has an invalid ${artifact.kind} layout`,
      )
    }
    return Effect.void
  }
  if (artifact.kind === "icn-base") {
    for (const required of [
      `bin/${ICN_EXECUTABLE_NAME}${extension}`,
      "catalog/model-planner-inputs.bundle",
    ]) {
      if (!paths.has(required)) {
        return archiveError(`${artifact.id} is missing ${required}`)
      }
    }
    if (![...paths].some((path) => path.startsWith("backends/"))) {
      return archiveError(`${artifact.id} has no CPU backend`)
    }
    if ([...paths].some((path) =>
      !path.startsWith("bin/") &&
      !path.startsWith("runtime/") &&
      !path.startsWith("backends/") &&
      !path.startsWith("catalog/")
    )) {
      return archiveError(`${artifact.id} has an unexpected path`)
    }
    return Effect.void
  }
  if (artifact.kind === "icn-backend") {
    if (
      ![...paths].some((path) => path.startsWith("backends/")) ||
      [...paths].some((path) => !path.startsWith("runtime/") && !path.startsWith("backends/"))
    ) {
      return archiveError(`${artifact.id} has an invalid backend-pack layout`)
    }
  }
  return Effect.void
}

// tar-stream exposes Node callback streams rather than Effect streams. This is the single
// platform adapter; acquisition, staging, cleanup, and publication remain in Effect.
const extractWithNodeStreams = (
  archive: string,
  destination: string,
  artifact: ReleaseArtifact,
  onProgress: Option.Option<
    (progress: ArtifactByteProgress) => Effect.Effect<void>
  >,
): Effect.Effect<void, ReleaseAcquisitionError> =>
  Effect.gen(function* () {
    const runtime = yield* Effect.runtime<never>()
    return yield* Effect.async<void, ReleaseAcquisitionError>((resume) => {
      const reader = extract()
      const source = createReadStream(archive)
      let compressed = 0
      const reportProgress = new Transform({
        transform(chunk: Buffer, _encoding, callback) {
          compressed += chunk.byteLength
          Option.match(onProgress, {
            onNone: () => callback(null, chunk),
            onSome: (report) => {
              Runtime.runPromise(runtime)(
                report({
                  completedBytes: compressed,
                  totalBytes: artifact.bytes,
                })
              ).then(
                () => callback(null, chunk),
                () => callback(new Error("archive progress reporter failed"))
              )
            },
          })
        },
      })
      const gunzip = createGunzip()
      const root = resolve(destination)
      const paths = new Set<string>()
      let entries = 0
      let expanded = 0
      reader.on("entry", (header, stream, next) => {
        const fail = (cause: Error): void => {
          stream.resume()
          reader.destroy(cause)
        }
        if (header.type !== "file") {
          fail(archiveError("release archives may contain only files"))
          return
        }
        const relative = safePath(header.name)
        if (Either.isLeft(relative)) {
          fail(relative.left)
          return
        }
        if (paths.has(relative.right) || ++entries > ENTRY_LIMIT) {
          fail(
            archiveError(
              `duplicate or excessive archive entry ${relative.right}`,
            ),
          )
          return
        }
        paths.add(relative.right)
        const output = resolve(root, relative.right)
        if (!output.startsWith(`${root}${sep}`)) {
          fail(archiveError(`${relative.right} escapes staging`))
          return
        }
        void (async () => {
          await mkdir(dirname(output), { recursive: true, mode: 0o700 })
          stream.on("data", (chunk: Buffer) => {
            expanded += chunk.byteLength
            if (expanded > EXPANDED_LIMIT) {
              stream.destroy(archiveError("expanded archive is too large"))
            }
          })
          await pipeline(stream, createWriteStream(output, {
            flags: "wx",
            mode: (header.mode ?? 0o644) & 0o777,
          }))
          await chmod(output, (header.mode ?? 0o644) & 0o777)
          next()
        })().catch((cause) => fail(new Error(String(cause))))
      })
      pipeline(source, reportProgress, gunzip, reader)
        .then(() => resume(validateLayout(artifact, paths)))
        .catch((cause) =>
          resume(Effect.fail(
            archiveError(`archive extraction failed: ${String(cause)}`),
          ))
        )
      return Effect.sync(() => {
        source.destroy()
        reportProgress.destroy()
        gunzip.destroy()
        reader.destroy()
      })
    })
  })

export const NodeArchiveExtractor = Layer.succeed(
  ArchiveExtractor,
  ArchiveExtractor.of({ extract: extractWithNodeStreams }),
)
