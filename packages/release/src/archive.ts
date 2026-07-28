import { createReadStream, createWriteStream } from "node:fs"
import { chmod, mkdir } from "node:fs/promises"
import { dirname, resolve, sep } from "node:path"
import { pipeline } from "node:stream/promises"
import { createGunzip } from "node:zlib"
import { extract } from "tar-stream"
import { Context, Effect, Layer, Option } from "effect"
import type { ReleaseArtifact } from "./contracts"
import { ReleaseAcquisitionError } from "./errors"

const EXPANDED_LIMIT = 8 * 1024 * 1024 * 1024
const ENTRY_LIMIT = 65_536

export interface ArchiveExtractorService {
  readonly extract: (
    archive: string,
    destination: string,
    artifact: ReleaseArtifact,
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

const safePath = (value: string): string => {
  if (
    value.length === 0 ||
    value.length > 1024 ||
    value.startsWith("/") ||
    value.startsWith("\\") ||
    value.includes("\\") ||
    value.includes("\0") ||
    /^[a-zA-Z]:/.test(value) ||
    value.split("/").some((part) => part === "" || part === "." || part === "..")
  ) throw archiveError(`unsafe archive path ${value}`)
  return value
}

const validateLayout = (artifact: ReleaseArtifact, paths: ReadonlySet<string>): void => {
  const host = Option.getOrUndefined(artifact.host)
  const extension = host === "windows-x64-msvc" ? ".exe" : ""
  if (artifact.kind === "cli" || artifact.kind === "acn") {
    const expected = `bin/magnitude-${artifact.kind}${extension}`
    if (paths.size !== 1 || !paths.has(expected)) {
      throw archiveError(`${artifact.id} has an invalid ${artifact.kind} layout`)
    }
    return
  }
  if (artifact.kind === "icn-base") {
    for (const required of [
      `bin/magnitude-icn${extension}`,
      "catalog/release-catalog.lock.json",
      "catalog/model-planner-inputs.bundle",
    ]) {
      if (!paths.has(required)) throw archiveError(`${artifact.id} is missing ${required}`)
    }
    if (![...paths].some((path) => path.startsWith("backends/"))) {
      throw archiveError(`${artifact.id} has no CPU backend`)
    }
    if ([...paths].some((path) =>
      !path.startsWith("bin/") &&
      !path.startsWith("runtime/") &&
      !path.startsWith("backends/") &&
      !path.startsWith("catalog/")
    )) throw archiveError(`${artifact.id} has an unexpected path`)
    return
  }
  if (artifact.kind === "icn-backend") {
    if (
      ![...paths].some((path) => path.startsWith("backends/")) ||
      [...paths].some((path) => !path.startsWith("runtime/") && !path.startsWith("backends/"))
    ) throw archiveError(`${artifact.id} has an invalid backend-pack layout`)
  }
}

// tar-stream exposes Node callback streams rather than Effect streams. This is the single
// platform adapter; acquisition, staging, cleanup, and publication remain in Effect.
const extractWithNodeStreams = (
  archive: string,
  destination: string,
  artifact: ReleaseArtifact,
): Effect.Effect<void, ReleaseAcquisitionError> =>
  Effect.async<void, ReleaseAcquisitionError>((resume) => {
    const reader = extract()
    const source = createReadStream(archive)
    const gunzip = createGunzip()
    const root = resolve(destination)
    const paths = new Set<string>()
    let entries = 0
    let expanded = 0
    reader.on("entry", (header, stream, next) => {
      const fail = (cause: unknown): void => {
        stream.resume()
        reader.destroy(cause instanceof Error ? cause : new Error(String(cause)))
      }
      try {
        if (header.type !== "file") throw archiveError("release archives may contain only files")
        const relative = safePath(header.name)
        if (paths.has(relative) || ++entries > ENTRY_LIMIT) {
          throw archiveError(`duplicate or excessive archive entry ${relative}`)
        }
        paths.add(relative)
        const output = resolve(root, relative)
        if (!output.startsWith(`${root}${sep}`)) throw archiveError(`${relative} escapes staging`)
        void (async () => {
          await mkdir(dirname(output), { recursive: true, mode: 0o700 })
          stream.on("data", (chunk: Buffer) => {
            expanded += chunk.byteLength
            if (expanded > EXPANDED_LIMIT) stream.destroy(archiveError("expanded archive is too large"))
          })
          await pipeline(stream, createWriteStream(output, {
            flags: "wx",
            mode: (header.mode ?? 0o644) & 0o777,
          }))
          await chmod(output, (header.mode ?? 0o644) & 0o777)
          next()
        })().catch(fail)
      } catch (cause) {
        fail(cause)
      }
    })
    pipeline(source, gunzip, reader)
      .then(() => {
        try {
          validateLayout(artifact, paths)
          resume(Effect.void)
        } catch (cause) {
          resume(Effect.fail(
            cause instanceof ReleaseAcquisitionError
              ? cause
              : archiveError("release archive layout validation failed"),
          ))
        }
      })
      .catch((cause) =>
        resume(Effect.fail(
          cause instanceof ReleaseAcquisitionError
            ? cause
            : archiveError(cause instanceof Error ? cause.message : "archive extraction failed"),
        ))
      )
    return Effect.sync(() => {
      source.destroy()
      gunzip.destroy()
      reader.destroy()
    })
  })

export const NodeArchiveExtractor = Layer.succeed(
  ArchiveExtractor,
  ArchiveExtractor.of({ extract: extractWithNodeStreams }),
)
