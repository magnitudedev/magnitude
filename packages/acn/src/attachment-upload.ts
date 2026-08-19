import { type SessionError, SessionOperationFailed } from "@magnitudedev/acn-protocol"
import { Effect, Option } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import type { PlatformError } from "@effect/platform/Error"
import { createId } from "@magnitudedev/generate-id"

// ---------------------------------------------------------------------------
// Attachment upload — writes base64-decoded content to $M/attachments/
// ---------------------------------------------------------------------------

const ATTACHMENTS_SUBDIR = "attachments"

export function attachmentLogicalPath(filename: string): string {
  return `$M/${ATTACHMENTS_SUBDIR}/${filename}`
}

/** Sanitize a filename: strip path components, replace dangerous characters. */
function sanitizeFilename(name: string): string {
  const base = name.replace(/[^a-zA-Z0-9._-]/g, "_")
  return base.length > 0 ? base : createId().slice(0, 8)
}

function filenameCandidate(filename: string, index: number): string {
  if (index === 0) return filename
  return filenameWithSuffix(filename, String(index))
}

function filenameWithSuffix(filename: string, suffix: string): string {
  const dotIndex = filename.lastIndexOf(".")
  const stem = dotIndex > 0 ? filename.slice(0, dotIndex) : filename
  const ext = dotIndex > 0 ? filename.slice(dotIndex) : ""
  return `${stem}-${suffix}${ext}`
}

function isAlreadyExists(error: PlatformError): boolean {
  return error._tag === "SystemError" && error.reason === "AlreadyExists"
}

/**
 * Upload a raw attachment (base64 data) to the session's attachments
 * directory. Returns the logical agent path and resolved filename.
 */
export const uploadAttachment = (
  scratchpadPath: string,
  filename: Option.Option<string>,
  data: string,
): Effect.Effect<{ path: string; filename: string }, SessionError, FileSystem.FileSystem | Path.Path> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path

    const attachmentsDir = path.join(scratchpadPath, ATTACHMENTS_SUBDIR)
    yield* fs.makeDirectory(attachmentsDir, { recursive: true })

    const bytes = Buffer.from(data, "base64")
    const preferredFilename = Option.match(filename, {
      onNone: () => createId().slice(0, 12),
      onSome: sanitizeFilename,
    })

    for (let index = 0; index < 1000; index++) {
      const candidate = filenameCandidate(preferredFilename, index)
      const fullPath = path.join(attachmentsDir, candidate)
      const result = yield* Effect.either(Effect.scoped(
        fs.open(fullPath, { flag: "wx" }).pipe(
          Effect.flatMap(file => file.writeAll(bytes).pipe(Effect.zipRight(file.sync))),
        ),
      ))
      if (result._tag === "Right") {
        return { path: attachmentLogicalPath(candidate), filename: candidate }
      }
      if (!isAlreadyExists(result.left)) {
        yield* fs.remove(fullPath, { force: true }).pipe(Effect.ignore)
        return yield* result.left
      }
    }

    const fallback = filenameWithSuffix(preferredFilename, createId().slice(0, 12))
    const fullPath = path.join(attachmentsDir, fallback)
    yield* Effect.scoped(
      fs.open(fullPath, { flag: "wx" }).pipe(
        Effect.flatMap(file => file.writeAll(bytes).pipe(Effect.zipRight(file.sync))),
      ),
    ).pipe(Effect.onError(() => fs.remove(fullPath, { force: true }).pipe(Effect.ignore)))
    return { path: attachmentLogicalPath(fallback), filename: fallback }
  }).pipe(
    Effect.mapError(() =>
      new SessionOperationFailed({ operation: "UploadAttachment", reason: "Failed to write attachment file" })
    ),
  )
