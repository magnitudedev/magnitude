import { type SessionError, SessionOperationFailed } from "@magnitudedev/protocol"
import { Effect, Option } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
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

/** Resolve a non-colliding filename within the attachments directory. */
function resolveUniqueFilename(
  attachmentsDir: string,
  filename: string,
  exists: (p: string) => Effect.Effect<boolean>,
): Effect.Effect<string, never, FileSystem.FileSystem | Path.Path> {
  return Effect.gen(function* () {
    const path = yield* Path.Path
    const fullPath = path.join(attachmentsDir, filename)
    const fileExists = yield* exists(fullPath)
    if (!fileExists) return filename

    const dotIndex = filename.lastIndexOf(".")
    const stem = dotIndex > 0 ? filename.slice(0, dotIndex) : filename
    const ext = dotIndex > 0 ? filename.slice(dotIndex) : ""

    for (let i = 1; i < 1000; i++) {
      const candidate = `${stem}-${i}${ext}`
      const candidatePath = path.join(attachmentsDir, candidate)
      const candidateExists = yield* exists(candidatePath)
      if (!candidateExists) return candidate
    }
    // Extremely unlikely fallback
    return `${stem}-${createId().slice(0, 8)}${ext}`
  })
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

    const exists = (p: string) =>
      fs.exists(p).pipe(Effect.catchAll(() => Effect.succeed(false)))

    const resolvedFilename = yield* Option.match(filename, {
      onNone: () => Effect.succeed(createId().slice(0, 12)),
      onSome: (name) => resolveUniqueFilename(attachmentsDir, sanitizeFilename(name), exists),
    })

    const fullPath = path.join(attachmentsDir, resolvedFilename)
    const bytes = Buffer.from(data, "base64")
    yield* fs.writeFile(fullPath, bytes)

    return { path: attachmentLogicalPath(resolvedFilename), filename: resolvedFilename }
  }).pipe(
    Effect.mapError(() =>
      new SessionOperationFailed({ operation: "UploadAttachment", reason: "Failed to write attachment file" })
    ),
  )
