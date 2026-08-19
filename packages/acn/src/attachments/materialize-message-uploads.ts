import { isUtf8 } from "node:buffer"
import { Effect, Option } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { createId } from "@magnitudedev/generate-id"
import { captureContextImageFromFile, type ContextImagePart } from "@magnitudedev/agent"
import {
  MAX_TEXT_FILE_UPLOAD_BYTES,
  MAX_IMAGE_FILE_UPLOAD_BYTES,
  MAX_MESSAGE_UPLOAD_BYTES,
  MAX_MESSAGE_UPLOAD_COUNT,
  SessionOperationFailed,
  canonicalExtensionForImageMediaType,
  filenameWithImageExtension,
  type RawImageAttachment,
  type RawMessageUpload,
  type RawMentionOccurrence,
  type SessionError,
} from "@magnitudedev/acn-protocol"
import { uploadAttachment } from "../attachment-upload"

export interface MaterializedMessageUploads {
  readonly images: readonly ContextImagePart[]
  readonly trailingMentions: readonly RawMentionOccurrence[]
  /** Internal rollback targets retained until the user-message event is admitted. */
  readonly capturedAbsolutePaths: readonly string[]
}

function canonicalBase64Bytes(data: string): Uint8Array | null {
  if (data.length === 0 || data.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(data)) {
    return null
  }
  const bytes = Buffer.from(data, "base64")
  return bytes.toString("base64") === data ? bytes : null
}

function validateTextUpload(upload: Extract<RawMessageUpload, { type: "raw_text_file" }>): Effect.Effect<number, SessionError> {
  return Effect.gen(function* () {
    const bytes = canonicalBase64Bytes(upload.data)
    if (!bytes) {
      return yield* new SessionOperationFailed({
        operation: "MaterializeMessageUpload",
        reason: upload.data.length === 0
          ? `${upload.filename} is empty`
          : `${upload.filename} is not valid base64 data`,
      })
    }
    if (bytes.byteLength > MAX_TEXT_FILE_UPLOAD_BYTES) {
      return yield* new SessionOperationFailed({
        operation: "MaterializeMessageUpload",
        reason: `${upload.filename} exceeds the 500 KiB text attachment limit`,
      })
    }
    if (bytes.includes(0) || !isUtf8(bytes)) {
      return yield* new SessionOperationFailed({
        operation: "MaterializeMessageUpload",
        reason: `${upload.filename} is not a UTF-8 text file`,
      })
    }
    return bytes.byteLength
  })
}

function validateImageUpload(upload: RawImageAttachment): Effect.Effect<number, SessionError> {
  return Effect.gen(function* () {
    const bytes = canonicalBase64Bytes(upload.data)
    if (!bytes) {
      return yield* new SessionOperationFailed({
        operation: "MaterializeMessageUpload",
        reason: "Image attachment is not valid base64 data",
      })
    }
    if (bytes.byteLength > MAX_IMAGE_FILE_UPLOAD_BYTES) {
      return yield* new SessionOperationFailed({
        operation: "MaterializeMessageUpload",
        reason: "Image attachment exceeds the 10 MiB limit",
      })
    }
    return bytes.byteLength
  })
}

function imageFilename(upload: RawImageAttachment): string {
  return upload.type === "raw_image_file"
    ? filenameWithImageExtension(upload.filename, upload.mediaType)
    : `clipboard-${createId().slice(0, 12)}.${canonicalExtensionForImageMediaType(upload.mediaType)}`
}

export function materializeMessageUploads(input: {
  readonly scratchpadPath: string
  readonly uploads: readonly RawMessageUpload[]
}): Effect.Effect<MaterializedMessageUploads, SessionError, FileSystem.FileSystem | Path.Path> {
  return Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path

    if (input.uploads.length > MAX_MESSAGE_UPLOAD_COUNT) {
      return yield* new SessionOperationFailed({
        operation: "MaterializeMessageUpload",
        reason: `A message can include at most ${MAX_MESSAGE_UPLOAD_COUNT} attachments`,
      })
    }

    let aggregateBytes = 0
    for (const upload of input.uploads) {
      aggregateBytes += upload.type === "raw_text_file"
        ? yield* validateTextUpload(upload)
        : yield* validateImageUpload(upload)
      if (aggregateBytes > MAX_MESSAGE_UPLOAD_BYTES) {
        return yield* new SessionOperationFailed({
          operation: "MaterializeMessageUpload",
          reason: "Attachments exceed the 25 MiB total message limit",
        })
      }
    }

    const capturedPaths: string[] = []
    const images: ContextImagePart[] = []
    const trailingMentions: RawMentionOccurrence[] = []

    const materialize = Effect.gen(function* () {
      for (const upload of input.uploads) {
        const filename = upload.type === "raw_text_file" ? upload.filename : imageFilename(upload)
        const uploaded = yield* uploadAttachment(
          input.scratchpadPath,
          Option.some(filename),
          upload.data,
        )
        const absolutePath = path.join(input.scratchpadPath, uploaded.path.replace(/^\$M\//, ""))
        capturedPaths.push(absolutePath)

        if (upload.type === "raw_text_file") {
          trailingMentions.push({
            occurrenceId: createId(),
            attachment: { type: "mention_file", path: uploaded.path },
            placement: { _tag: "trailing" },
          })
          continue
        }

        const image = yield* captureContextImageFromFile({
          absolutePath,
          logicalPath: uploaded.path,
          name: uploaded.filename,
        }).pipe(Effect.mapError(error => new SessionOperationFailed({
          operation: "MaterializeMessageUpload",
          reason: error.message,
        })))
        images.push(image)
      }

      return {
        images,
        trailingMentions,
        capturedAbsolutePaths: [...capturedPaths],
      } satisfies MaterializedMessageUploads
    })

    return yield* materialize.pipe(
      Effect.onError(() => Effect.forEach(
        capturedPaths,
        capturedPath => fs.remove(capturedPath).pipe(Effect.catchAll(() => Effect.void)),
        { discard: true },
      )),
    )
  })
}
