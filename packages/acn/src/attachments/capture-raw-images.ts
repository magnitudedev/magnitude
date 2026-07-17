import { Effect, Option } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { createId } from "@magnitudedev/generate-id"
import { captureContextImageFromFile, type ContextImagePart } from "@magnitudedev/agent"
import {
  SessionOperationFailed,
  canonicalExtensionForImageMediaType,
  filenameWithImageExtension,
  type RawImageAttachment,
  type SessionError,
} from "@magnitudedev/protocol"
import { uploadAttachment } from "../attachment-upload"

export interface CaptureRawImagesInput {
  readonly scratchpadPath: string
  readonly attachments: readonly RawImageAttachment[]
}

function attachmentFilename(attachment: RawImageAttachment): string {
  if (attachment.type === "raw_image_file") {
    return filenameWithImageExtension(attachment.filename, attachment.mediaType)
  }
  return `clipboard-${createId().slice(0, 12)}.${canonicalExtensionForImageMediaType(attachment.mediaType)}`
}

export function captureRawImages(
  input: CaptureRawImagesInput,
): Effect.Effect<readonly ContextImagePart[], SessionError, FileSystem.FileSystem | Path.Path> {
  return Effect.gen(function* () {
    const path = yield* Path.Path
    return yield* Effect.forEach(input.attachments, attachment => Effect.gen(function* () {
      const uploaded = yield* uploadAttachment(
        input.scratchpadPath,
        Option.some(attachmentFilename(attachment)),
        attachment.data,
      )
      return yield* captureContextImageFromFile({
        absolutePath: path.join(input.scratchpadPath, uploaded.path.replace(/^\$M\//, "")),
        logicalPath: uploaded.path,
        name: uploaded.filename,
      }).pipe(Effect.mapError(error => new SessionOperationFailed({
        operation: "CaptureImageAttachment",
        reason: error.message,
      })))
    }), { concurrency: "unbounded" })
  })
}
