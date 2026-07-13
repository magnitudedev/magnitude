import { Effect, Option } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { createId } from "@magnitudedev/generate-id"
import {
  canonicalExtensionForImageMediaType,
  filenameWithImageExtension,
  type MessageAttachment,
  type RawMessageAttachment,
} from "@magnitudedev/protocol"
import type { SessionError } from "@magnitudedev/protocol"
import { uploadAttachment } from "../attachment-upload"
import { mergeInlineMentions } from "../file-mentions"

export interface MaterializeRawMessageAttachmentsInput {
  readonly cwd: string
  readonly scratchpadPath: string
  readonly messageContent: string
  readonly attachments: readonly RawMessageAttachment[]
}

function clipboardFilename(attachment: Extract<RawMessageAttachment, { type: "raw_image_clipboard" }>): string {
  return `clipboard-${createId().slice(0, 12)}.${canonicalExtensionForImageMediaType(attachment.mediaType)}`
}

function uploadImageAttachment(
  scratchpadPath: string,
  attachment: Extract<RawMessageAttachment, { type: "raw_image_clipboard" | "raw_image_file" }>,
) {
  const filename = attachment.type === "raw_image_file"
    ? filenameWithImageExtension(attachment.filename, attachment.mediaType)
    : clipboardFilename(attachment)

  return uploadAttachment(
    scratchpadPath,
    Option.some(filename),
    attachment.data,
  )
}

export function materializeRawMessageAttachments(
  input: MaterializeRawMessageAttachmentsInput,
): Effect.Effect<readonly MessageAttachment[], SessionError, FileSystem.FileSystem | Path.Path> {
  return Effect.gen(function* () {
    const materialized: MessageAttachment[] = []

    for (const attachment of input.attachments) {
      switch (attachment.type) {
        case "raw_image_clipboard":
        case "raw_image_file": {
          const uploaded = yield* uploadImageAttachment(input.scratchpadPath, attachment)
          materialized.push({
            type: "image",
            path: uploaded.path,
            filename: uploaded.filename,
            mediaType: attachment.mediaType,
            width: attachment.width,
            height: attachment.height,
          })
          break
        }
        case "mention_file":
        case "mention_file_range":
        case "mention_directory":
          materialized.push(attachment)
          break
      }
    }

    return yield* mergeInlineMentions(
      input.cwd,
      input.scratchpadPath,
      input.messageContent,
      materialized,
    )
  })
}
