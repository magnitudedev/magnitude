import { Schema } from "effect"

export const ImageMediaType = Schema.Literal(
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
)
export type ImageMediaType = Schema.Schema.Type<typeof ImageMediaType>

export const MAX_IMAGE_FILE_UPLOAD_BYTES = 10 * 1024 * 1024
const MAX_IMAGE_FILE_UPLOAD_BASE64_LENGTH = Math.ceil(MAX_IMAGE_FILE_UPLOAD_BYTES / 3) * 4

export const RawClipboardImageAttachment = Schema.Struct({
  type: Schema.Literal("raw_image_clipboard"),
  data: Schema.String.pipe(Schema.maxLength(MAX_IMAGE_FILE_UPLOAD_BASE64_LENGTH)),
  mediaType: ImageMediaType,
  width: Schema.Number,
  height: Schema.Number,
})
export type RawClipboardImageAttachment = Schema.Schema.Type<typeof RawClipboardImageAttachment>

export const RawFileImageAttachment = Schema.Struct({
  type: Schema.Literal("raw_image_file"),
  data: Schema.String.pipe(Schema.maxLength(MAX_IMAGE_FILE_UPLOAD_BASE64_LENGTH)),
  filename: Schema.String,
  mediaType: ImageMediaType,
  width: Schema.Number,
  height: Schema.Number,
})
export type RawFileImageAttachment = Schema.Schema.Type<typeof RawFileImageAttachment>

export const RawImageAttachment = Schema.Union(
  RawClipboardImageAttachment,
  RawFileImageAttachment,
)
export type RawImageAttachment = Schema.Schema.Type<typeof RawImageAttachment>

export const MAX_TEXT_FILE_UPLOAD_BYTES = 500 * 1024
export const MAX_MESSAGE_UPLOAD_COUNT = 20
export const MAX_MESSAGE_UPLOAD_BYTES = 25 * 1024 * 1024

export const RawTextFileUpload = Schema.Struct({
  type: Schema.Literal("raw_text_file"),
  data: Schema.String.pipe(
    Schema.maxLength(Math.ceil(MAX_TEXT_FILE_UPLOAD_BYTES / 3) * 4),
  ),
  filename: Schema.String,
})
export type RawTextFileUpload = Schema.Schema.Type<typeof RawTextFileUpload>

export const RawMessageUpload = Schema.Union(
  RawImageAttachment,
  RawTextFileUpload,
)
export type RawMessageUpload = Schema.Schema.Type<typeof RawMessageUpload>

export const RawMessageUploads = Schema.Array(RawMessageUpload).pipe(
  Schema.maxItems(MAX_MESSAGE_UPLOAD_COUNT),
)
export type RawMessageUploads = Schema.Schema.Type<typeof RawMessageUploads>

export const MentionFileAttachment = Schema.Struct({
  type: Schema.Literal("mention_file"),
  path: Schema.String,
})
export type MentionFileAttachment = Schema.Schema.Type<typeof MentionFileAttachment>

export const MentionFileRangeAttachment = Schema.Struct({
  type: Schema.Literal("mention_file_range"),
  path: Schema.String,
  startLine: Schema.Number,
  endLine: Schema.Number,
})
export type MentionFileRangeAttachment = Schema.Schema.Type<typeof MentionFileRangeAttachment>

export const MentionDirectoryAttachment = Schema.Struct({
  type: Schema.Literal("mention_directory"),
  path: Schema.String,
})
export type MentionDirectoryAttachment = Schema.Schema.Type<typeof MentionDirectoryAttachment>

export const MentionAttachment = Schema.Union(
  MentionFileAttachment,
  MentionFileRangeAttachment,
  MentionDirectoryAttachment,
)
export type MentionAttachment = Schema.Schema.Type<typeof MentionAttachment>

export const MentionPlacement = Schema.Union(
  Schema.TaggedStruct("inline", {
    start: Schema.Number,
    end: Schema.Number,
  }),
  Schema.TaggedStruct("trailing", {}),
)
export type MentionPlacement = Schema.Schema.Type<typeof MentionPlacement>

export const RawMentionOccurrence = Schema.Struct({
  occurrenceId: Schema.String,
  attachment: MentionAttachment,
  placement: MentionPlacement,
})
export type RawMentionOccurrence = Schema.Schema.Type<typeof RawMentionOccurrence>

export const ImageAttachment = Schema.Struct({
  type: Schema.Literal("image"),
  path: Schema.String,
  filename: Schema.String,
  mediaType: ImageMediaType,
  width: Schema.Number,
  height: Schema.Number,
})
export type ImageAttachment = Schema.Schema.Type<typeof ImageAttachment>

export const MessageAttachment = Schema.Union(
  ImageAttachment,
  MentionAttachment,
)
export type MessageAttachment = Schema.Schema.Type<typeof MessageAttachment>

export const DisplayAttachment = MessageAttachment
export type DisplayAttachment = MessageAttachment

const IMAGE_TYPES_BY_EXTENSION: Readonly<Record<string, ImageMediaType>> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".gif": "image/gif",
}

const CANONICAL_EXTENSIONS: Readonly<Record<ImageMediaType, string>> = {
  "image/png": "png",
  "image/jpeg": "jpg",
  "image/webp": "webp",
  "image/gif": "gif",
}

function extensionOf(filename: string): string {
  const dot = filename.lastIndexOf(".")
  return dot >= 0 ? filename.slice(dot).toLowerCase() : ""
}

export function imageMediaTypeFromFilename(filename: string): ImageMediaType | null {
  return IMAGE_TYPES_BY_EXTENSION[extensionOf(filename)] ?? null
}

export function imageMediaTypeFromMime(mime: string): ImageMediaType | null {
  return (mime === "image/png" || mime === "image/jpeg" || mime === "image/webp" || mime === "image/gif")
    ? (mime as ImageMediaType)
    : null
}

export function canonicalExtensionForImageMediaType(mediaType: ImageMediaType): string {
  return CANONICAL_EXTENSIONS[mediaType]
}

export function isSupportedImageFilename(filename: string): boolean {
  return imageMediaTypeFromFilename(filename) !== null
}

export function filenameWithImageExtension(filename: string, mediaType: ImageMediaType): string {
  const ext = `.${canonicalExtensionForImageMediaType(mediaType)}`
  const dot = filename.lastIndexOf(".")
  if (dot <= 0) return `${filename}${ext}`
  return `${filename.slice(0, dot)}${ext}`
}
