import type { MentionAttachment } from "@magnitudedev/sdk"

export {
  canonicalExtensionForImageMediaType,
  filenameWithImageExtension,
  imageMediaTypeFromFilename,
  imageMediaTypeFromMime,
  isSupportedImageFilename,
} from "@magnitudedev/sdk"

export interface MentionSegment {
  path: string
  contentType: "text" | "directory"
  lineRange?: { start: number; end: number }
}

export function mentionAttachmentFromSegment(mention: MentionSegment): MentionAttachment {
  if (mention.contentType === "directory") {
    return { type: "mention_directory", path: mention.path }
  }
  if (mention.lineRange) {
    return {
      type: "mention_file_range",
      path: mention.path,
      startLine: mention.lineRange.start,
      endLine: mention.lineRange.end,
    }
  }
  return { type: "mention_file", path: mention.path }
}
