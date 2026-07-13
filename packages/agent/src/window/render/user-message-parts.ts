import { ContentBuilder } from '../../content'
import type { UserPart } from '@magnitudedev/ai'
import type { TimelineEntry } from '../inbox/types'

export interface RenderTimelineUserMessagePartsOptions {
  readonly open: string
  readonly close: string
  readonly attachmentsInsideWrapper?: boolean
}

const defaultUserMessageOptions: RenderTimelineUserMessagePartsOptions = {
  open: '<message from="user">',
  close: '</message>',
  attachmentsInsideWrapper: false,
}

export function renderTimelineUserMessageParts(
  entry: Extract<TimelineEntry, { kind: 'user_message' }>,
  options: RenderTimelineUserMessagePartsOptions = defaultUserMessageOptions,
): UserPart[] {
  const builder = new ContentBuilder()
  const attachmentsInsideWrapper = options.attachmentsInsideWrapper === true
  builder.pushText(
    attachmentsInsideWrapper
      ? `${options.open}${entry.text}`
      : `${options.open}${entry.text}${options.close}`,
  )

  for (const attachment of entry.attachments) {
    switch (attachment.kind) {
      case 'image':
        builder.pushText(`\n<attachment path="${attachment.path}" filename="${attachment.filename}" media_type="${attachment.mediaType}" width="${attachment.width}" height="${attachment.height}" />`)
        break
      case 'mention': {
        const mention = attachment.attachment
        const mentionType = mention.type === 'mention_directory' ? 'directory' : 'file'
        const lineRange = mention.type === 'mention_file_range'
          ? ` lines="${mention.startLine}-${mention.endLine}"`
          : ''
        if (attachment.resolution.status === 'failed') {
          builder.pushText(`\n<mention path="${mention.path}" type="${mentionType}"${lineRange} status="failed" reason="${attachment.resolution.reason}" />`)
          break
        }
        const truncated = attachment.resolution.truncated ? ' truncated="true"' : ''
        builder.pushText(`\n<mention path="${mention.path}" type="${mentionType}"${lineRange}${truncated} original_bytes="${attachment.resolution.originalBytes}">${attachment.resolution.content}</mention>`)
        break
      }
    }
  }

  if (attachmentsInsideWrapper) {
    builder.pushText(options.close)
  }

  return builder.build()
}
