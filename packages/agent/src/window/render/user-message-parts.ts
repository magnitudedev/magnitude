import { ContextBuilder, type ContextImagePart, type ContextPart } from '../../content'
import type { TimelineEntry } from '../inbox/types'

export interface RenderTimelineUserMessagePartsOptions {
  readonly open: string
  readonly close: string
}

const defaultUserMessageOptions: RenderTimelineUserMessagePartsOptions = {
  open: '<message from="user">',
  close: '</message>',
}

function imageFrom(parts: readonly ContextPart[]): ContextImagePart | undefined {
  return parts.find((part): part is ContextImagePart => part._tag === 'ContextImage')
}

export function renderTimelineUserMessageParts(
  entry: Extract<TimelineEntry, { kind: 'user_message' }>,
  options: RenderTimelineUserMessagePartsOptions = defaultUserMessageOptions,
): ContextPart[] {
  const builder = new ContextBuilder()
  builder.pushText(options.open)

  for (const item of entry.items) {
    if (item.kind === 'body') {
      for (const part of item.parts) builder.pushPart(part)
      continue
    }

    if (item.kind === 'attachment') {
      const image = imageFrom(item.parts)
      const path = image?.path ?? 'unknown'
      builder.pushText(`\n<attachment type="image" path="${path}">`)
      for (const part of item.parts) builder.pushPart(part)
      builder.pushText('</attachment>')
      continue
    }

    const occurrence = item.mention.occurrence
    const mention = occurrence.attachment
    const mentionType = mention.type === 'mention_directory' ? 'directory' : 'file'
    const lineRange = mention.type === 'mention_file_range'
      ? ` lines="${mention.startLine}-${mention.endLine}"`
      : ''
    const resolution = item.mention.resolution
    if (resolution.status === 'failed') {
      builder.pushText(`<mention path="${mention.path}" type="${mentionType}"${lineRange} status="failed" reason="${resolution.reason}" />`)
      continue
    }
    const truncated = resolution.truncated ? ' truncated="true"' : ''
    builder.pushText(`<mention path="${mention.path}" type="${mentionType}"${lineRange}${truncated}>`)
    for (const part of resolution.parts) builder.pushPart(part)
    builder.pushText('</mention>')
  }

  builder.pushText(options.close)
  return builder.build()
}
