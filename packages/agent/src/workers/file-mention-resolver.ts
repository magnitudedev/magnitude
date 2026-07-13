import { relative, sep } from 'path'
import { Effect } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import type { AppEvent, MentionAttachment, MentionResolution } from '../events'
import { resolveFileRefPath } from '../scratchpad/file-ref-resolution'
import { SessionContextProjection } from '../projections/session-context'
import { Fs } from '../services/fs'

const MAX_MENTION_TEXT_BYTES = 500 * 1024

function isPathUnderPrefix(absolutePath: string, prefix: string): boolean {
  const rel = relative(prefix, absolutePath)
  return rel !== '..' && !rel.startsWith(`..${sep}`) && rel !== ''
    ? true
    : absolutePath === prefix
}

function isPathAllowed(absolutePath: string, cwd: string, allowedPrefixes?: string[]): boolean {
  if (isPathUnderPrefix(absolutePath, cwd)) return true
  if (!allowedPrefixes || allowedPrefixes.length === 0) return false
  return allowedPrefixes.some(prefix => isPathUnderPrefix(absolutePath, prefix))
}

async function resolveTextMention(
  attachment: MentionAttachment,
  absolutePath: string,
  fs: Effect.Effect.Success<typeof Fs>,
  lineRange: { start: number; end: number } | null
): Promise<MentionResolution> {
  const buffer = await Effect.runPromise(fs.readFile(absolutePath))
  const originalBytes = buffer.byteLength
  const truncated = originalBytes > MAX_MENTION_TEXT_BYTES
  const contentBuffer = truncated ? buffer.subarray(0, MAX_MENTION_TEXT_BYTES) : buffer
  let content = contentBuffer.toString('utf8')

  if (lineRange) {
    const lines = content.split('\n')
    // Clamp to available lines (1-indexed, inclusive)
    // The expanded range from CLI may exceed the file's actual line count — clamp end to lines.length
    const start = Math.max(1, lineRange.start)
    const end = Math.min(lines.length, lineRange.end)
    if (start <= end) {
      content = lines.slice(start - 1, end).join('\n')
    } else {
      content = ''
    }
  }

  return {
    status: 'resolved',
    attachment,
    content,
    truncated,
    originalBytes,
  }
}

async function resolveDirectoryMention(attachment: MentionAttachment, absolutePath: string, fs: Effect.Effect.Success<typeof Fs>): Promise<MentionResolution> {
  const entries = await Effect.runPromise(fs.walk(absolutePath))
  const lines: string[] = []
  for (const entry of entries) {
    const relPath = relative(absolutePath, entry.fullPath)
    if (!relPath || relPath.startsWith('..')) continue
    lines.push(`<entry path="${relPath}" name="${entry.name}" type="${entry.type}" depth="${entry.depth}" />`)
  }
  const content = `<tree>${lines.join('')}</tree>`
  return {
    status: 'resolved',
    attachment,
    content,
    truncated: false,
    originalBytes: Buffer.byteLength(content, 'utf8'),
  }
}

async function resolveMention(
  cwd: string,
  scratchpadPath: string,
  attachment: MentionAttachment,
  fs: Effect.Effect.Success<typeof Fs>,
  allowedPrefixes?: string[]
): Promise<MentionResolution> {
  const resolved = resolveFileRefPath(attachment.path, cwd, scratchpadPath)
  if (!resolved) {
    throw new Error(`Path not found: ${attachment.path}`)
  }

  const absolutePath = resolved.resolvedPath
  if (!isPathAllowed(absolutePath, cwd, allowedPrefixes)) {
    throw new Error(`Path is outside cwd: ${attachment.path}`)
  }

  const fileStat = await Effect.runPromise(fs.stat(absolutePath))

  if (attachment.type === 'mention_directory') {
    if (!fileStat.isDirectory()) throw new Error(`Mention is not a directory: ${attachment.path}`)
    return resolveDirectoryMention(attachment, absolutePath, fs)
  }

  if (fileStat.isDirectory()) {
    throw new Error(`Mention expected file but got directory: ${attachment.path}`)
  }

  return resolveTextMention(
    attachment,
    absolutePath,
    fs,
    attachment.type === 'mention_file_range'
      ? { start: attachment.startLine, end: attachment.endLine }
      : null,
  )
}

export const FileMentionResolver = Worker.define<AppEvent>()({
  name: 'FileMentionResolver',

  eventHandlers: {
    user_message: (event, publish, read) => Effect.gen(function* () {
      const mentions = event.attachments.filter((attachment): attachment is MentionAttachment =>
        attachment.type === 'mention_file'
        || attachment.type === 'mention_file_range'
        || attachment.type === 'mention_directory'
      )

      // No mentions — immediate passthrough
      if (mentions.length === 0) {
        yield* publish({
          type: 'user_message_ready',
          messageId: event.messageId,
          forkId: event.forkId,
          mentionResolutions: [],
        })
        return
      }

      const fs = yield* Fs
      const sessionContext = yield* read(SessionContextProjection)
      const cwd = sessionContext.context?.cwd
      const scratchpadPath = sessionContext.context?.scratchpadPath
      if (!scratchpadPath) throw new Error('scratchpadPath not available in session context')

      const mentionResolutions = yield* Effect.promise(async () => {
        const results: MentionResolution[] = []
        for (const mention of mentions) {
          if (!cwd) {
            results.push({
              status: 'failed',
              attachment: mention,
              reason: 'Missing session cwd',
            })
            continue
          }

          try {
            results.push(await resolveMention(cwd, scratchpadPath, mention, fs, [scratchpadPath]))
          } catch (error) {
            results.push({
              status: 'failed',
              attachment: mention,
              reason: error instanceof Error ? error.message : String(error),
            })
          }
        }
        return results
      })

      yield* publish({
        type: 'user_message_ready',
        messageId: event.messageId,
        forkId: event.forkId,
        mentionResolutions,
      })
    }).pipe(
      Effect.catchAllCause(cause =>
        Effect.sync(() => {
          logger.error({ cause: cause.toString() }, '[FileMentionResolver] Unexpected error while resolving file mentions')
        })
      )
    )
  }
})
