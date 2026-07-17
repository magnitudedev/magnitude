import { extname, relative, sep } from 'node:path'
import { Data, Effect } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import type { AppEvent, MentionOccurrence } from '../events'
import { resolveFileRefPath } from '../scratchpad/file-ref-resolution'
import { SessionContextProjection } from '../projections/session-context'
import { Fs } from '../services/fs'
import { captureContextImageFromFile } from '../util/capture-context-image'

const MAX_MENTION_TEXT_BYTES = 500 * 1024
const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.gif'])

class MentionResolutionError extends Data.TaggedError('MentionResolutionError')<{
  readonly message: string
}> {}

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

function resolveTextMention(
  occurrence: MentionOccurrence,
  absolutePath: string,
  fs: Effect.Effect.Success<typeof Fs>,
  lineRange: { start: number; end: number } | null,
) {
  return Effect.gen(function* () {
    const buffer = yield* fs.readFile(absolutePath)
    const truncated = buffer.byteLength > MAX_MENTION_TEXT_BYTES
    const contentBuffer = truncated ? buffer.subarray(0, MAX_MENTION_TEXT_BYTES) : buffer
    let content = contentBuffer.toString('utf8')

    if (lineRange) {
      const lines = content.split('\n')
      const start = Math.max(1, lineRange.start)
      const end = Math.min(lines.length, lineRange.end)
      content = start <= end ? lines.slice(start - 1, end).join('\n') : ''
    }

    return {
      occurrenceId: occurrence.occurrenceId,
      status: 'resolved' as const,
      parts: [{ _tag: 'ContextText' as const, text: content }],
      truncated,
    }
  })
}

function resolveDirectoryMention(
  occurrence: MentionOccurrence,
  absolutePath: string,
  fs: Effect.Effect.Success<typeof Fs>,
) {
  return Effect.gen(function* () {
    const entries = yield* fs.walk(absolutePath)
    const lines: string[] = []
    for (const entry of entries) {
      const relPath = relative(absolutePath, entry.fullPath)
      if (!relPath || relPath.startsWith('..')) continue
      lines.push(`<entry path="${relPath}" name="${entry.name}" type="${entry.type}" depth="${entry.depth}" />`)
    }
    return {
      occurrenceId: occurrence.occurrenceId,
      status: 'resolved' as const,
      parts: [{ _tag: 'ContextText' as const, text: `<tree>${lines.join('')}</tree>` }],
      truncated: false,
    }
  })
}

function resolveMention(
  cwd: string,
  scratchpadPath: string,
  occurrence: MentionOccurrence,
  fs: Effect.Effect.Success<typeof Fs>,
  allowedPrefixes?: string[],
) {
  return Effect.gen(function* () {
    const attachment = occurrence.attachment
    const resolved = resolveFileRefPath(attachment.path, cwd, scratchpadPath)
    if (!resolved) return yield* new MentionResolutionError({ message: `Path not found: ${attachment.path}` })

    const absolutePath = resolved.resolvedPath
    if (!isPathAllowed(absolutePath, cwd, allowedPrefixes)) {
      return yield* new MentionResolutionError({ message: `Path is outside cwd: ${attachment.path}` })
    }

    const fileStat = yield* fs.stat(absolutePath)
    if (attachment.type === 'mention_directory') {
      if (!fileStat.isDirectory()) return yield* new MentionResolutionError({ message: `Mention is not a directory: ${attachment.path}` })
      return yield* resolveDirectoryMention(occurrence, absolutePath, fs)
    }
    if (fileStat.isDirectory()) return yield* new MentionResolutionError({ message: `Mention expected file but got directory: ${attachment.path}` })

    if (attachment.type === 'mention_file' && IMAGE_EXTENSIONS.has(extname(attachment.path).toLowerCase())) {
      const image = yield* captureContextImageFromFile({
        absolutePath,
        logicalPath: attachment.path,
      })
      return {
        occurrenceId: occurrence.occurrenceId,
        status: 'resolved' as const,
        parts: [image],
        truncated: false,
      }
    }

    return yield* resolveTextMention(
      occurrence,
      absolutePath,
      fs,
      attachment.type === 'mention_file_range'
        ? { start: attachment.startLine, end: attachment.endLine }
        : null,
    )
  })
}

export const FileMentionResolver = Worker.define<AppEvent>()({
  name: 'FileMentionResolver',

  eventHandlers: {
    user_message: (event, publish, read) => Effect.gen(function* () {
      if (event.mentions.length === 0) {
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
      if (!scratchpadPath) {
        yield* publish({
          type: 'user_message_ready',
          messageId: event.messageId,
          forkId: event.forkId,
          mentionResolutions: event.mentions.map(occurrence => ({
            occurrenceId: occurrence.occurrenceId,
            status: 'failed',
            reason: 'scratchpadPath not available in session context',
          })),
        })
        return
      }

      const mentionResolutions = yield* Effect.forEach(
        event.mentions,
        (occurrence) => {
          if (!cwd) {
            return Effect.succeed({
              occurrenceId: occurrence.occurrenceId,
              status: 'failed' as const,
              reason: 'Missing session cwd',
            })
          }
          return resolveMention(cwd, scratchpadPath, occurrence, fs, [scratchpadPath]).pipe(
            Effect.catchAll((error) => Effect.succeed({
              occurrenceId: occurrence.occurrenceId,
              status: 'failed' as const,
              reason: error instanceof MentionResolutionError ? error.message : String(error),
            })),
          )
        },
        { concurrency: 'unbounded' },
      )

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
        }),
      ),
    ),
  },
})
