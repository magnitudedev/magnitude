import { extname, relative, sep } from 'path'
import { Effect } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import type { AppEvent, MentionAttachment, ResolvedMention } from '../events'
import { resolveFileRefPath } from '../workspace/file-ref-resolution'
import { SessionContextProjection } from '../projections/session-context'
import { Fs } from '../services/fs'

const MAX_MENTION_TEXT_BYTES = 500 * 1024

const IMAGE_MIME_TYPES: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
}

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

async function resolveTextMention(path: string, absolutePath: string, fs: Effect.Effect.Success<typeof Fs>): Promise<ResolvedMention> {
  const buffer = await Effect.runPromise(fs.readFile(absolutePath))
  const originalBytes = buffer.byteLength
  const truncated = originalBytes > MAX_MENTION_TEXT_BYTES
  const contentBuffer = truncated ? buffer.subarray(0, MAX_MENTION_TEXT_BYTES) : buffer
  return {
    path,
    contentType: 'text',
    content: contentBuffer.toString('utf8'),
    truncated: truncated || undefined,
    originalBytes,
  }
}

async function resolveImageMention(path: string, absolutePath: string, fs: Effect.Effect.Success<typeof Fs>): Promise<ResolvedMention> {
  const extension = extname(absolutePath).toLowerCase()
  const mime = IMAGE_MIME_TYPES[extension] ?? 'application/octet-stream'
  const buffer = await Effect.runPromise(fs.readFile(absolutePath))
  const base64 = buffer.toString('base64')
  return {
    path,
    contentType: 'image',
    content: `data:${mime};base64,${base64}`,
  }
}

async function resolveDirectoryMention(path: string, absolutePath: string, fs: Effect.Effect.Success<typeof Fs>): Promise<ResolvedMention> {
  const entries = await Effect.runPromise(fs.walk(absolutePath))
  const lines: string[] = []
  for (const entry of entries) {
    const relPath = relative(absolutePath, entry.fullPath)
    if (!relPath || relPath.startsWith('..')) continue
    lines.push(`<entry path="${relPath}" name="${entry.name}" type="${entry.type}" depth="${entry.depth}" />`)
  }
  const content = `<tree>${lines.join('')}</tree>`
  return {
    path,
    contentType: 'directory',
    content,
  }
}

async function resolveMention(
  cwd: string,
  workspacePath: string,
  attachment: MentionAttachment,
  fs: Effect.Effect.Success<typeof Fs>,
  allowedPrefixes?: string[]
): Promise<ResolvedMention> {
  const resolved = resolveFileRefPath(attachment.path, cwd, workspacePath)
  if (!resolved) {
    throw new Error(`Path not found: ${attachment.path}`)
  }

  const absolutePath = resolved.resolvedPath
  if (!isPathAllowed(absolutePath, cwd, allowedPrefixes)) {
    throw new Error(`Path is outside cwd: ${attachment.path}`)
  }

  const fileStat = await Effect.runPromise(fs.stat(absolutePath))

  if (attachment.contentType === 'directory') {
    if (!fileStat.isDirectory()) throw new Error(`Mention is not a directory: ${attachment.path}`)
    return resolveDirectoryMention(attachment.path, absolutePath, fs)
  }

  if (fileStat.isDirectory()) {
    throw new Error(`Mention expected file but got directory: ${attachment.path}`)
  }

  if (attachment.contentType === 'image') {
    return resolveImageMention(attachment.path, absolutePath, fs)
  }

  return resolveTextMention(attachment.path, absolutePath, fs)
}

export const FileMentionResolver = Worker.define<AppEvent>()({
  name: 'FileMentionResolver',

  eventHandlers: {
    user_message: (event, publish, read) => Effect.gen(function* () {
      const mentions = event.attachments.filter((attachment): attachment is MentionAttachment => attachment.type === 'mention')

      // No mentions — immediate passthrough
      if (mentions.length === 0) {
        yield* publish({
          type: 'user_message_ready',
          messageId: event.messageId,
          forkId: event.forkId,
          resolvedMentions: [],
        })
        return
      }

      const fs = yield* Fs
      const sessionContext = yield* read(SessionContextProjection)
      const cwd = sessionContext.context?.cwd
      const workspacePath = sessionContext.context?.workspacePath
      if (!workspacePath) throw new Error('workspacePath not available in session context')

      const resolvedMentions = yield* Effect.promise(async () => {
        const results: ResolvedMention[] = []
        for (const mention of mentions) {
          if (!cwd) {
            results.push({
              path: mention.path,
              contentType: mention.contentType,
              error: 'Missing session cwd',
            })
            continue
          }

          try {
            results.push(await resolveMention(cwd, workspacePath, mention, fs, [workspacePath]))
          } catch (error) {
            results.push({
              path: mention.path,
              contentType: mention.contentType,
              error: error instanceof Error ? error.message : String(error),
            })
          }
        }
        return results
      })

      yield* publish({
        type: 'user_message_ready',
        messageId: event.messageId,
        forkId: event.forkId,
        resolvedMentions,
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
