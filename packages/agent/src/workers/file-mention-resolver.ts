import { readFile, stat } from 'fs/promises'
import { extname, resolve, relative, sep } from 'path'
import { Effect } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import type { AppEvent, MentionAttachment, ResolvedMention } from '../events'
import { SessionContextProjection } from '../projections/session-context'
import { walk } from '../util/walk'

const MAX_MENTION_TEXT_BYTES = 500 * 1024

const IMAGE_MIME_TYPES: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
}

function isPathUnderCwd(absolutePath: string, cwd: string): boolean {
  const rel = relative(cwd, absolutePath)
  return rel !== '..' && !rel.startsWith(`..${sep}`) && rel !== ''
    ? true
    : absolutePath === cwd
}

function resolveMentionPath(cwd: string, mentionPath: string): string {
  const absolutePath = resolve(cwd, mentionPath)
  if (!isPathUnderCwd(absolutePath, cwd)) {
    throw new Error(`Path is outside cwd: ${mentionPath}`)
  }
  return absolutePath
}

async function resolveTextMention(path: string, absolutePath: string): Promise<ResolvedMention> {
  const buffer = await readFile(absolutePath)
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

async function resolveImageMention(path: string, absolutePath: string): Promise<ResolvedMention> {
  const extension = extname(absolutePath).toLowerCase()
  const mime = IMAGE_MIME_TYPES[extension] ?? 'application/octet-stream'
  const buffer = await readFile(absolutePath)
  const base64 = buffer.toString('base64')
  return {
    path,
    contentType: 'image',
    content: `data:${mime};base64,${base64}`,
  }
}

function escapeXml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
}

async function resolveDirectoryMention(path: string, absolutePath: string): Promise<ResolvedMention> {
  const entries = await walk(absolutePath, absolutePath, 0, undefined, null, { respectGitignore: true })
  const lines = entries.map((entry) =>
    `<entry path="${escapeXml(entry.relativePath)}" name="${escapeXml(entry.name)}" type="${entry.type}" depth="${entry.depth}" />`
  )
  const content = `<fs-tree>${lines.join('')}</fs-tree>`
  return {
    path,
    contentType: 'directory',
    content,
  }
}

async function resolveMention(cwd: string, attachment: MentionAttachment): Promise<ResolvedMention> {
  const absolutePath = resolveMentionPath(cwd, attachment.path)
  const fileStat = await stat(absolutePath)

  if (attachment.contentType === 'directory') {
    if (!fileStat.isDirectory()) throw new Error(`Mention is not a directory: ${attachment.path}`)
    return resolveDirectoryMention(attachment.path, absolutePath)
  }

  if (fileStat.isDirectory()) {
    throw new Error(`Mention expected file but got directory: ${attachment.path}`)
  }

  if (attachment.contentType === 'image') {
    return resolveImageMention(attachment.path, absolutePath)
  }

  return resolveTextMention(attachment.path, absolutePath)
}

export const FileMentionResolver = Worker.define<AppEvent>()({
  name: 'FileMentionResolver',

  eventHandlers: {
    user_message: (event, publish, read) => Effect.gen(function* () {
      const mentions = event.attachments.filter((attachment): attachment is MentionAttachment => attachment.type === 'mention')
      if (mentions.length === 0) return

      const sessionContext = yield* read(SessionContextProjection)
      const cwd = sessionContext.context?.cwd

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
            results.push(await resolveMention(cwd, mention))
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
        type: 'file_mention_resolved',
        forkId: event.forkId,
        sourceMessageTimestamp: event.timestamp,
        mentions: resolvedMentions,
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