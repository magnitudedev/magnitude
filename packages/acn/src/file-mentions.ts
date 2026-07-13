import { homedir } from "node:os"
import { extname, isAbsolute, relative, resolve, sep } from "node:path"
import { statSync } from "node:fs"
import { Effect } from "effect"
import type { MentionAttachment, MessageAttachment, SessionError } from "@magnitudedev/protocol"

const TRAILING_PUNCTUATION = new Set([".", ",", ";", "!", "?", ")", "]", "}"])

function resolveSessionPath(requestedPath: string, cwd: string): string {
  if (requestedPath.startsWith("~/")) return resolve(homedir(), requestedPath.slice(2))
  if (isAbsolute(requestedPath)) return requestedPath
  return resolve(cwd, requestedPath)
}

function isPathUnderPrefix(absolutePath: string, prefix: string): boolean {
  const rel = relative(prefix, absolutePath)
  return rel === "" || (rel !== ".." && !rel.startsWith(`..${sep}`))
}

function isAllowedPath(absolutePath: string, cwd: string, allowedPrefixes: readonly string[]): boolean {
  if (isPathUnderPrefix(absolutePath, cwd)) return true
  return allowedPrefixes.some((prefix) => isPathUnderPrefix(absolutePath, prefix))
}

function expandLineRange(lineRange: { start: number; end: number }): { start: number; end: number } {
  if (lineRange.start !== lineRange.end) return lineRange
  return { start: Math.max(1, lineRange.start - 10), end: lineRange.end + 10 }
}

function parsePathAndRange(raw: string): { path: string; lineRange?: { start: number; end: number } } {
  const rangeMatch = raw.match(/:([\d]+)(?:-([\d]+))?$/)
  if (!rangeMatch || rangeMatch.index === 1) return { path: raw }

  const start = parseInt(rangeMatch[1], 10)
  const end = rangeMatch[2] ? parseInt(rangeMatch[2], 10) : start
  if (!Number.isFinite(start) || !Number.isFinite(end) || start < 1 || end < 1 || end < start) {
    return { path: raw }
  }

  return {
    path: raw.slice(0, rangeMatch.index),
    lineRange: expandLineRange({ start, end }),
  }
}

function looksFileLike(raw: string): boolean {
  return raw.includes("/")
    || raw.includes(".")
    || raw.includes("~")
    || raw.startsWith("./")
    || raw.startsWith("../")
    || /:\d+(?:-\d+)?$/.test(raw)
}

function maskCode(text: string): string {
  const chars = [...text]
  let i = 0
  let inFence = false
  let lineStart = true

  while (i < chars.length) {
    if (lineStart) {
      const rest = chars.slice(i).join("")
      const fence = rest.match(/^([ \t]*)(```|~~~)/)
      if (fence) {
        inFence = !inFence
      }
    }

    if (inFence) {
      if (chars[i] !== "\n") chars[i] = " "
      lineStart = chars[i] === "\n"
      i++
      continue
    }

    if (chars[i] === "`") {
      const start = i
      i++
      while (i < chars.length && chars[i] !== "`") i++
      if (i < chars.length) {
        for (let j = start; j <= i; j++) {
          if (chars[j] !== "\n") chars[j] = " "
        }
        i++
        continue
      }
    }

    lineStart = chars[i] === "\n"
    i++
  }

  return chars.join("")
}

function stripTrailingPunctuation(raw: string): string {
  let value = raw
  while (value.length > 0 && TRAILING_PUNCTUATION.has(value[value.length - 1]!)) {
    value = value.slice(0, -1)
  }
  return value
}

function extractInlineMentionCandidates(text: string): string[] {
  const masked = maskCode(text)
  const candidates: string[] = []
  const regex = /(^|[\s([{])@([^\s<>"'`]+)/g
  let match: RegExpExecArray | null
  while ((match = regex.exec(masked)) !== null) {
    const raw = match[2]
    if (!raw) continue
    if (!looksFileLike(raw)) continue
    candidates.push(raw)
  }
  return candidates
}

function mentionRange(mention: MentionAttachment): { start: number; end: number } | null {
  return mention.type === "mention_file_range"
    ? { start: mention.startLine, end: mention.endLine }
    : null
}

function mentionKeyFromAttachment(cwd: string, mention: MentionAttachment): string {
  const resolved = resolveSessionPath(mention.path, cwd)
  const lineRange = mentionRange(mention)
  return lineRange
    ? `${resolved}:${lineRange.start}-${lineRange.end}`
    : resolved
}

function expandExplicitMentionRange(mention: MentionAttachment): MentionAttachment {
  if (mention.type !== "mention_file_range") return mention
  const range = expandLineRange({ start: mention.startLine, end: mention.endLine })
  return { ...mention, startLine: range.start, endLine: range.end }
}

function mentionKey(resolved: string, lineRange?: { start: number; end: number }): string {
  return lineRange ? `${resolved}:${lineRange.start}-${lineRange.end}` : resolved
}

function resolveInlineMentionCandidate(
  cwd: string,
  candidate: string,
  allowedPrefixes: readonly string[],
): { attachment: MentionAttachment; key: string } | null {
  const attempts = [candidate]
  const stripped = stripTrailingPunctuation(candidate)
  if (stripped !== candidate) attempts.push(stripped)

  for (const attempt of attempts) {
    const parsed = parsePathAndRange(attempt)
    if (!parsed.path || !looksFileLike(parsed.path)) continue

    const absolutePath = resolveSessionPath(parsed.path, cwd)
    if (!isAllowedPath(absolutePath, cwd, allowedPrefixes)) continue

    let info: ReturnType<typeof statSync>
    try {
      info = statSync(absolutePath)
    } catch {
      continue
    }

    if (info.isDirectory()) {
      const attachment: MentionAttachment = { type: "mention_directory", path: parsed.path }
      return { attachment, key: mentionKey(absolutePath) }
    }

    const attachment: MentionAttachment = parsed.lineRange
      ? {
          type: "mention_file_range",
          path: parsed.path,
          startLine: parsed.lineRange.start,
          endLine: parsed.lineRange.end,
        }
      : { type: "mention_file", path: parsed.path }
    return { attachment, key: mentionKey(absolutePath, parsed.lineRange) }
  }

  return null
}

export function mergeInlineMentions(
  cwd: string,
  scratchpadPath: string,
  content: string,
  attachments: readonly MessageAttachment[],
): Effect.Effect<MessageAttachment[], SessionError> {
  return Effect.sync(() => {
    const allowedPrefixes = scratchpadPath ? [scratchpadPath] : []
    const merged: MessageAttachment[] = []
    const seenMentions = new Set<string>()

    for (const attachment of attachments) {
      if (attachment.type === "image") {
        merged.push(attachment)
        continue
      }
      const mention = expandExplicitMentionRange(attachment)
      const key = mentionKeyFromAttachment(cwd, mention)
      if (seenMentions.has(key)) continue
      seenMentions.add(key)
      merged.push(mention)
    }

    for (const candidate of extractInlineMentionCandidates(content)) {
      const resolved = resolveInlineMentionCandidate(cwd, candidate, allowedPrefixes)
      if (!resolved) continue
      if (seenMentions.has(resolved.key)) continue
      seenMentions.add(resolved.key)
      merged.push(resolved.attachment)
    }

    return merged
  })
}
