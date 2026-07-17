import { homedir } from "node:os"
import { extname, isAbsolute, relative, resolve, sep } from "node:path"
import { statSync } from "node:fs"
import { Effect } from "effect"
import { createId } from "@magnitudedev/generate-id"
import { SessionOperationFailed, type MentionAttachment, type RawMentionOccurrence, type SessionError } from "@magnitudedev/protocol"

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
  // Mention placements use JavaScript string offsets (UTF-16 code units).
  // split("") preserves that coordinate system; spreading would collapse
  // surrogate pairs and shift every span after an astral character.
  const chars = text.split("")
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

interface InlineMentionCandidate {
  readonly raw: string
  readonly start: number
  readonly end: number
}

function extractInlineMentionCandidates(text: string): InlineMentionCandidate[] {
  const masked = maskCode(text)
  const candidates: InlineMentionCandidate[] = []
  const regex = /(^|[\s([{])@([^\s<>"'`]+)/g
  let match: RegExpExecArray | null
  while ((match = regex.exec(masked)) !== null) {
    const raw = match[2]
    if (!raw) continue
    if (!looksFileLike(raw)) continue
    const start = match.index + (match[1]?.length ?? 0)
    const stripped = stripTrailingPunctuation(raw)
    candidates.push({ raw, start, end: start + 1 + stripped.length })
  }
  return candidates
}

function expandExplicitMentionRange(mention: MentionAttachment): MentionAttachment {
  if (mention.type !== "mention_file_range") return mention
  const range = expandLineRange({ start: mention.startLine, end: mention.endLine })
  return { ...mention, startLine: range.start, endLine: range.end }
}

function resolveInlineMentionCandidate(
  cwd: string,
  candidate: string,
  allowedPrefixes: readonly string[],
): MentionAttachment | null {
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
      return attachment
    }

    const attachment: MentionAttachment = parsed.lineRange
      ? {
          type: "mention_file_range",
          path: parsed.path,
          startLine: parsed.lineRange.start,
          endLine: parsed.lineRange.end,
        }
      : { type: "mention_file", path: parsed.path }
    return attachment
  }

  return null
}

function overlaps(start: number, end: number, otherStart: number, otherEnd: number): boolean {
  return start < otherEnd && otherStart < end
}

function validateProvidedOccurrences(content: string, provided: readonly RawMentionOccurrence[]): RawMentionOccurrence[] {
  const inline = provided
    .filter((item): item is RawMentionOccurrence & { placement: { _tag: 'inline'; start: number; end: number } } => item.placement._tag === 'inline')
    .sort((a, b) => a.placement.start - b.placement.start)
  let previousEnd = -1
  for (const occurrence of inline) {
    const { start, end } = occurrence.placement
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start || end > content.length) {
      throw new Error(`Invalid mention span ${start}-${end}`)
    }
    if (start < previousEnd) throw new Error(`Overlapping mention span ${start}-${end}`)
    if (!content.slice(start, end).startsWith('@')) throw new Error(`Mention span ${start}-${end} does not cover an @ mention`)
    previousEnd = end
  }
  return [...inline, ...provided.filter(item => item.placement._tag === 'trailing')]
}

export function collectMentionOccurrences(
  cwd: string,
  scratchpadPath: string,
  content: string,
  provided: readonly RawMentionOccurrence[],
): Effect.Effect<RawMentionOccurrence[], SessionError> {
  return Effect.try({
    try: () => {
      const allowedPrefixes = scratchpadPath ? [scratchpadPath] : []
      const validated = validateProvidedOccurrences(content, provided).map(item => ({
        ...item,
        attachment: expandExplicitMentionRange(item.attachment),
      }))
      const inline = validated.filter((item): item is RawMentionOccurrence & { placement: { _tag: 'inline'; start: number; end: number } } => item.placement._tag === 'inline')
      const discovered: Array<RawMentionOccurrence & { placement: { _tag: 'inline'; start: number; end: number } }> = []

      for (const candidate of extractInlineMentionCandidates(content)) {
        if (inline.some(item => overlaps(candidate.start, candidate.end, item.placement.start, item.placement.end))) continue
        const resolved = resolveInlineMentionCandidate(cwd, candidate.raw, allowedPrefixes)
        if (!resolved) continue
        discovered.push({
          occurrenceId: createId(),
          attachment: resolved,
          placement: { _tag: 'inline', start: candidate.start, end: candidate.end },
        })
      }

      const orderedInline: RawMentionOccurrence[] = [...inline, ...discovered]
        .sort((a, b) => a.placement.start - b.placement.start)
      const trailing = validated.filter(item => item.placement._tag === 'trailing')
      return [...orderedInline, ...trailing]
    },
    catch: cause => new SessionOperationFailed({
      operation: 'collect message mentions',
      reason: cause instanceof Error ? cause.message : String(cause),
    }),
  })
}
