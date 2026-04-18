import path from 'node:path'

export interface FileRef {
  path: string
  section?: string
  start: number
  end: number
  raw: string
}

function collectIgnoredRanges(text: string): Array<{ start: number; end: number }> {
  const ranges: Array<{ start: number; end: number }> = []

  // Fenced code blocks (``` ... ```)
  const fenceRegex = /```[\s\S]*?```/g
  for (const match of text.matchAll(fenceRegex)) {
    const start = match.index ?? -1
    if (start >= 0) {
      ranges.push({ start, end: start + match[0].length })
    }
  }

  // Inline code (`...`)
  const inlineRegex = /`[^`\n]*`/g
  for (const match of text.matchAll(inlineRegex)) {
    const start = match.index ?? -1
    if (start >= 0) {
      ranges.push({ start, end: start + match[0].length })
    }
  }

  ranges.sort((a, b) => a.start - b.start)
  return ranges
}

function isIgnoredIndex(index: number, ranges: Array<{ start: number; end: number }>, cursor: { value: number }): boolean {
  while (cursor.value < ranges.length && ranges[cursor.value].end <= index) {
    cursor.value += 1
  }

  if (cursor.value >= ranges.length) return false
  const range = ranges[cursor.value]
  return index >= range.start && index < range.end
}


function splitSection(candidate: string): { refPath: string; section?: string } {
  const hashIndex = candidate.indexOf('#')
  if (hashIndex < 0) return { refPath: candidate }
  const refPath = candidate.slice(0, hashIndex)
  const section = candidate.slice(hashIndex + 1)
  return section.length > 0 ? { refPath, section } : { refPath }
}

export function normalizeReferencedPath(refPath: string): string | null {
  let normalized = refPath

  if (normalized.startsWith('./')) {
    normalized = normalized.slice(2)
  }

  const explicitWorkspacePrefix = normalized.startsWith('$M/') || normalized.startsWith('${M}/')
  const prefix = normalized.startsWith('${M}/') ? '${M}/' : normalized.startsWith('$M/') ? '$M/' : ''
  const body = explicitWorkspacePrefix ? normalized.slice(prefix.length) : normalized

  if (body.length === 0) return null

  const normalizedBody = path.posix.normalize(body)

  if (normalizedBody === '..' || normalizedBody.startsWith('../')) {
    return null
  }

  // Absolute paths are allowed — they resolve to themselves
  // when processed by resolveFileRefPath

  if (explicitWorkspacePrefix) {
    return `${prefix}${normalizedBody}`
  }

  return normalizedBody
}

export function scanFileRefs(text: string): FileRef[] {
  const refs: FileRef[] = []
  const ignoredRanges = collectIgnoredRanges(text)
  const ignoredCursor = { value: 0 }
  const linkRegex = /\[[^\]\n]*\]\(([^)\n]+)\)/g

  for (const match of text.matchAll(linkRegex)) {
    const start = match.index ?? -1
    if (start < 0) continue
    if (isIgnoredIndex(start, ignoredRanges, ignoredCursor)) continue

    const raw = match[0]
    const target = match[1]?.trim()
    if (!target) continue
    if (target.includes('://')) continue

    const { refPath, section } = splitSection(target)
    const normalizedPath = normalizeReferencedPath(refPath)
    if (!normalizedPath) continue

    refs.push({
      path: normalizedPath,
      section,
      start,
      end: start + raw.length,
      raw,
    })
  }

  return refs
}
