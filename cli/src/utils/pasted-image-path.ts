import os from 'os'
import path from 'path'
import type { ImageMediaType } from '@magnitudedev/agent'
import { extractImageDimensions } from './clipboard'

export interface PastedImageFileResult {
  path: string
  filename: string
  base64: string
  mediaType: ImageMediaType
  width: number
  height: number
}

const SUPPORTED_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'])

function splitLines(value: string): string[] {
  return value.split(/\r\n|\n|\r/)
}

function tokenizeShellWords(value: string): string[] {
  const tokens: string[] = []
  let current = ''
  let inSingle = false
  let inDouble = false
  let escaping = false

  for (const char of value) {
    if (escaping) {
      current += char
      escaping = false
      continue
    }

    if (char === '\\') {
      escaping = true
      continue
    }

    if (!inDouble && char === "'") {
      inSingle = !inSingle
      continue
    }

    if (!inSingle && char === '"') {
      inDouble = !inDouble
      continue
    }

    if (!inSingle && !inDouble && /\s/.test(char)) {
      if (current.trim().length > 0) {
        tokens.push(current)
      }
      current = ''
      continue
    }

    current += char
  }

  if (escaping) current += '\\'
  if (current.trim().length > 0) tokens.push(current)

  return tokens
}

function normalizePastedPath(raw: string): string | null {
  if (!raw) return null

  const trimmed = raw.trim()
  if (!trimmed) return null

  let value = trimmed

  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    value = value.slice(1, -1).trim()
  }

  if (!value) return null

  if (value.startsWith('file://')) {
    try {
      const url = new URL(value)
      if (url.protocol !== 'file:') return null
      value = decodeURIComponent(url.pathname)
      if (process.platform === 'win32' && value.startsWith('/')) {
        value = value.slice(1)
      }
    } catch {
      return null
    }
  }

  value = value.replace(/\\ /g, ' ')

  if (value.startsWith('~')) {
    if (value === '~') value = os.homedir()
    else if (value.startsWith('~/')) value = path.join(os.homedir(), value.slice(2))
  }

  return path.resolve(value)
}

function dedupeOrdered(values: string[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const value of values) {
    if (seen.has(value)) continue
    seen.add(value)
    out.push(value)
  }
  return out
}

export function extractPastedPathCandidates(rawPasteText: string): string[] {
  const raw = rawPasteText.trim()
  if (!raw) return []

  const uriLines = splitLines(raw)
    .map(line => line.trim())
    .filter(line => line.length > 0 && !line.startsWith('#'))
  if (uriLines.length > 0 && uriLines.every(line => line.startsWith('file://'))) {
    return dedupeOrdered(
      uriLines
        .map(normalizePastedPath)
        .filter((value): value is string => value != null),
    )
  }

  if (/[\r\n]/.test(raw)) {
    return dedupeOrdered(
      raw
        .split(/\r\n|\n|\r|\t|\0/g)
        .map(token => normalizePastedPath(token))
        .filter((value): value is string => value != null),
    )
  }

  const shellWordTokens = tokenizeShellWords(raw)
  if (shellWordTokens.length > 1) {
    return dedupeOrdered(
      shellWordTokens
        .map(token => normalizePastedPath(token))
        .filter((value): value is string => value != null),
    )
  }

  const normalizedSingle = normalizePastedPath(raw)
  return normalizedSingle ? [normalizedSingle] : []
}

function isSupportedImageExtension(filePath: string): boolean {
  return SUPPORTED_EXTENSIONS.has(path.extname(filePath).toLowerCase())
}

function mimeFromExtension(filePath: string): ImageMediaType | null {
  const ext = path.extname(filePath).toLowerCase()
  if (ext === '.png') return 'image/png'
  if (ext === '.jpg' || ext === '.jpeg') return 'image/jpeg'
  if (ext === '.gif') return 'image/gif'
  if (ext === '.webp') return 'image/webp'
  return null
}

export async function tryReadPastedImageFileCandidate(
  candidatePath: string,
): Promise<PastedImageFileResult | null> {
  const normalizedPath = normalizePastedPath(candidatePath)
  if (!normalizedPath) return null

  if (!isSupportedImageExtension(normalizedPath)) return null

  const file = Bun.file(normalizedPath)
  if (!(await file.exists())) return null
  if (file.type === 'application/x-directory') return null

  const mediaType = mimeFromExtension(normalizedPath)
  if (!mediaType) return null

  const buffer = Buffer.from(await file.arrayBuffer())
  if (buffer.length === 0) return null

  const dimensions = extractImageDimensions(buffer)
  if (!dimensions) return null

  return {
    path: normalizedPath,
    filename: path.basename(normalizedPath),
    base64: buffer.toString('base64'),
    mediaType,
    width: dimensions.width,
    height: dimensions.height,
  }
}

export async function tryReadPastedImageFile(
  rawPasteText: string,
): Promise<PastedImageFileResult | null> {
  return tryReadPastedImageFileCandidate(rawPasteText)
}
