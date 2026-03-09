/**
 * Hashline format — LINE#HASH anchored edits.
 * LLM sees file with LINENUM#HASH:content prefixes, returns JSON edits referencing anchors.
 *
 * Hash computation ported from oh-my-pi's hashline.ts (xxHash32, NIBBLE_STR alphabet).
 */
import type { EditFormat } from './types'

// =============================================================================
// Hash computation (ported from oh-my-pi)
// =============================================================================

const NIBBLE_STR = 'ZPMQVRWSNKTXJBYH'

const DICT = Array.from({ length: 256 }, (_, i) => {
  const h = i >>> 4
  const l = i & 0x0f
  return `${NIBBLE_STR[h]}${NIBBLE_STR[l]}`
})

const RE_SIGNIFICANT = /[\p{L}\p{N}]/u

function computeLineHash(idx: number, line: string): string {
  if (line.endsWith('\r')) line = line.slice(0, -1)
  line = line.replace(/\s+/g, '')
  let seed = 0
  if (!RE_SIGNIFICANT.test(line)) seed = idx
  return DICT[Bun.hash.xxHash32(line, seed) & 0xff]
}

function formatHashLines(text: string): string {
  const lines = text.split('\n')
  return lines
    .map((line, i) => {
      const num = i + 1
      return `${num}#${computeLineHash(num, line)}:${line}`
    })
    .join('\n')
}

// =============================================================================
// Edit parsing & application
// =============================================================================

interface Anchor { line: number; hash: string }

interface HashlineEditOp {
  op: 'replace' | 'append' | 'prepend'
  pos?: Anchor
  end?: Anchor
  lines: string[]
}

function parseAnchor(ref: string): Anchor {
  const match = ref.match(/^\s*(\d+)\s*#\s*([ZPMQVRWSNKTXJBYH]{2})/)
  if (!match) throw new Error(`Invalid anchor "${ref}". Expected "LINE#ID" (e.g. "5#ZK").`)
  return { line: parseInt(match[1], 10), hash: match[2] }
}

function parseEditsFromResponse(response: string): HashlineEditOp[] {
  let text = response.trim()
  // Strip markdown code fences
  const fenceMatch = text.match(/```(?:json)?\n([\s\S]*?)```/)
  if (fenceMatch) text = fenceMatch[1].trim()

  const parsed = JSON.parse(text)
  const rawEdits: unknown[] = parsed.edits ?? (Array.isArray(parsed) ? parsed : [parsed])

  return rawEdits.map((raw: unknown) => {
    const e = raw as Record<string, unknown>
    const op = e.op as string
    const pos = e.pos ? parseAnchor(String(e.pos)) : undefined
    const end = e.end ? parseAnchor(String(e.end)) : undefined

    let lines: string[]
    if (e.lines === null || (Array.isArray(e.lines) && e.lines.length === 0)) {
      lines = []
    } else if (typeof e.lines === 'string') {
      lines = [e.lines]
    } else if (Array.isArray(e.lines)) {
      lines = e.lines.map(String)
    } else {
      lines = []
    }

    return { op: op as HashlineEditOp['op'], pos, end, lines }
  })
}

function applyEdits(text: string, edits: HashlineEditOp[]): string {
  const fileLines = text.split('\n')

  // Validate all anchors first
  for (const edit of edits) {
    if (edit.pos) {
      if (edit.pos.line < 1 || edit.pos.line > fileLines.length) {
        throw new Error(`Line ${edit.pos.line} out of range (file has ${fileLines.length} lines)`)
      }
      const actual = computeLineHash(edit.pos.line, fileLines[edit.pos.line - 1])
      if (actual !== edit.pos.hash) {
        throw new Error(`Hash mismatch at line ${edit.pos.line}: expected ${edit.pos.hash}, got ${actual}`)
      }
    }
    if (edit.end) {
      if (edit.end.line < 1 || edit.end.line > fileLines.length) {
        throw new Error(`Line ${edit.end.line} out of range (file has ${fileLines.length} lines)`)
      }
      const actual = computeLineHash(edit.end.line, fileLines[edit.end.line - 1])
      if (actual !== edit.end.hash) {
        throw new Error(`Hash mismatch at line ${edit.end.line}: expected ${edit.end.hash}, got ${actual}`)
      }
    }
  }

  // Sort bottom-up
  const sorted = [...edits].sort((a, b) => {
    const lineA = a.op === 'replace' ? (a.end?.line ?? a.pos?.line ?? 0) : (a.pos?.line ?? 0)
    const lineB = b.op === 'replace' ? (b.end?.line ?? b.pos?.line ?? 0) : (b.pos?.line ?? 0)
    return lineB - lineA
  })

  for (const edit of sorted) {
    switch (edit.op) {
      case 'replace': {
        if (!edit.pos) throw new Error('replace requires pos')
        const start = edit.pos.line - 1
        const count = edit.end ? (edit.end.line - edit.pos.line + 1) : 1
        if (edit.lines.length === 0) {
          // Delete
          fileLines.splice(start, count)
        } else {
          fileLines.splice(start, count, ...edit.lines)
        }
        break
      }
      case 'append': {
        if (edit.pos) {
          fileLines.splice(edit.pos.line, 0, ...edit.lines)
        } else {
          fileLines.push(...edit.lines)
        }
        break
      }
      case 'prepend': {
        if (edit.pos) {
          fileLines.splice(edit.pos.line - 1, 0, ...edit.lines)
        } else {
          fileLines.unshift(...edit.lines)
        }
        break
      }
    }
  }

  return fileLines.join('\n')
}

// =============================================================================
// Format export
// =============================================================================

export const hashlineFormat: EditFormat = {
  id: 'hashline',

  formatFile(_filename: string, content: string): string {
    return formatHashLines(content)
  },

  systemInstructions(): string {
    return [
      '## Edit format: hashline',
      '',
      'The file is shown with `LINE#ID:content` prefixes where ID is a 2-character hash.',
      '',
      'Return a JSON object with an `edits` array. Each edit has:',
      '- `op`: "replace", "append", or "prepend"',
      '- `pos`: anchor string like "23#XY" (copied from the file display)',
      '- `end`: (optional, for range replace) end anchor like "25#ZW"',
      '- `lines`: array of replacement lines (or null to delete)',
      '',
      'Example — single-line replace:',
      '```json',
      '{ "edits": [{ "op": "replace", "pos": "23#XY", "lines": ["  const timeout = 30_000;"] }] }',
      '```',
      '',
      'Example — range replace:',
      '```json',
      '{ "edits": [{ "op": "replace", "pos": "23#XY", "end": "25#ZW", "lines": ["line1", "line2"] }] }',
      '```',
      '',
      'Example — delete line:',
      '```json',
      '{ "edits": [{ "op": "replace", "pos": "23#XY", "lines": null }] }',
      '```',
      '',
      'Return ONLY the JSON. No explanation or commentary.',
    ].join('\n')
  },

  applyResponse(response: string, originalContent: string): string {
    const edits = parseEditsFromResponse(response)
    return applyEdits(originalContent, edits)
  },
}
