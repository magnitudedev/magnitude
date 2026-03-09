/**
 * XML-hybrid-strict format — content-verified anchored range edits.
 *
 * The file is shown with `LINENUM|content` prefixes.
 * Edits use <edit> blocks with <from>, <to>, and <with> sub-elements.
 *
 * <from> and <to> are anchors: they contain numbered lines copied from the file.
 * The replaced range spans from the first line in <from> to the last line in <to>.
 * Every line's content is verified against the actual file — mismatches are errors.
 * <from> and <to> may overlap (e.g. both reference line 5 for a single-line edit).
 *
 * <with> is the replacement content. Line numbers are required.
 */
import type { EditFormat } from './types'

// =============================================================================
// Formatting
// =============================================================================

function formatNumberedLines(content: string): string {
  const lines = content.split('\n')
  return lines.map((line, i) => `${i + 1}|${line}`).join('\n')
}

// =============================================================================
// Parsing
// =============================================================================

interface AnchorLine {
  lineNum: number
  content: string
}

interface HybridEdit {
  fromLines: AnchorLine[]
  toLines: AnchorLine[]
  withContent: string
}

function parseNumberedLine(raw: string): AnchorLine {
  const match = raw.match(/^(\d+)\|(.*)$/)
  if (!match) {
    throw new Error(`Anchor line does not match LINENUM|content format: ${JSON.stringify(raw)}`)
  }
  return { lineNum: parseInt(match[1], 10), content: match[2] }
}

function requireLineNumbers(text: string): string {
  return text
    .split('\n')
    .map(line => {
      const match = line.match(/^\d+\|(.*)$/)
      if (!match) {
        throw new Error(`<with> line must have LINENUM|content format: ${JSON.stringify(line)}`)
      }
      return match[1]
    })
    .join('\n')
}

function parseEdits(response: string): HybridEdit[] {
  const edits: HybridEdit[] = []
  const editRegex = /<edit>([\s\S]*?)<\/edit>/g
  let editMatch: RegExpExecArray | null

  while ((editMatch = editRegex.exec(response)) !== null) {
    const body = editMatch[1]

    const fromMatch = body.match(/<from>\n?([\s\S]*?)\n?<\/from>/)
    const toMatch = body.match(/<to>\n?([\s\S]*?)\n?<\/to>/)
    const withMatch = body.match(/<with>\n?([\s\S]*?)\n?<\/with>/)

    if (!fromMatch || !toMatch || !withMatch) {
      throw new Error('<edit> block missing <from>, <to>, or <with>')
    }

    const fromRaw = fromMatch[1].trim()
    const toRaw = toMatch[1].trim()
    const withRaw = withMatch[1]

    if (!fromRaw) throw new Error('<from> is empty')
    if (!toRaw) throw new Error('<to> is empty')

    const fromLines = fromRaw.split('\n').map(parseNumberedLine)
    const toLines = toRaw.split('\n').map(parseNumberedLine)

    // Strip leading/trailing newline from <with> content, but preserve internal structure
    let withContent = withRaw
    if (withContent.startsWith('\n')) withContent = withContent.slice(1)
    if (withContent.endsWith('\n')) withContent = withContent.slice(0, -1)

    withContent = requireLineNumbers(withContent)

    edits.push({ fromLines, toLines, withContent })
  }

  return edits
}

// =============================================================================
// Application
// =============================================================================

function resolveAndVerify(
  fileLines: string[],
  edit: HybridEdit,
): { rangeStart: number; rangeEnd: number } {
  const firstFrom = edit.fromLines[0]
  const lastTo = edit.toLines[edit.toLines.length - 1]

  const rangeStart = firstFrom.lineNum
  const rangeEnd = lastTo.lineNum

  if (rangeStart < 1 || rangeEnd > fileLines.length) {
    throw new Error(
      `Range ${rangeStart}-${rangeEnd} out of bounds (file has ${fileLines.length} lines)`
    )
  }

  if (rangeStart > rangeEnd) {
    throw new Error(
      `<from> starts at line ${rangeStart} but <to> ends at line ${rangeEnd} — from must not come after to`
    )
  }

  // Verify all <from> anchor lines
  for (const anchor of edit.fromLines) {
    if (anchor.lineNum < 1 || anchor.lineNum > fileLines.length) {
      throw new Error(`<from> line ${anchor.lineNum} out of bounds`)
    }
    const actual = fileLines[anchor.lineNum - 1]
    if (actual !== anchor.content) {
      throw new Error(
        `Content mismatch at line ${anchor.lineNum}:\n  expected: ${JSON.stringify(anchor.content)}\n  actual:   ${JSON.stringify(actual)}`
      )
    }
  }

  // Verify all <to> anchor lines
  for (const anchor of edit.toLines) {
    if (anchor.lineNum < 1 || anchor.lineNum > fileLines.length) {
      throw new Error(`<to> line ${anchor.lineNum} out of bounds`)
    }
    const actual = fileLines[anchor.lineNum - 1]
    if (actual !== anchor.content) {
      throw new Error(
        `Content mismatch at line ${anchor.lineNum}:\n  expected: ${JSON.stringify(anchor.content)}\n  actual:   ${JSON.stringify(actual)}`
      )
    }
  }

  // Verify <from> lines are consecutive and ascending
  for (let i = 1; i < edit.fromLines.length; i++) {
    if (edit.fromLines[i].lineNum !== edit.fromLines[i - 1].lineNum + 1) {
      throw new Error(`<from> lines must be consecutive`)
    }
  }

  // Verify <to> lines are consecutive and ascending
  for (let i = 1; i < edit.toLines.length; i++) {
    if (edit.toLines[i].lineNum !== edit.toLines[i - 1].lineNum + 1) {
      throw new Error(`<to> lines must be consecutive`)
    }
  }

  return { rangeStart, rangeEnd }
}

function applyEdits(originalContent: string, edits: HybridEdit[]): string {
  const fileLines = originalContent.split('\n')

  // Resolve and verify all edits first
  const resolved = edits.map(edit => ({
    edit,
    ...resolveAndVerify(fileLines, edit),
  }))

  // Sort bottom-up to preserve line numbers
  resolved.sort((a, b) => b.rangeStart - a.rangeStart)

  for (const { edit, rangeStart, rangeEnd } of resolved) {
    const replacementLines = edit.withContent === '' ? [] : edit.withContent.split('\n')
    const count = rangeEnd - rangeStart + 1
    fileLines.splice(rangeStart - 1, count, ...replacementLines)
  }

  return fileLines.join('\n')
}

// =============================================================================
// Format export
// =============================================================================

export const xmlHybridStrictFormat: EditFormat = {
  id: 'xml-hybrid-strict',

  formatFile(_filename: string, content: string): string {
    return formatNumberedLines(content)
  },

  systemInstructions(): string {
    return [
      '## Edit format: XML hybrid strict edits',
      '',
      'The file is shown with `LINENUM|content` prefixes (1-indexed).',
      '',
      'Return one or more `<edit>` blocks. Each edit has three parts:',
      '',
      '- `<from>`: One or more lines copied from the file marking the START of the range to replace.',
      '- `<to>`: One or more lines copied from the file marking the END of the range to replace.',
      '- `<with>`: The replacement content. Each line MUST be prefixed with `LINENUM|` (matching the new intended line numbers).',
      '',
      'The replaced range spans from the first `<from>` line to the last `<to>` line (inclusive).',
      '`<from>` and `<to>` may overlap (e.g. for a single-line edit, use the same line in both).',
      '',
      'Example — replace a range:',
      '```',
      '<edit>',
      '<from>',
      '10|  const timeout = 5000;',
      '</from>',
      '<to>',
      '12|  const retries = 3;',
      '</to>',
      '<with>',
      '10|  const timeout = 30_000;',
      '11|  const host = "localhost";',
      '</with>',
      '</edit>',
      '```',
      '',
      'Example — single-line edit:',
      '```',
      '<edit>',
      '<from>',
      '5|  return null;',
      '</from>',
      '<to>',
      '5|  return null;',
      '</to>',
      '<with>',
      '5|  return defaultValue;',
      '</with>',
      '</edit>',
      '```',
      '',
      'Every line in `<from>` and `<to>` MUST exactly match the file. Copy them verbatim.',
      'Every line in `<with>` MUST have a `LINENUM|` prefix.',
      'Return ONLY the edit blocks. No explanation or commentary.',
    ].join('\n')
  },

  applyResponse(response: string, originalContent: string): string {
    const edits = parseEdits(response)
    if (edits.length === 0) {
      throw new Error('No <edit> blocks found in response')
    }
    return applyEdits(originalContent, edits)
  },
}
