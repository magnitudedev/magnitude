import stringWidth from 'string-width'
import type { Span } from './blocks'

export type TableAlignment = 'left' | 'center' | 'right' | null
export type TableWidthMode = 'content' | 'full'
export type TableFitterMode = 'proportional' | 'balanced'
export type TableWrapMode = 'none' | 'word' | 'char'

export interface TableBorderOptions {
  outer: boolean
  inner: boolean
}

export interface TableLayoutOptions {
  headers: Span[][]
  rows: Span[][][]
  alignments?: TableAlignment[]
  availableWidth: number
  fixedColumnWidths?: number[]
  minColumnWidth?: number
  widthMode?: TableWidthMode
  fitter?: TableFitterMode
  wrapMode?: TableWrapMode
  cellPadding?: number
  borders?: TableBorderOptions
}

export interface TableLayoutCellLine {
  spans: Span[]
  width: number
}

export interface TableLayoutCell {
  lines: TableLayoutCellLine[]
}

export interface TableLayoutRow {
  cells: TableLayoutCell[]
  height: number
}

export interface TableLayoutPlan {
  columnWidths: number[]
  naturalColumnWidths: number[]
  alignments: TableAlignment[]
  headers: TableLayoutRow
  rows: TableLayoutRow[]
  tableWidth: number
  contentBudget: number
  wrapMode: TableWrapMode
  border: Required<TableBorderOptions>
  cellPadding: number
}

const DEFAULT_MIN_COLUMN_WIDTH = 3
const DEFAULT_CELL_PADDING = 1

function spansToText(spans: Span[]): string {
  return spans.map((s) => s.text).join('')
}

function clampInt(v: number, min: number): number {
  return Math.max(min, Math.floor(v))
}

function repeatSpaces(width: number): Span[] {
  if (width <= 0) return []
  return [{ text: ' '.repeat(width) }]
}

function fitTextToWidth(text: string, width: number): string {
  if (width <= 0) return ''
  let out = ''
  let used = 0
  for (const ch of text) {
    const w = stringWidth(ch)
    if (used + w > width) break
    out += ch
    used += w
  }
  return out
}

function splitSpanToWidth(span: Span, width: number): { fit: Span | null; rest: Span | null } {
  if (width <= 0) return { fit: null, rest: { ...span } }
  let out = ''
  let used = 0
  let idx = 0
  const chars = [...span.text]
  for (; idx < chars.length; idx++) {
    const ch = chars[idx]
    const w = stringWidth(ch)
    if (used + w > width) break
    out += ch
    used += w
  }
  const fit = out ? { ...span, text: out } : null
  const remaining = chars.slice(idx).join('')
  const rest = remaining ? { ...span, text: remaining } : null
  return { fit, rest }
}

function normalizeRows(headers: Span[][], rows: Span[][][]): Span[][][] {
  const allRows = [headers, ...rows]
  const cols = Math.max(0, ...allRows.map((r) => r.length))
  return allRows.map((row) => {
    const normalized = row.slice()
    while (normalized.length < cols) normalized.push([])
    return normalized
  })
}

function measureIntrinsicWidths(allRows: Span[][][], minColumnWidth: number): number[] {
  const cols = Math.max(0, ...allRows.map((r) => r.length))
  const natural = Array(cols).fill(minColumnWidth)
  for (const row of allRows) {
    for (let c = 0; c < cols; c++) {
      const w = stringWidth(spansToText(row[c] ?? []))
      natural[c] = Math.max(natural[c], w, minColumnWidth)
    }
  }
  return natural
}

function computeGeometry(columns: number, padding: number, border: Required<TableBorderOptions>) {
  const outerCount = border.outer ? 2 : 0
  const innerCount = border.inner ? Math.max(0, columns - 1) : 0
  const boundaryCount = outerCount + innerCount
  const paddingCost = columns * padding * 2
  return {
    boundaryCount,
    paddingCost,
    nonContent: boundaryCount + paddingCost,
  }
}

function distributeGrow(base: number[], targetTotal: number): number[] {
  const out = base.slice()
  let remaining = targetTotal - out.reduce((a, b) => a + b, 0)
  let i = 0
  while (remaining > 0 && out.length > 0) {
    out[i % out.length] += 1
    remaining--
    i++
  }
  return out
}

function shrinkProportional(widths: number[], mins: number[], targetTotal: number): number[] {
  const out = widths.slice()
  let current = out.reduce((a, b) => a + b, 0)
  if (current <= targetTotal) return out

  while (current > targetTotal) {
    const flex = out.map((w, i) => Math.max(0, w - mins[i]))
    const totalFlex = flex.reduce((a, b) => a + b, 0)
    if (totalFlex <= 0) break
    let changed = false
    for (let i = 0; i < out.length && current > targetTotal; i++) {
      if (flex[i] <= 0) continue
      const share = Math.max(1, Math.floor((flex[i] / totalFlex) * (current - targetTotal)))
      const dec = Math.min(share, out[i] - mins[i], current - targetTotal)
      if (dec > 0) {
        out[i] -= dec
        current -= dec
        changed = true
      }
    }
    if (!changed) break
  }

  let idx = 0
  while (current > targetTotal && idx < out.length * 4) {
    const i = idx % out.length
    if (out[i] > mins[i]) {
      out[i]--
      current--
    }
    idx++
  }

  return out
}

function shrinkBalanced(widths: number[], mins: number[], targetTotal: number): number[] {
  const out = widths.slice()
  let current = out.reduce((a, b) => a + b, 0)
  if (current <= targetTotal) return out

  while (current > targetTotal) {
    const idx = out
      .map((w, i) => ({ i, w, flex: w - mins[i] }))
      .filter((x) => x.flex > 0)
      .sort((a, b) => b.w - a.w)[0]?.i
    if (idx == null) break
    out[idx]--
    current--
  }

  return out
}

function fitColumnWidths(
  natural: number[],
  availableContentWidth: number,
  minColumnWidth: number,
  widthMode: TableWidthMode,
  fitter: TableFitterMode,
): number[] {
  const mins = natural.map(() => minColumnWidth)
  const naturalTotal = natural.reduce((a, b) => a + b, 0)
  if (availableContentWidth <= 0) return mins.slice()

  if (naturalTotal <= availableContentWidth) {
    if (widthMode === 'full') return distributeGrow(natural, availableContentWidth)
    return natural
  }

  if (fitter === 'balanced') return shrinkBalanced(natural, mins, availableContentWidth)
  return shrinkProportional(natural, mins, availableContentWidth)
}

function splitByWord(text: string): string[] {
  const parts = text.match(/(\s+|\S+)/g)
  return parts ? parts : [text]
}

function wrapText(text: string, width: number, mode: TableWrapMode): string[] {
  if (width <= 0) return ['']
  if (!text) return ['']

  if (mode === 'none') {
    const clipped = fitTextToWidth(text, width)
    return [clipped]
  }

  const units = mode === 'char' ? [...text] : splitByWord(text)
  const lines: string[] = []
  let current = ''
  let currentW = 0

  for (const unit of units) {
    const uw = stringWidth(unit)
    if (currentW + uw <= width) {
      current += unit
      currentW += uw
      continue
    }

    if (current) {
      lines.push(current)
      current = ''
      currentW = 0
    }

    if (uw <= width) {
      current = unit
      currentW = uw
      continue
    }

    let rem = unit
    while (rem) {
      const chunk = fitTextToWidth(rem, width)
      if (!chunk) break
      lines.push(chunk)
      rem = rem.slice(chunk.length)
      if (stringWidth(rem) <= width) {
        current = rem
        currentW = stringWidth(rem)
        rem = ''
      }
    }
  }

  if (current || lines.length === 0) lines.push(current)
  return lines
}

function alignCellLine(spans: Span[], width: number, align: TableAlignment): Span[] {
  const contentWidth = stringWidth(spansToText(spans))
  if (contentWidth >= width) return spans
  const pad = width - contentWidth

  if (align === 'right') return [...repeatSpaces(pad), ...spans]
  if (align === 'center') {
    const left = Math.floor(pad / 2)
    const right = pad - left
    return [...repeatSpaces(left), ...spans, ...repeatSpaces(right)]
  }
  return [...spans, ...repeatSpaces(pad)]
}

function splitSpansIntoLinesChar(spans: Span[], width: number): Span[][] {
  if (width <= 0) return [[]]
  if (spans.length === 0) return [[]]

  const lines: Span[][] = []
  let line: Span[] = []
  let lineWidth = 0

  const pushLine = () => {
    lines.push(line)
    line = []
    lineWidth = 0
  }

  for (const span of spans) {
    let rest: Span | null = { ...span }
    while (rest) {
      const remainingWidth = width - lineWidth
      if (remainingWidth <= 0) pushLine()
      const { fit, rest: next } = splitSpanToWidth(rest, width - lineWidth)
      if (!fit) {
        pushLine()
        continue
      }
      line.push(fit)
      lineWidth += stringWidth(fit.text)
      rest = next
      if (lineWidth >= width && rest) pushLine()
    }
  }

  if (line.length > 0 || lines.length === 0) lines.push(line)
  return lines
}

function wrapCellSpans(spans: Span[], width: number, mode: TableWrapMode): Span[][] {
  if (width <= 0) return [[]]
  if (spans.length === 0) return [[]]

  if (mode === 'none') {
    const fullWidth = stringWidth(spansToText(spans))
    if (fullWidth <= width) return [spans]

    if (width <= 1) return [[{ text: '…' }]]

    const out: Span[] = []
    let remain = width - 1
    for (const span of spans) {
      if (remain <= 0) break
      const { fit } = splitSpanToWidth(span, remain)
      if (!fit) break
      out.push(fit)
      remain -= stringWidth(fit.text)
    }
    out.push({ text: '…' })
    return [out]
  }

  if (mode === 'char') return splitSpansIntoLinesChar(spans, width)

  const text = spansToText(spans)
  const wrappedTextLines = wrapText(text, width, 'word')
  const sourceQueue = spans.map((s) => ({ ...s }))
  const lines: Span[][] = []

  for (const target of wrappedTextLines) {
    let remaining = target
    const line: Span[] = []

    while (remaining.length > 0 && sourceQueue.length > 0) {
      const current = sourceQueue[0]
      if (!current.text.length) {
        sourceQueue.shift()
        continue
      }

      if (current.text.length <= remaining.length) {
        line.push({ ...current })
        remaining = remaining.slice(current.text.length)
        sourceQueue.shift()
      } else {
        line.push({ ...current, text: current.text.slice(0, remaining.length) })
        sourceQueue[0] = { ...current, text: current.text.slice(remaining.length) }
        remaining = ''
      }
    }

    lines.push(line)
  }

  return lines.length > 0 ? lines : [[]]
}

function buildRowLayout(row: Span[][], widths: number[], alignments: TableAlignment[], wrapMode: TableWrapMode): TableLayoutRow {
  const cells = row.map((cell, idx) => {
    const w = widths[idx] ?? 0
    const wrapped = wrapCellSpans(cell, w, wrapMode).map((lineSpans) => {
      const aligned = alignCellLine(lineSpans, w, alignments[idx] ?? null)
      return { spans: aligned, width: w }
    })
    return { lines: wrapped.length > 0 ? wrapped : [{ spans: repeatSpaces(w), width: w }] }
  })

  const height = Math.max(1, ...cells.map((c) => c.lines.length))
  for (const cell of cells) {
    while (cell.lines.length < height) {
      const w = cell.lines[0]?.width ?? 0
      cell.lines.push({ spans: repeatSpaces(w), width: w })
    }
  }

  return { cells, height }
}

export function computeTableLayoutPlan(options: TableLayoutOptions): TableLayoutPlan {
  const minColumnWidth = clampInt(options.minColumnWidth ?? DEFAULT_MIN_COLUMN_WIDTH, 1)
  const cellPadding = clampInt(options.cellPadding ?? DEFAULT_CELL_PADDING, 0)
  const widthMode = options.widthMode ?? 'content'
  const fitter = options.fitter ?? 'balanced'
  const wrapMode = options.wrapMode ?? 'word'
  const border: Required<TableBorderOptions> = {
    outer: options.borders?.outer ?? true,
    inner: options.borders?.inner ?? true,
  }

  const normalizedRows = normalizeRows(options.headers, options.rows)
  const [headerRow, ...bodyRows] = normalizedRows
  const columnCount = headerRow?.length ?? 0
  const alignments: TableAlignment[] = Array.from({ length: columnCount }, (_, i) => options.alignments?.[i] ?? null)
  const naturalColumnWidths = measureIntrinsicWidths(normalizedRows, minColumnWidth)

  const geometry = computeGeometry(columnCount, cellPadding, border)
  const totalWidth = Math.max(1, options.availableWidth)
  const contentBudget = Math.max(columnCount * minColumnWidth, totalWidth - geometry.nonContent)
  const columnWidths = options.fixedColumnWidths && options.fixedColumnWidths.length === columnCount
    ? options.fixedColumnWidths.map((w) => Math.max(minColumnWidth, Math.floor(w)))
    : fitColumnWidths(naturalColumnWidths, contentBudget, minColumnWidth, widthMode, fitter)

  const headers = buildRowLayout(headerRow ?? [], columnWidths, alignments, wrapMode)
  const rows = bodyRows.map((row) => buildRowLayout(row, columnWidths, alignments, wrapMode))
  const tableWidth = columnWidths.reduce((a, b) => a + b, 0) + geometry.nonContent

  return {
    columnWidths,
    naturalColumnWidths,
    alignments,
    headers,
    rows,
    tableWidth,
    contentBudget,
    wrapMode,
    border,
    cellPadding,
  }
}
