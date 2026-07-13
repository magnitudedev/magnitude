import { TextAttributes } from '@opentui/core'

import { useTheme } from '../hooks/use-theme'

export type DiffHunkProps = {
  contextBefore?: readonly string[]
  removedLines: readonly string[]
  addedLines: readonly string[]
  contextAfter?: readonly string[]
  streamingCursor?: boolean
  startLine?: number
}

function padLineNum(n?: number): string {
  return n !== undefined ? String(n).padStart(3, ' ') : '   '
}

type DiffRow = {
  lineNum?: number
  prefix: string
  text: string
  bg?: string
  dim?: boolean
}

export function DiffHunk({
  contextBefore = [],
  removedLines,
  addedLines,
  contextAfter = [],
  streamingCursor = false,
  startLine = 1,
}: DiffHunkProps) {
  const theme = useTheme()

  const contextRadius = contextBefore.length
  const lineStart = startLine - contextRadius

  let lineNum = lineStart

  const rows: DiffRow[] = []

  for (const line of contextBefore) {
    rows.push({
      lineNum,
      prefix: ' ',
      text: line,
      dim: true,
    })
    lineNum++
  }

  const addedStartLine = lineNum
  let afterContextStartLine = lineNum

  for (const line of removedLines) {
    rows.push({
      lineNum,
      prefix: '-',
      text: line,
      bg: theme.diffRedBg,
    })
    lineNum++
  }

  afterContextStartLine = lineNum
  lineNum = addedStartLine

  for (let i = 0; i < addedLines.length; i++) {
    const line = addedLines[i]
    const isLast = i === addedLines.length - 1
    rows.push({
      lineNum,
      prefix: '+',
      text: line + (streamingCursor && isLast ? '▍' : ''),
      bg: theme.diffGreenBg,
    })
    lineNum++
  }

  lineNum = afterContextStartLine

  for (const line of contextAfter) {
    rows.push({
      lineNum,
      prefix: ' ',
      text: line,
      dim: true,
    })
    lineNum++
  }

  return (
    <box style={{ flexDirection: 'column' }}>
      {rows.map((row, index) => (
        <box
          key={`row-${index}`}
          style={{
            flexDirection: 'row',
            backgroundColor: row.bg,
          }}
        >
          <text
            style={{ fg: theme.foreground }}
            attributes={row.dim ? TextAttributes.DIM : undefined}
          >
            <span>{`${padLineNum(row.lineNum)} │ ${row.prefix} ${row.text}`}</span>
          </text>
        </box>
      ))}
    </box>
  )
}
