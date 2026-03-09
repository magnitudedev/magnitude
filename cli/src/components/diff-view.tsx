import { memo, useMemo } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import type { EditDiff } from '@magnitudedev/agent'

interface DiffViewProps {
  diffs: readonly EditDiff[]
  dimmed?: boolean
}

interface DiffLine {
  lineNum: number
  sign: '+' | '-'
  content: string
  type: 'added' | 'removed'
}

export const DiffView = memo(function DiffView({ diffs, dimmed = false }: DiffViewProps) {
  const theme = useTheme()

  // Build flat list of lines with line numbers and separators
  const { lines, gutterWidth } = useMemo(() => {
    const allLines: (DiffLine | 'separator')[] = []
    let maxLineNum = 0

    for (let di = 0; di < diffs.length; di++) {
      const diff = diffs[di]
      if (di > 0) allLines.push('separator')

      for (let li = 0; li < diff.removedLines.length; li++) {
        const lineNum = diff.startLine + li
        maxLineNum = Math.max(maxLineNum, lineNum)
        allLines.push({ lineNum, sign: '-', content: diff.removedLines[li], type: 'removed' })
      }
      for (let li = 0; li < diff.addedLines.length; li++) {
        const lineNum = diff.startLine + li
        maxLineNum = Math.max(maxLineNum, lineNum)
        allLines.push({ lineNum, sign: '+', content: diff.addedLines[li], type: 'added' })
      }
    }

    return {
      lines: allLines,
      gutterWidth: Math.max(1, maxLineNum.toString().length),
    }
  }, [diffs])

  if (lines.length === 0) return null

  return (
    <box style={{ flexDirection: 'column' }}>
      {lines.map((line, i) => {
        if (line === 'separator') {
          const pad = ' '.repeat(gutterWidth)
          return (
            <text key={`sep-${i}`} style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
              {pad}  ⋮
            </text>
          )
        }

        const color = line.type === 'added' ? theme.success : theme.error
        const numStr = line.lineNum.toString().padStart(gutterWidth)

        return (
          <text key={`${line.type[0]}-${i}`} attributes={dimmed ? TextAttributes.DIM : undefined}>
            <span fg={theme.muted} attributes={TextAttributes.DIM}>{numStr}</span>
            {'  '}
            <span fg={color}>{line.sign} {line.content}</span>
          </text>
        )
      })}
    </box>
  )
})

export function computeDiffStats(diffs: readonly EditDiff[]) {
  const totalRemoved = diffs.reduce((sum, d) => sum + d.removedLines.length, 0)
  const totalAdded = diffs.reduce((sum, d) => sum + d.addedLines.length, 0)
  return { totalRemoved, totalAdded, changeCount: diffs.length }
}
