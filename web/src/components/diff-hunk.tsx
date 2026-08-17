/**
 * DiffHunk — spec Appendix (Diff hunks)
 *
 * Renders a unified diff hunk with added/removed/context lines.
 * Compact unified-diff rows with palette-backed added/removed tones and line
 * numbers aligned in a fixed-width gutter.
 * Streaming cursor: ▍ on last added line when streaming.
 */
import type { ReactNode } from "react"
export interface DiffHunkProps {
  contextBefore?: readonly string[]
  removedLines: readonly string[]
  addedLines: readonly string[]
  contextAfter?: readonly string[]
  /** Show streaming cursor on the last added line */
  streamingCursor?: boolean
  /** Starting line number for the hunk (1-based) */
  startLine?: number
}
interface DiffRow {
  lineNum: number
  prefix: string
  text: string
  kind: "context" | "added" | "removed"
  isStreamingLast?: boolean
}
const STREAMING_CURSOR = "\u258D" // ▍

export function DiffHunk({
  contextBefore = [],
  removedLines,
  addedLines,
  contextAfter = [],
  streamingCursor = false,
  startLine = 1,
}: DiffHunkProps): ReactNode {
  // Build rows with line numbers — unified diff style:
  // contextBefore → line numbers increment
  // removedLines → line numbers continue from contextBefore
  // addedLines → line numbers restart from the first removed line number
  // contextAfter → line numbers continue after addedLines
  const contextRadius = contextBefore.length
  const lineStart = startLine - contextRadius
  const rows: DiffRow[] = []
  let lineNum = lineStart

  // Context before
  for (const line of contextBefore) {
    rows.push({
      lineNum,
      prefix: " ",
      text: line,
      kind: "context",
    })
    lineNum++
  }

  // Removed lines — line numbers continue
  const addedStartLine = lineNum
  for (const line of removedLines) {
    rows.push({
      lineNum,
      prefix: "-",
      text: line,
      kind: "removed",
    })
    lineNum++
  }

  // Added lines — line numbers restart from where removed started
  lineNum = addedStartLine
  for (let i = 0; i < addedLines.length; i++) {
    const line = addedLines[i]
    const isLast = i === addedLines.length - 1
    rows.push({
      lineNum,
      prefix: "+",
      text: line,
      kind: "added",
      isStreamingLast: streamingCursor && isLast,
    })
    lineNum++
  }

  // Context after — line numbers continue after added lines
  for (const line of contextAfter) {
    rows.push({
      lineNum,
      prefix: " ",
      text: line,
      kind: "context",
    })
    lineNum++
  }
  return (
    <div className="mt-1 overflow-hidden rounded border border-slate-300 font-mono text-[13px] leading-[1.5] dark:border-slate-750">
      {rows.map((row, index) => (
        <DiffRowView key={`row-${index}`} row={row} />
      ))}
    </div>
  )
}
function DiffRowView({ row }: { row: DiffRow }): ReactNode {
  let rowClass = ""
  let contentClass = ""
  let prefixClass = ""
  switch (row.kind) {
    case "added":
      rowClass = "bg-green-200/35 dark:bg-green-800/25"
      contentClass = prefixClass = "text-green-700 dark:text-green-400"
      break
    case "removed":
      rowClass = "bg-red-200/35 dark:bg-red-800/25"
      contentClass = prefixClass = "text-red-700 dark:text-red-400"
      break
    case "context":
      contentClass = "text-slate-600 dark:text-slate-400"
      prefixClass = "text-transparent"
      break
  }
  return (
    <div className={`${rowClass} flex items-start`}>
      <span className="w-12 shrink-0 pr-2 text-right text-[11px] whitespace-nowrap text-slate-500 select-none">
        {row.lineNum}
      </span>
      <span className={`${prefixClass} w-4 shrink-0 text-center select-none`}>
        {row.prefix}
      </span>
      <span
        className={`${contentClass} min-w-0 flex-1 pr-2 whitespace-pre-wrap break-words`}
      >
        {row.text}
        {row.isStreamingLast && (
          <span className="animate-blink text-blue-700 dark:text-blue-500">
            {STREAMING_CURSOR}
          </span>
        )}
      </span>
    </div>
  )
}
