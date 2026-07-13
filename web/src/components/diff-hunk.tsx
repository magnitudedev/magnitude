/**
 * DiffHunk — spec Appendix (Diff hunks)
 *
 * Renders a unified diff hunk with added/removed/context lines.
 * Container: border 1px var(--border-default), border-radius 4px, overflow hidden.
 * Added lines: background var(--diff-added-bg), color var(--diff-added-fg), prefix +.
 * Removed lines: background var(--diff-removed-bg), color var(--diff-removed-fg), prefix -.
 * Context lines: no background, var(--fg-secondary).
 * Line numbers: 48px column, var(--fg-tertiary) text-xs, right-aligned.
 * Streaming cursor: ▍ on last added line when streaming.
 */
import type { CSSProperties, ReactNode } from "react"

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
    rows.push({ lineNum, prefix: " ", text: line, kind: "context" })
    lineNum++
  }

  // Removed lines — line numbers continue
  const addedStartLine = lineNum
  for (const line of removedLines) {
    rows.push({ lineNum, prefix: "-", text: line, kind: "removed" })
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
    rows.push({ lineNum, prefix: " ", text: line, kind: "context" })
    lineNum++
  }

  const containerStyle: CSSProperties = {
    border: "1px solid var(--border-default)",
    borderRadius: 4,
    overflow: "hidden",
    fontFamily: "var(--font-mono)",
    fontSize: 13,
    lineHeight: 1.5,
    marginTop: 4,
  }

  return (
    <div style={containerStyle}>
      {rows.map((row, index) => (
        <DiffRowView key={`row-${index}`} row={row} />
      ))}
    </div>
  )
}

function DiffRowView({ row }: { row: DiffRow }): ReactNode {
  const lineNumStyle: CSSProperties = {
    width: 48,
    flexShrink: 0,
    textAlign: "right",
    paddingRight: 8,
    color: "var(--fg-tertiary)",
    fontSize: 11,
    userSelect: "none",
    whiteSpace: "nowrap",
  }

  let rowStyle: CSSProperties = {
    display: "flex",
    alignItems: "flex-start",
  }

  let contentStyle: CSSProperties = {
    flex: 1,
    minWidth: 0,
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
    paddingRight: 8,
  }

  let prefixStyle: CSSProperties = {
    flexShrink: 0,
    width: 16,
    textAlign: "center",
    userSelect: "none",
  }

  switch (row.kind) {
    case "added":
      rowStyle = { ...rowStyle, background: "var(--diff-added-bg)" }
      contentStyle = { ...contentStyle, color: "var(--diff-added-fg)" }
      prefixStyle = { ...prefixStyle, color: "var(--diff-added-fg)" }
      break
    case "removed":
      rowStyle = { ...rowStyle, background: "var(--diff-removed-bg)" }
      contentStyle = { ...contentStyle, color: "var(--diff-removed-fg)" }
      prefixStyle = { ...prefixStyle, color: "var(--diff-removed-fg)" }
      break
    case "context":
      contentStyle = { ...contentStyle, color: "var(--fg-secondary)" }
      prefixStyle = { ...prefixStyle, color: "transparent" }
      break
  }

  return (
    <div style={rowStyle}>
      <span style={lineNumStyle}>{row.lineNum}</span>
      <span style={prefixStyle}>{row.prefix}</span>
      <span style={contentStyle}>
        {row.text}
        {row.isStreamingLast && (
          <span className="animate-blink" style={{ color: "var(--accent-primary)" }}>
            {STREAMING_CURSOR}
          </span>
        )}
      </span>
    </div>
  )
}
