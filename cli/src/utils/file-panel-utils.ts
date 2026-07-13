import type { Span } from '../markdown/blocks'

export interface ChangedRange {
  start: number
  end: number
}

export interface OptimisticUpdatePreview {
  content: string
  changedRanges: ChangedRange[]
}

// ── Local tool-state types for the live file-stream preview ──────────────
// The file-panel previews in-progress file write/edit tool state to render
// an optimistic diff. These are the minimal shape the preview needs; the
// full typed tool-state schemas live in the agent package (server-side) and
// are not imported by clients. `useFilePanel` is currently always called
// with `toolState: null`, so these types exist only to keep the preview
// path type-checked. If the live-stream feature is re-enabled, the state
// should arrive via the presentation layer, not raw tool handles.

export type ToolPhase = 'idle' | 'streaming' | 'executing' | 'completed' | 'error' | 'rejected' | 'interrupted'

export interface FileWriteState {
  phase: ToolPhase
  path?: string
  body: string
  charCount: number
  lineCount: number
}

export interface FileEditState {
  phase: ToolPhase
  path?: string
  oldText: string
  newText: string
  replaceAll: boolean
  streamingTarget?: 'old' | 'new'
  diffs: unknown[]
}

export interface ToolHandle {
  toolKey: string
  state?: FileEditState | FileWriteState | { phase: ToolPhase }
}

export interface TurnState {
  handles?: {
    handles?: Map<string, ToolHandle>
  }
}

export function computeOptimisticUpdatePreview(
  baseContent: string | null | undefined,
  oldString: string | undefined,
  newString: string | undefined,
  replaceAll: boolean | undefined,
): OptimisticUpdatePreview | null {
  if (!baseContent || !oldString || newString === undefined) return null
  if (!baseContent.includes(oldString)) return null

  if (!replaceAll) {
    const index = baseContent.indexOf(oldString)
    if (index === -1) return null
    return {
      content: baseContent.slice(0, index) + newString + baseContent.slice(index + oldString.length),
      changedRanges: [{ start: index, end: index + newString.length }],
    }
  }

  const changedRanges: ChangedRange[] = []
  let cursor = 0
  let result = ''

  while (cursor < baseContent.length) {
    const index = baseContent.indexOf(oldString, cursor)
    if (index === -1) {
      result += baseContent.slice(cursor)
      break
    }
    result += baseContent.slice(cursor, index)
    const start = result.length
    result += newString
    changedRanges.push({ start, end: start + newString.length })
    cursor = index + oldString.length
  }

  return { content: result, changedRanges }
}

export function highlightCodeLines(
  lines: Span[][],
  highlightRanges: Array<{ start: number; end: number; backgroundColor: string }>,
): Span[][] {
  if (highlightRanges.length === 0) return lines

  let contentOffset = 0

  return lines.map((line) => {
    const lineLength = line.reduce((sum, span) => sum + span.text.length, 0)
    const lineStart = contentOffset
    const lineEnd = lineStart + lineLength
    contentOffset = lineEnd + 1

    const lineRanges = highlightRanges.filter((range) => range.start < lineEnd && range.end > lineStart)
    if (lineRanges.length === 0) return line

    const nextLine: Span[] = []
    let spanOffset = lineStart

    for (const span of line) {
      const originalSpan = { ...span }
      const spanStart = spanOffset
      const spanEnd = spanStart + originalSpan.text.length
      spanOffset = spanEnd

      let segments: Array<{ text: string; bg?: string }> = [{ text: originalSpan.text, bg: originalSpan.bg }]
      for (const range of lineRanges) {
        const updated: Array<{ text: string; bg?: string }> = []
        let segmentOffset = spanStart

        for (const segment of segments) {
          const segmentStart = segmentOffset
          const segmentEnd = segmentStart + segment.text.length
          segmentOffset = segmentEnd

          const overlapStart = Math.max(segmentStart, range.start)
          const overlapEnd = Math.min(segmentEnd, range.end)

          if (overlapStart >= overlapEnd) {
            updated.push(segment)
            continue
          }

          const localStart = overlapStart - segmentStart
          const localEnd = overlapEnd - segmentStart

          if (localStart > 0) {
            updated.push({ text: segment.text.slice(0, localStart), bg: segment.bg })
          }

          updated.push({
            text: segment.text.slice(localStart, localEnd),
            bg: range.backgroundColor,
          })

          if (localEnd < segment.text.length) {
            updated.push({ text: segment.text.slice(localEnd), bg: segment.bg })
          }
        }

        segments = updated
      }

      nextLine.push(
        ...segments.map((segment) => ({
          ...originalSpan,
          text: segment.text,
          bg: segment.bg,
        })),
      )
    }

    return nextLine
  })
}

export function findActiveFileStream(
  toolHandles: Record<string, ToolHandle> | undefined,
  targetPath: string,
): { toolCallId: string; state: FileEditState | FileWriteState } | null {
  if (!toolHandles) return null

  let result: { toolCallId: string; state: FileEditState | FileWriteState } | null = null
  for (const [callId, handle] of Object.entries(toolHandles)) {
    const state = getActiveFileState(handle, targetPath)
    if (state) result = { toolCallId: callId, state }
  }
  return result
}

function getActiveFileState(handle: ToolHandle, targetPath: string): FileEditState | FileWriteState | null {
  if (!handle.state) return null
  if (handle.state.phase !== 'streaming' && handle.state.phase !== 'executing') return null

  if (handle.toolKey === 'fileEdit' && isFileEditState(handle.state) && handle.state.path === targetPath) {
    return handle.state
  }

  if (handle.toolKey === 'fileWrite' && isFileWriteState(handle.state) && handle.state.path === targetPath) {
    return handle.state
  }

  return null
}

function isFileEditState(state: ToolHandle['state']): state is FileEditState {
  return !!state
    && 'oldText' in state
    && typeof (state as FileEditState).oldText === 'string'
    && 'newText' in state
    && typeof (state as FileEditState).newText === 'string'
    && 'replaceAll' in state
    && typeof (state as FileEditState).replaceAll === 'boolean'
    && 'diffs' in state
    && Array.isArray((state as FileEditState).diffs)
}

function isFileWriteState(state: ToolHandle['state']): state is FileWriteState {
  return !!state
    && 'body' in state
    && typeof (state as FileWriteState).body === 'string'
    && 'charCount' in state
    && typeof (state as FileWriteState).charCount === 'number'
    && 'lineCount' in state
    && typeof (state as FileWriteState).lineCount === 'number'
}
