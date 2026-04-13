import type { FileEditState, FileWriteState, ToolState } from '@magnitudedev/agent'
import type { Span } from '../markdown/blocks'

export interface ChangedRange {
  start: number
  end: number
}

export interface OptimisticUpdatePreview {
  content: string
  changedRanges: ChangedRange[]
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
  toolHandles: Record<string, { state: ToolState }> | undefined,
  targetPath: string,
): { toolCallId: string; state: FileEditState | FileWriteState } | null {
  if (!toolHandles) return null

  let result: { toolCallId: string; state: FileEditState | FileWriteState } | null = null
  for (const [callId, handle] of Object.entries(toolHandles)) {
    if (!handle.state) continue
    const s = handle.state
    if (s.toolKey !== 'fileEdit' && s.toolKey !== 'fileWrite') continue
    if (!s.path || s.path !== targetPath) continue
    if (s.phase !== 'streaming' && s.phase !== 'executing') continue
    result = { toolCallId: callId, state: s }
  }
  return result
}
