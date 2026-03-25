import { useMemo } from 'react'
import { useTheme } from './use-theme'
import type { FilePanelStream } from './use-file-panel'
import { useStreamingReveal } from './use-streaming-reveal'
import { findUniqueMatchRange } from '../utils/diff-utils'
import { computeOptimisticUpdatePreview } from '../utils/file-panel-utils'

export function usePanelStreaming(
  streaming: FilePanelStream | null | undefined,
  content: string | null,
): {
  displayedContent: string
  showCursor: boolean
  highlightCharRanges: Array<{ start: number; end: number; backgroundColor: string }>
  highlightAnchorId: string | undefined
  copyContent: string
  isActivelyStreaming: boolean
} {
  const theme = useTheme()

  const isWriteStream = streaming?.mode === 'write'
  const isReplaceStream = streaming?.mode === 'edit'
  const isActivelyStreaming = streaming?.status === 'receiving'

  const optimisticUpdatePreview = useMemo(
    () => (
      isReplaceStream
        ? computeOptimisticUpdatePreview(
            streaming.baseContent,
            streaming.oldText,
            streaming.newText,
            streaming.replaceAll,
          )
        : null
    ),
    [isReplaceStream, streaming],
  )

  const locatingRange = useMemo(
    () => (
      isReplaceStream && streaming.status === 'receiving' && streaming.streamingTarget !== 'new'
        ? findUniqueMatchRange(streaming.baseContent, streaming.oldText)
        : null
    ),
    [isReplaceStream, streaming],
  )

  const baseDisplayContent = useMemo(() => content ?? '', [content])

  // --- Write stream ---
  const isWriteActive = !!(isWriteStream && (streaming?.status === 'receiving' || streaming?.status === 'applying'))
  const isWriteStreaming = !!(isWriteStream && streaming?.status === 'receiving')
  const writeContent = isWriteActive ? (streaming?.body ?? '') : ''
  const { displayedContent: revealedWrite, showCursor: writeCursor } = useStreamingReveal(writeContent, isWriteStreaming)

  // --- Edit stream: new text ---
  const newStrContent = isReplaceStream ? streaming.newText : ''
  const isNewStreaming = isReplaceStream && isActivelyStreaming && streaming.streamingTarget === 'new'
  const { displayedContent: revealedNew, showCursor: editCursor } = useStreamingReveal(
    newStrContent,
    isNewStreaming,
    undefined,
    newStrContent.length,
  )

  // --- Edit stream: old text highlight ---
  const oldHighlightContent = useMemo(() => {
    if (!isReplaceStream || streaming.status !== 'receiving' || streaming.streamingTarget === 'new') return ''
    const base = streaming.baseContent ?? content ?? ''
    const match = findUniqueMatchRange(base, streaming.oldText)
    if (!match) return ''
    return streaming.oldText
  }, [isReplaceStream, streaming, content])

  const isOldHighlightStreaming = isReplaceStream && isActivelyStreaming && streaming.streamingTarget === 'old' && oldHighlightContent.length > 0

  const { displayedContent: revealedOldHighlight } = useStreamingReveal(
    oldHighlightContent,
    isOldHighlightStreaming,
  )

  // --- Displayed content ---
  const displayedContent = useMemo(() => {
    if (isWriteActive) return revealedWrite
    if (isReplaceStream && (streaming.status === 'receiving' || streaming.status === 'applying')) {
      const base = streaming.baseContent ?? content ?? ''
      if (streaming.replaceAll) {
        return optimisticUpdatePreview?.content ?? base
      }

      const match = findUniqueMatchRange(base, streaming.oldText)
      if (!match) return base
      if (streaming.streamingTarget !== 'new') return base

      return base.slice(0, match.start) + revealedNew + base.slice(match.end)
    }
    return baseDisplayContent
  }, [
    isWriteActive,
    revealedWrite,
    isReplaceStream,
    streaming,
    optimisticUpdatePreview,
    content,
    baseDisplayContent,
    revealedNew,
  ])

  // --- Cursor ---
  const showCursor = isWriteStreaming ? writeCursor : isNewStreaming ? editCursor : false

  // --- Highlight ranges ---
  const previewChangedRanges = useMemo(() => {
    if (!isReplaceStream) return []
    if (streaming.status !== 'receiving' && streaming.status !== 'applying') return []
    if (streaming.replaceAll) return optimisticUpdatePreview?.changedRanges ?? []

    const base = streaming.baseContent ?? content ?? ''
    const match = findUniqueMatchRange(base, streaming.oldText)
    if (!match || streaming.streamingTarget !== 'new') return []
    return [{ start: match.start, end: match.start + revealedNew.length }]
  }, [isReplaceStream, streaming, optimisticUpdatePreview, content, revealedNew])

  const progressiveOldHighlightRanges = useMemo(() => {
    if (!locatingRange || !isReplaceStream || streaming.status !== 'receiving' || streaming.streamingTarget === 'new') return []
    const base = streaming.baseContent ?? content ?? ''
    if (displayedContent !== base) return []
    const revealedLen = revealedOldHighlight.length
    if (revealedLen === 0) return []
    return [{ start: locatingRange.start, end: locatingRange.start + revealedLen }]
  }, [locatingRange, isReplaceStream, streaming, content, displayedContent, revealedOldHighlight])

  const activeHighlightRanges = previewChangedRanges.length > 0 ? previewChangedRanges : progressiveOldHighlightRanges

  // --- Copy content ---
  const copyContent = useMemo(() => {
    if (isWriteStream && streaming.body) return streaming.body
    if (isReplaceStream && optimisticUpdatePreview) return optimisticUpdatePreview.content
    if (displayedContent) return displayedContent
    return isReplaceStream ? streaming.newText : ''
  }, [isWriteStream, isReplaceStream, streaming, optimisticUpdatePreview, displayedContent])

  // --- Highlight char ranges with colors ---
  const highlightCharRanges = useMemo(
    () => activeHighlightRanges.map((range) => ({
      start: range.start,
      end: range.end,
      backgroundColor: previewChangedRanges.length > 0 ? theme.success : theme.error,
    })),
    [activeHighlightRanges, previewChangedRanges.length, theme.success, theme.error],
  )

  const highlightAnchorId = highlightCharRanges.length > 0 ? 'file-highlight-anchor' : undefined

  return {
    displayedContent,
    showCursor,
    highlightCharRanges,
    highlightAnchorId,
    copyContent,
    isActivelyStreaming,
  }
}
