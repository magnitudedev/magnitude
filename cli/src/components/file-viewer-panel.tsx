import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTerminalDimensions } from '@opentui/react'
import { useMountedRef } from '../hooks/use-mounted-ref'
import { useSafeTimeout } from '../hooks/use-safe-timeout'
import { useStreamingReveal } from '../hooks/use-streaming-reveal'
import { useTheme } from '../hooks/use-theme'
import { safeRenderableAccess, safeRenderableCall } from '../utils/safe-renderable-access'
import { BOX_CHARS } from '../utils/ui-constants'
import { StreamingMarkdownContent } from '../markdown/markdown-content'
import { Button } from './button'
import { slugify, type Span } from '../markdown/blocks'
import { highlightFile } from '../markdown/highlight-file'
import type { FilePanelStream } from '../hooks/use-file-panel'

interface ChangedRange {
  start: number
  end: number
}

interface OptimisticUpdatePreview {
  content: string
  changedRanges: ChangedRange[]
}

function findUniqueMatchRange(content: string | null | undefined, needle: string | undefined): ChangedRange | null {
  if (!content || !needle) return null
  const first = content.indexOf(needle)
  if (first === -1) return null
  const second = content.indexOf(needle, first + 1)
  if (second !== -1) return null
  return { start: first, end: first + needle.length }
}

function computeOptimisticUpdatePreview(
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

function CopyButton({ content, theme }: { content: string; theme: any }) {
  const [copied, setCopied] = useState(false)
  const [hovered, setHovered] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const safeTimeout = useSafeTimeout()

  const handleCopy = useCallback(() => {
    const proc = require('child_process').spawn('pbcopy', [], { stdio: ['pipe', 'ignore', 'ignore'] })
    proc.stdin.write(content)
    proc.stdin.end()

    setCopied(true)
    safeTimeout.clear(timerRef.current)
    timerRef.current = safeTimeout.set(() => setCopied(false), 2000)
  }, [content, safeTimeout])

  const color = copied ? theme.success : hovered ? theme.foreground : theme.muted

  return (
    <Button
      onClick={handleCopy}
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
    >
      <text style={{ fg: color }}>
        {copied ? '[✓]' : '[Copy]'}
      </text>
    </Button>
  )
}

function CloseButton({ theme, onClose }: { theme: any; onClose: () => void }) {
  const [hovered, setHovered] = useState(false)
  const color = hovered ? theme.foreground : theme.muted

  return (
    <Button
      onClick={onClose}
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
    >
      <text style={{ fg: color }}>[✕]</text>
    </Button>
  )
}

interface FileViewerPanelProps {
  filePath: string
  content: string | null
  scrollToSection?: string
  onClose: () => void
  onOpenFile?: (path: string, section?: string) => void
  streaming?: FilePanelStream | null
}





function isMarkdownFile(filePath: string): boolean {
  return filePath.toLowerCase().endsWith('.md')
}

function renderCodeLines(lines: Span[], idx: number, fallbackFg: string) {
  return (
    <text key={idx} style={{ fg: fallbackFg }}>
      {lines.map((span, i) => (
        <span key={i} fg={span.fg ?? fallbackFg}>{span.text}</span>
      ))}
    </text>
  )
}

export const FileViewerPanel = memo(function FileViewerPanel({
  filePath, content, scrollToSection, onClose, onOpenFile, streaming
}: FileViewerPanelProps) {
  const theme = useTheme()
  const { width: terminalWidth } = useTerminalDimensions()
  const scrollboxRef = useRef<any>(null)
  const mountedRef = useMountedRef()
  const safeTimeout = useSafeTimeout()
  const markdown = isMarkdownFile(filePath)

  const isWriteStream = streaming?.mode === 'write'
  const isReplaceStream = streaming?.mode === 'replace'
  const optimisticUpdatePreview = useMemo(
    () => (
      isReplaceStream
        ? computeOptimisticUpdatePreview(
            streaming.baseContent,
            streaming.oldStringSoFar,
            streaming.newStringSoFar,
            streaming.replaceAll,
          )
        : null
    ),
    [isReplaceStream, streaming],
  )
  const locatingRange = useMemo(
    () => (
      isReplaceStream && streaming.status === 'receiving' && !streaming.newStringSoFar
        ? findUniqueMatchRange(streaming.baseContent, streaming.oldStringSoFar)
        : null
    ),
    [isReplaceStream, streaming],
  )

  const baseDisplayContent = useMemo(() => content ?? '', [content])

  const isActivelyStreaming = streaming?.status === 'receiving'

  const writeContent = isWriteStream && (streaming?.status === 'receiving' || streaming?.status === 'applying')
    ? (streaming?.contentSoFar ?? '')
    : ''
  const isWriteStreaming = !!(isWriteStream && streaming?.status === 'receiving')
  const { displayedContent: revealedWrite, showCursor: writeCursor } = useStreamingReveal(writeContent, isWriteStreaming)

  const newStrContent = isReplaceStream ? streaming.newStringSoFar : ''
  const isNewStreaming = isReplaceStream && isActivelyStreaming && !!streaming.newStringSoFar
  const { displayedContent: revealedNew, showCursor: editCursor } = useStreamingReveal(
    newStrContent,
    isNewStreaming,
    undefined,
    newStrContent.length,
  )

  const oldHighlightContent = useMemo(() => {
    if (!isReplaceStream || streaming.status !== 'receiving' || streaming.newStringSoFar) return ''
    const base = streaming.baseContent ?? content ?? ''
    const match = findUniqueMatchRange(base, streaming.oldStringSoFar)
    if (!match) return ''
    return streaming.oldStringSoFar
  }, [isReplaceStream, streaming, content])

  const isOldHighlightStreaming = isReplaceStream && isActivelyStreaming && !streaming.newStringSoFar && oldHighlightContent.length > 0

  const { displayedContent: revealedOldHighlight } = useStreamingReveal(
    oldHighlightContent,
    isOldHighlightStreaming,
  )

  const displayedContent = useMemo(() => {
    if (isWriteStreaming) return revealedWrite
    if (isReplaceStream && (streaming.status === 'receiving' || streaming.status === 'applying')) {
      const base = streaming.baseContent ?? content ?? ''
      if (streaming.replaceAll) {
        return optimisticUpdatePreview?.content ?? base
      }

      const match = findUniqueMatchRange(base, streaming.oldStringSoFar)
      if (!match) return base
      if (!streaming.newStringSoFar) return base

      return base.slice(0, match.start) + revealedNew + base.slice(match.end)
    }
    return baseDisplayContent
  }, [
    isWriteStreaming,
    revealedWrite,
    isReplaceStream,
    streaming,
    optimisticUpdatePreview,
    content,
    baseDisplayContent,
    revealedNew,
  ])

  const showCursor = isWriteStreaming ? writeCursor : isNewStreaming ? editCursor : false

  const previewChangedRanges = useMemo(() => {
    if (!isReplaceStream) return []
    if (streaming.status !== 'receiving' && streaming.status !== 'applying') return []
    if (streaming.replaceAll) return optimisticUpdatePreview?.changedRanges ?? []

    const base = streaming.baseContent ?? content ?? ''
    const match = findUniqueMatchRange(base, streaming.oldStringSoFar)
    if (!match || !streaming.newStringSoFar) return []
    return [{ start: match.start, end: match.start + revealedNew.length }]
  }, [isReplaceStream, streaming, optimisticUpdatePreview, content, revealedNew])

  const progressiveOldHighlightRanges = useMemo(() => {
    if (!locatingRange || !isReplaceStream || streaming.status !== 'receiving' || streaming.newStringSoFar) return []
    const base = streaming.baseContent ?? content ?? ''
    if (displayedContent !== base) return []
    const revealedLen = revealedOldHighlight.length
    if (revealedLen === 0) return []
    return [{ start: locatingRange.start, end: locatingRange.start + revealedLen }]
  }, [locatingRange, isReplaceStream, streaming, content, displayedContent, revealedOldHighlight])
  const activeHighlightRanges = previewChangedRanges.length > 0 ? previewChangedRanges : progressiveOldHighlightRanges

  const copyContent = useMemo(() => {
    if (isWriteStream && streaming.contentSoFar) return streaming.contentSoFar
    if (isReplaceStream && optimisticUpdatePreview) return optimisticUpdatePreview.content
    if (displayedContent) return displayedContent
    return isReplaceStream ? streaming.newStringSoFar : ''
  }, [isWriteStream, isReplaceStream, streaming, optimisticUpdatePreview, displayedContent])

  const highlightCharRanges = useMemo(
    () => activeHighlightRanges.map((range) => ({
      start: range.start,
      end: range.end,
      backgroundColor: previewChangedRanges.length > 0 ? theme.success : theme.error,
    })),
    [activeHighlightRanges, previewChangedRanges.length, theme.success, theme.error],
  )

  const highlightAnchorId = highlightCharRanges.length > 0 ? 'file-highlight-anchor' : undefined
  const codeBlockWidth = Math.max(20, terminalWidth - 10)
  const targetSectionScrollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const highlightAnchorScrollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const targetSectionId = useMemo(
    () => markdown && scrollToSection ? `section-${slugify(scrollToSection)}` : null,
    [markdown, scrollToSection],
  )

  useEffect(() => {
    safeTimeout.clear(targetSectionScrollTimeoutRef.current)
    if (!targetSectionId) return

    safeRenderableCall(
      scrollboxRef.current,
      (scrollbox) => {
        scrollbox.stickyScroll = false
      },
      { mountedRef },
    )

    const doScroll = () => {
      const offsetY = safeRenderableAccess(
        scrollboxRef.current,
        (scrollbox) => {
          const contentNode = scrollbox.content
          if (!contentNode) return null

          const targetEl = contentNode.findDescendantById(targetSectionId)
          if (!targetEl) return null

          let offsetY = 0
          let node: any = targetEl
          while (node && node !== contentNode) {
            const yogaNode = node.yogaNode || node.getLayoutNode?.()
            if (yogaNode) {
              offsetY += yogaNode.getComputedTop()
            }
            node = node.parent
          }

          return offsetY
        },
        {
          mountedRef,
          fallback: null,
        },
      )
      if (offsetY == null) return

      safeRenderableCall(
        scrollboxRef.current,
        (sb) => sb.scrollTo(offsetY),
        { mountedRef },
      )
    }

    targetSectionScrollTimeoutRef.current = safeTimeout.set(doScroll, 50)

    return () => {
      safeTimeout.clear(targetSectionScrollTimeoutRef.current)
      targetSectionScrollTimeoutRef.current = null
    }
  }, [targetSectionId, mountedRef, safeTimeout])

  useEffect(() => {
    safeTimeout.clear(highlightAnchorScrollTimeoutRef.current)
    if (!highlightAnchorId) return

    highlightAnchorScrollTimeoutRef.current = safeTimeout.set(() => {
      const offsetY = safeRenderableAccess(
        scrollboxRef.current,
        (scrollbox) => {
          const contentNode = scrollbox.content
          if (!contentNode) return null

          const targetEl = contentNode.findDescendantById(highlightAnchorId)
          if (!targetEl) return null

          let offsetY = 0
          let node: any = targetEl
          while (node && node !== contentNode) {
            const yogaNode = node.yogaNode || node.getLayoutNode?.()
            if (yogaNode) offsetY += yogaNode.getComputedTop()
            node = node.parent
          }

          return offsetY
        },
        {
          mountedRef,
          fallback: null,
        },
      )
      if (offsetY == null) return

      safeRenderableCall(
        scrollboxRef.current,
        (sb) => sb.scrollTo(offsetY),
        { mountedRef },
      )
    }, 50)

    return () => {
      safeTimeout.clear(highlightAnchorScrollTimeoutRef.current)
      highlightAnchorScrollTimeoutRef.current = null
    }
  }, [highlightAnchorId, displayedContent, mountedRef, safeTimeout])

  const headerLabel = scrollToSection && markdown
    ? `≡  ${filePath} > ${scrollToSection}`
    : `≡  ${filePath}`



  const codeLines = useMemo(() => highlightFile(displayedContent, filePath), [displayedContent, filePath])

  return (
    <box style={{
      flexDirection: 'column',
      height: '100%',
      borderStyle: 'single',
      borderColor: theme.border || theme.muted,
      customBorderChars: BOX_CHARS,
      paddingLeft: 1,
      paddingRight: 1,
    }}>
      <box style={{ flexDirection: 'row', justifyContent: 'space-between', flexShrink: 0 }}>
        <box style={{ flexDirection: 'row', gap: 1 }}>
          <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>
            {headerLabel}
          </text>

        </box>
        <box style={{ flexDirection: 'row', gap: 2 }}>
          <CopyButton content={copyContent} theme={theme} />
          <CloseButton theme={theme} onClose={onClose} />
        </box>
      </box>

      <scrollbox
        ref={scrollboxRef}
        stickyScroll={!scrollToSection}
        stickyStart="top"
        scrollX={false}
        scrollbarOptions={{ visible: false }}
        verticalScrollbarOptions={{ visible: false }}
        style={{
          flexGrow: 1,
          rootOptions: {
            flexGrow: 1,
            backgroundColor: 'transparent',
          },
          wrapperOptions: {
            border: false,
            backgroundColor: 'transparent',
          },
          contentOptions: {
            paddingTop: 1,
          },
        }}
      >
        {markdown ? (
          <StreamingMarkdownContent
            content={displayedContent}
            onOpenFile={onOpenFile}
            showCursor={showCursor}
            highlightRanges={highlightCharRanges}
            highlightAnchorId={highlightAnchorId}
            streaming={isActivelyStreaming}
            codeBlockWidth={codeBlockWidth}
          />
        ) : (
          <box style={{ flexDirection: 'column' }}>
            {codeLines.map((line, idx) => renderCodeLines(line, idx, theme.foreground))}
            {showCursor && <text style={{ fg: theme.foreground }}>▍</text>}
          </box>
        )}
      </scrollbox>
    </box>
  )
})
