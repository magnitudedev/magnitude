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
import { slugify } from '../markdown/blocks'



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

interface ArtifactReaderPanelProps {
  artifactName: string
  content: string | null
  scrollToSection?: string
  onClose: () => void
  onOpenArtifact?: (name: string, section?: string) => void
  streaming?: {
    active: boolean
    toolKey: 'artifactWrite' | 'artifactUpdate'
    phase: 'streaming' | 'executing' | 'error' | 'success' | 'rejected' | 'interrupted'
    contentSoFar?: string
    oldStringSoFar?: string
    newStringSoFar?: string
    replaceAll?: boolean
    baseContent?: string | null
  }
}

function HeaderBadge({ label, color }: { label: string; color: string }) {
  return (
    <text style={{ fg: color }} attributes={TextAttributes.BOLD}>
      [{label}]
    </text>
  )
}

export const ArtifactReaderPanel = memo(function ArtifactReaderPanel({
  artifactName, content, scrollToSection, onClose, onOpenArtifact, streaming
}: ArtifactReaderPanelProps) {
  const theme = useTheme()
  const { width: terminalWidth } = useTerminalDimensions()
  const scrollboxRef = useRef<any>(null)
  const mountedRef = useMountedRef()
  const safeTimeout = useSafeTimeout()

  const isWriteStream = streaming?.toolKey === 'artifactWrite'
  const isUpdateStream = streaming?.toolKey === 'artifactUpdate'
  const isStreamingPhase = streaming?.phase === 'streaming'
  const failedStream = streaming?.phase === 'error' || streaming?.phase === 'interrupted' || streaming?.phase === 'rejected'
  const optimisticUpdatePreview = useMemo(
    () => computeOptimisticUpdatePreview(
      streaming?.baseContent,
      streaming?.oldStringSoFar,
      streaming?.newStringSoFar,
      streaming?.replaceAll,
    ),
    [streaming],
  )
  const locatingRange = useMemo(
    () => (
      isUpdateStream && streaming?.phase === 'streaming' && !streaming?.newStringSoFar
        ? findUniqueMatchRange(streaming.baseContent, streaming.oldStringSoFar)
        : null
    ),
    [isUpdateStream, streaming],
  )

  const baseDisplayContent = useMemo(() => {
    if (failedStream && isWriteStream && content) return content
    if (failedStream && isWriteStream) return streaming?.contentSoFar ?? ''
    if ((failedStream || streaming?.phase === 'success') && isUpdateStream) {
      return content ?? streaming?.baseContent ?? ''
    }
    return content ?? ''
  }, [streaming, isWriteStream, isUpdateStream, content, failedStream])

  const isActivelyStreaming = !!streaming && streaming.phase === 'streaming'

  const writeContent = isWriteStream && (streaming?.phase === 'streaming' || streaming?.phase === 'executing')
    ? (streaming?.contentSoFar ?? '')
    : ''
  const isWriteStreaming = isWriteStream && isActivelyStreaming
  const { displayedContent: revealedWrite, showCursor: writeCursor } = useStreamingReveal(writeContent, isWriteStreaming)

  const newStrContent = streaming?.newStringSoFar ?? ''
  const isNewStreaming = isUpdateStream && isActivelyStreaming && !!streaming?.newStringSoFar
  const { displayedContent: revealedNew, showCursor: editCursor } = useStreamingReveal(
    newStrContent,
    isNewStreaming,
    undefined,
    newStrContent.length,
  )

  const oldHighlightContent = useMemo(() => {
    if (!isUpdateStream || streaming?.phase !== 'streaming' || streaming?.newStringSoFar) return ''
    const base = streaming?.baseContent ?? content ?? ''
    const match = findUniqueMatchRange(base, streaming?.oldStringSoFar)
    if (!match) return ''
    return streaming?.oldStringSoFar ?? ''
  }, [isUpdateStream, streaming, content])

  const isOldHighlightStreaming = isUpdateStream && isActivelyStreaming && !streaming?.newStringSoFar && oldHighlightContent.length > 0

  const { displayedContent: revealedOldHighlight } = useStreamingReveal(
    oldHighlightContent,
    isOldHighlightStreaming,
  )

  const displayedContent = useMemo(() => {
    if (isWriteStreaming) return revealedWrite
    if (isUpdateStream && (streaming?.phase === 'streaming' || streaming?.phase === 'executing')) {
      const base = streaming?.baseContent ?? content ?? ''
      if (streaming?.replaceAll) {
        return optimisticUpdatePreview?.content ?? base
      }

      const match = findUniqueMatchRange(base, streaming?.oldStringSoFar)
      if (!match) return base
      if (!streaming?.newStringSoFar) return base

      return base.slice(0, match.start) + revealedNew + base.slice(match.end)
    }
    return baseDisplayContent
  }, [
    isWriteStreaming,
    revealedWrite,
    isUpdateStream,
    streaming,
    optimisticUpdatePreview,
    content,
    baseDisplayContent,
    revealedNew,
  ])

  const showCursor = isWriteStreaming ? writeCursor : isNewStreaming ? editCursor : false

  const previewChangedRanges = useMemo(() => {
    if (!isUpdateStream) return []
    if (streaming?.phase !== 'streaming' && streaming?.phase !== 'executing') return []
    if (streaming?.replaceAll) return optimisticUpdatePreview?.changedRanges ?? []

    const base = streaming?.baseContent ?? content ?? ''
    const match = findUniqueMatchRange(base, streaming?.oldStringSoFar)
    if (!match || !streaming?.newStringSoFar) return []
    return [{ start: match.start, end: match.start + revealedNew.length }]
  }, [isUpdateStream, streaming?.phase, streaming?.replaceAll, optimisticUpdatePreview, streaming?.baseContent, content, streaming?.oldStringSoFar, streaming?.newStringSoFar, revealedNew])

  const progressiveOldHighlightRanges = useMemo(() => {
    if (!locatingRange || !isUpdateStream || streaming?.phase !== 'streaming' || streaming?.newStringSoFar) return []
    const base = streaming?.baseContent ?? content ?? ''
    if (displayedContent !== base) return []
    const revealedLen = revealedOldHighlight.length
    if (revealedLen === 0) return []
    return [{ start: locatingRange.start, end: locatingRange.start + revealedLen }]
  }, [locatingRange, isUpdateStream, streaming, content, displayedContent, revealedOldHighlight])
  const activeHighlightRanges = previewChangedRanges.length > 0 ? previewChangedRanges : progressiveOldHighlightRanges

  const copyContent = useMemo(() => {
    if (isWriteStream && streaming?.contentSoFar) return streaming.contentSoFar
    if (isUpdateStream && optimisticUpdatePreview) return optimisticUpdatePreview.content
    if (displayedContent) return displayedContent
    return streaming?.newStringSoFar ?? ''
  }, [isWriteStream, isUpdateStream, streaming, optimisticUpdatePreview, displayedContent])



  const highlightCharRanges = useMemo(
    () => activeHighlightRanges.map((range) => ({
      start: range.start,
      end: range.end,
      backgroundColor: previewChangedRanges.length > 0 ? theme.success : theme.error,
    })),
    [activeHighlightRanges, previewChangedRanges.length, theme.success, theme.error],
  )

  const highlightAnchorId = highlightCharRanges.length > 0 ? 'artifact-highlight-anchor' : undefined
  const codeBlockWidth = Math.max(20, terminalWidth - 10)
  const targetSectionScrollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const highlightAnchorScrollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const targetSectionId = useMemo(
    () => scrollToSection ? `section-${slugify(scrollToSection)}` : null,
    [scrollToSection],
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

  const headerLabel = scrollToSection
    ? `≡  ${artifactName} > ${scrollToSection}`
    : `≡  ${artifactName}`

  const headerBadge = useMemo(() => {
    if (!streaming) return null
    if (streaming.phase === 'executing') {
      return <HeaderBadge label="Saving…" color={theme.info} />
    }
    if (streaming.phase === 'streaming') {
      if (isUpdateStream && !streaming?.newStringSoFar) {
        return <HeaderBadge label={locatingRange ? 'Preparing update…' : 'Locating target…'} color={theme.warning} />
      }
      return <HeaderBadge label={isWriteStream ? 'Writing…' : 'Updating…'} color={theme.info} />
    }
    if (streaming.phase === 'error' || streaming.phase === 'rejected' || streaming.phase === 'interrupted') {
      return <HeaderBadge label="Error" color={theme.error} />
    }
    return null
  }, [streaming, theme, isWriteStream, isUpdateStream, locatingRange])

  const updateSummary = isUpdateStream
    ? (
        optimisticUpdatePreview
          ? `${streaming?.replaceAll ? 'Updating all matches…' : 'Updating 1 region…'}${showCursor ? ' ▍' : ''}`
          : streaming?.newStringSoFar
            ? `new: ${streaming.newStringSoFar}${isStreamingPhase ? '▍' : ''}`
            : null
      )
    : null

  const showNotSavedWarning = !!streaming && failedStream && (
    (streaming.baseContent ?? null) !== content || !!streaming.contentSoFar || !!streaming.newStringSoFar
  )

  const failedDraftForExistingArtifact = failedStream && isWriteStream && !!content && !!streaming?.contentSoFar

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
        <StreamingMarkdownContent
            content={displayedContent}
            onOpenArtifact={onOpenArtifact}
            showCursor={showCursor}
            highlightRanges={highlightCharRanges}
            highlightAnchorId={highlightAnchorId}
            streaming={isActivelyStreaming}
            codeBlockWidth={codeBlockWidth}
          />
      </scrollbox>
    </box>
  )
})