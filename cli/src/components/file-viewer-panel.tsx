import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTerminalDimensions } from '@opentui/react'
import { useMountedRef } from '../hooks/use-mounted-ref'
import { useSafeTimeout } from '../hooks/use-safe-timeout'
import { useTheme } from '../hooks/use-theme'
import { safeRenderableAccess, safeRenderableCall } from '../utils/safe-renderable-access'
import { BOX_CHARS } from '../utils/ui-constants'
import { MarkdownContent } from '../markdown/markdown-content'
import { Button } from './button'
import { slugify } from '../markdown/blocks'
import { highlightFile } from '../markdown/highlight-file'
import { isMarkdownFile, renderCodeLines } from '../utils/file-lang'

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
}


export const FileViewerPanel = memo(function FileViewerPanel({
  filePath, content, scrollToSection, onClose, onOpenFile,
}: FileViewerPanelProps) {
  const theme = useTheme()
  const { width: terminalWidth } = useTerminalDimensions()
  const scrollboxRef = useRef<any>(null)
  const mountedRef = useMountedRef()
  const safeTimeout = useSafeTimeout()
  const markdown = isMarkdownFile(filePath)

  const displayedContent = useMemo(() => content ?? '', [content])
  const copyContent = useMemo(() => content ?? '', [content])
  const codeBlockWidth = Math.max(20, terminalWidth - 10)
  const targetSectionScrollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

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

  const headerLabel = scrollToSection && markdown
    ? `≡  ${filePath} > ${scrollToSection}`
    : `≡  ${filePath}`

  const codeLines = useMemo(
    () => highlightFile(displayedContent, filePath, theme.syntax),
    [displayedContent, filePath, theme.syntax],
  )

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
          <MarkdownContent
            content={displayedContent}
            onOpenFile={onOpenFile}
            codeBlockWidth={codeBlockWidth}
          />
        ) : (
          <box style={{ flexDirection: 'column' }}>
            {codeLines.map((line, idx) => renderCodeLines(line, idx, theme.foreground))}
          </box>
        )}
      </scrollbox>
    </box>
  )
})
