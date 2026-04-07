import { memo, useMemo, useRef } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTerminalDimensions } from '@opentui/react'
import { useTheme } from '../hooks/use-theme'
import { usePanelStreaming } from '../hooks/use-panel-streaming'
import { useScrollToElement } from '../hooks/use-scroll-to-element'
import type { FilePanelStream } from '../hooks/use-file-panel'
import { StreamingMarkdownContent } from '../markdown/markdown-content'
import { highlightFile } from '../markdown/highlight-file'
import { slugify } from '../markdown/blocks'
import { isMarkdownFile, renderCodeLines } from '../utils/file-lang'
import { highlightCodeLines } from '../utils/file-panel-utils'
import { BOX_CHARS } from '../utils/ui-constants'
import { CloseButton, CopyButton } from './panel-buttons'

interface FileViewerPanelProps {
  filePath: string
  content: string | null
  scrollToSection?: string
  onClose: () => void
  onOpenFile?: (path: string, section?: string) => void
  streaming?: FilePanelStream | null
}

export const FileViewerPanel = memo(function FileViewerPanel({
  filePath, content, scrollToSection, onClose, onOpenFile, streaming,
}: FileViewerPanelProps) {
  const theme = useTheme()
  const { width: terminalWidth } = useTerminalDimensions()
  const scrollboxRef = useRef<any>(null)
  const markdown = isMarkdownFile(filePath)

  const {
    displayedContent,
    showCursor,
    highlightCharRanges,
    highlightAnchorId,
    copyContent,
    isActivelyStreaming,
  } = usePanelStreaming(streaming, content)

  const codeBlockWidth = Math.max(20, terminalWidth - 10)

  const targetSectionId = useMemo(
    () => markdown && scrollToSection ? `section-${slugify(scrollToSection)}` : null,
    [markdown, scrollToSection],
  )

  useScrollToElement(scrollboxRef, targetSectionId)
  useScrollToElement(scrollboxRef, highlightAnchorId, [displayedContent])

  const headerLabel = scrollToSection && markdown
    ? `≡  ${filePath} > ${scrollToSection}`
    : `≡  ${filePath}`

  const codeLines = useMemo(
    () => highlightFile(displayedContent, filePath, theme.syntax),
    [displayedContent, filePath, theme.syntax],
  )
  const highlightedCodeLines = useMemo(
    () => highlightCodeLines(codeLines, highlightCharRanges),
    [codeLines, highlightCharRanges],
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
        stickyStart={isActivelyStreaming && !scrollToSection ? 'bottom' : 'top'}
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
            {highlightedCodeLines.map((line, idx) => renderCodeLines(line, idx, theme.foreground))}
            {showCursor && <text style={{ fg: theme.foreground }}>▍</text>}
          </box>
        )}
      </scrollbox>
    </box>
  )
})
