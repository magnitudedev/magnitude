import { memo, useMemo } from 'react'
import { TextAttributes, type ScrollBoxRenderable, type Renderable } from '@opentui/core'
import { Atom, AtomRef, useAtomMount } from '@effect-atom/atom-react'
import { Effect, Option } from 'effect'
import { useLocalWidth } from '../../hooks/use-local-width'
import { useTheme } from '../../hooks/use-theme'
import { usePanelStreaming } from '../../hooks/use-panel-streaming'
import type { FilePanelStream } from '../../hooks/use-file-panel'
import { StreamingMarkdownContent } from '../../markdown/markdown-content'
import { highlightFile } from '../../markdown/highlight-file'
import { slugify } from '../../markdown/blocks'
import { isMarkdownFile, renderCodeLines } from '../../utils/file-lang'
import { highlightCodeLines } from '../../utils/file-panel-utils'
import { BOX_CHARS } from '../../utils/ui-constants'
import { CloseButton, CopyButton } from './panel-buttons'

/**
 * Compute the cumulative top offset of a renderable relative to a container.
 * Walks the parent chain summing yoga computed tops.
 */
function computeOffsetFromContainer(target: Renderable, container: Renderable): number {
  let offsetY = 0
  let node: Renderable | null = target
  while (node && node !== container) {
    offsetY += node.getLayoutNode().getComputedTop()
    node = node.parent
  }
  return offsetY
}

/** Scroll the scrollbox to bring a descendant element into view by ID. */
function scrollDescendantIntoView(scrollbox: ScrollBoxRenderable, elementId: string): void {
  const contentNode = scrollbox.content
  if (!contentNode) return
  const target = contentNode.findDescendantById(elementId)
  if (!target) return
  scrollbox.stickyScroll = false
  scrollbox.scrollTo(computeOffsetFromContainer(target, contentNode))
}

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
  const panel = useLocalWidth()
  const scrollboxAtomRef = useMemo(
    () => AtomRef.make<Option.Option<ScrollBoxRenderable>>(Option.none()),
    [],
  )
  const attachScrollbox = useMemo(
    () => (sb: ScrollBoxRenderable | null) => {
      scrollboxAtomRef.set(Option.fromNullable(sb))
    },
    [scrollboxAtomRef],
  )
  const markdown = isMarkdownFile(filePath)

  const {
    displayedContent,
    showCursor,
    highlightCharRanges,
    highlightAnchorId,
    copyContent,
    isActivelyStreaming,
  } = usePanelStreaming(streaming, content)

  const codeBlockWidth = Math.max(20, (panel.width ?? 30) - 10)

  const targetSectionId = useMemo(
    () => markdown && scrollToSection ? `section-${slugify(scrollToSection)}` : null,
    [markdown, scrollToSection],
  )

  // Scroll-to-element — useAtomMount lifecycle. Reads from AtomRef (reactive).
  // Effect.sleep('50 millis') lets OpenTUI's async yoga layout settle before measuring.
  const scrollToSectionAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          if (!targetSectionId) return
          yield* Effect.sleep('50 millis')
          Option.match(scrollboxAtomRef.value, {
            onNone: () => {},
            onSome: (sb) => scrollDescendantIntoView(sb, targetSectionId),
          })
        }),
      ),
    [targetSectionId, scrollboxAtomRef],
  )
  useAtomMount(scrollToSectionAtom)

  const scrollToHighlightAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          if (!highlightAnchorId) return
          yield* Effect.sleep('50 millis')
          Option.match(scrollboxAtomRef.value, {
            onNone: () => {},
            onSome: (sb) => scrollDescendantIntoView(sb, highlightAnchorId),
          })
        }),
      ),
    [highlightAnchorId, displayedContent, scrollboxAtomRef],
  )
  useAtomMount(scrollToHighlightAtom)

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
    <box
      ref={panel.ref}
      onSizeChange={panel.onSizeChange}
      style={{
        flexDirection: 'column',
        height: '100%',
        borderStyle: 'single',
        borderColor: theme.border || theme.muted,
        customBorderChars: BOX_CHARS,
        paddingLeft: 1,
        paddingRight: 1,
      }}
    >
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
        ref={attachScrollbox}
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
