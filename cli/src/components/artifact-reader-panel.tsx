import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import { BOX_CHARS } from '../utils/ui-constants'
import { MarkdownContent } from './markdown-content'
import { Button } from './button'

interface ContentSection {
  id: string
  heading?: string
  content: string
}

/** Split markdown content into sections at heading boundaries.
 *  Each section includes its heading line so MarkdownContent renders it properly. */
function splitContentBySections(content: string): ContentSection[] {
  const lines = content.split('\n')
  const sections: ContentSection[] = []
  let currentLines: string[] = []
  let currentHeading: string | undefined

  const flush = () => {
    const text = currentLines.join('\n')
    if (text.trim() || currentHeading) {
      sections.push({
        id: currentHeading ? `section-${slugify(currentHeading)}` : 'section-preamble',
        heading: currentHeading,
        content: text,
      })
    }
  }

  for (const line of lines) {
    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/)
    if (headingMatch) {
      flush()
      currentLines = [line] // Start new section with the heading line
      currentHeading = headingMatch[2].trim()
    } else {
      currentLines.push(line)
    }
  }
  flush()

  return sections
}

function slugify(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
}

function CopyButton({ content, theme }: { content: string; theme: any }) {
  const [copied, setCopied] = useState(false)
  const [hovered, setHovered] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const handleCopy = useCallback(() => {
    // Use pbcopy on macOS, xclip on Linux
    const proc = require('child_process').spawn('pbcopy', [], { stdio: ['pipe', 'ignore', 'ignore'] })
    proc.stdin.write(content)
    proc.stdin.end()

    setCopied(true)
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setCopied(false), 2000)
  }, [content])

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [])

  const color = copied ? '#22c55e' : hovered ? theme.foreground : theme.muted

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
  content: string
  scrollToSection?: string
  onClose: () => void
  onOpenArtifact?: (name: string, section?: string) => void
}

export const ArtifactReaderPanel = memo(function ArtifactReaderPanel({
  artifactName, content, scrollToSection, onClose, onOpenArtifact
}: ArtifactReaderPanelProps) {
  const theme = useTheme()
  const scrollboxRef = useRef<any>(null)

  const sections = useMemo(() => splitContentBySections(content), [content])

  // Find the target section ID for scrolling
  const targetSectionId = useMemo(() => {
    if (!scrollToSection) return null
    const lowerTarget = scrollToSection.toLowerCase()
    const section = sections.find(s => s.heading?.toLowerCase() === lowerTarget)
    return section?.id ?? null
  }, [scrollToSection, sections])

  // Scroll to target section after layout
  useEffect(() => {
    if (!targetSectionId) return

    const scrollbox = scrollboxRef.current
    if (!scrollbox) return

    // Disable sticky scroll so we can control position
    scrollbox.stickyScroll = false

    const doScroll = () => {
      const scrollbox = scrollboxRef.current
      if (!scrollbox) return

      const contentNode = scrollbox.content
      if (!contentNode) return

      const targetEl = contentNode.findDescendantById(targetSectionId)
      if (!targetEl) return

      // Walk up parent chain summing yogaNode.getComputedTop() (same pattern as app.tsx:1941-1952)
      let offsetY = 0
      let node: any = targetEl
      while (node && node !== contentNode) {
        const yogaNode = node.yogaNode || node.getLayoutNode?.()
        if (yogaNode) {
          offsetY += yogaNode.getComputedTop()
        }
        node = node.parent
      }

      scrollbox.scrollTo(offsetY)
    }

    // Wait for layout to complete
    setTimeout(doScroll, 50)
  }, [targetSectionId])

  const headerLabel = scrollToSection
    ? `≡  ${artifactName} > ${scrollToSection}`
    : `≡  ${artifactName}`

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
      {/* Header */}
      <box style={{ flexDirection: 'row', justifyContent: 'space-between', flexShrink: 0 }}>
        <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>
          {headerLabel}
        </text>
        <box style={{ flexDirection: 'row', gap: 2 }}>
          <CopyButton content={content} theme={theme} />
          <CloseButton theme={theme} onClose={onClose} />
        </box>
      </box>

      {/* Content - scrollable */}
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
        {sections.length > 1 ? (
          sections.map(section => (
            <box key={section.id} id={section.id}>
              <MarkdownContent content={section.content} onOpenArtifact={onOpenArtifact} />
            </box>
          ))
        ) : (
          <MarkdownContent content={content} onOpenArtifact={onOpenArtifact} />
        )}
      </scrollbox>
    </box>
  )
})
