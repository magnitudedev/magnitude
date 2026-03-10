/**
 * ArtifactRefLine
 *
 * Renders a line of text+ref segments as inline <span> children of a single
 * <text> element. Clicking a ref chip opens the artifact in the side panel.
 *
 * Click/hover handling: Since <span> elements can't receive mouse events in
 * OpenTUI, we hit-test clicks on the parent <text> element by converting
 * mouse (x,y) to a character index using the text buffer's lineInfo, then
 * checking which ref label's character range was clicked.
 */

import React, { useRef, useState } from 'react'
import { TextAttributes, type TextRenderable, type MouseEvent as OTMouseEvent, type TextBufferView, type LineInfo } from '@opentui/core'
import { useRenderer } from '@opentui/react'
import { useTheme } from '../hooks/use-theme'
import { useArtifacts } from '../hooks/use-artifacts'
import type { ArtifactRefSegment } from '../utils/artifact-refs'

interface RefHitZone {
  segmentIndex: number
  charStart: number
  charEnd: number
}

function getBufferView(textEl: TextRenderable): TextBufferView {
  const record = textEl as object as Record<string, unknown>
  return record['textBufferView'] as TextBufferView
}

function hitTestRef(
  event: OTMouseEvent,
  textRef: TextRenderable,
  hitZones: RefHitZone[],
): number | null {
  if (hitZones.length === 0) return null
  const localX = event.x - textRef.x
  const localY = event.y - textRef.y
  const view = getBufferView(textRef)
  const info: LineInfo = view.lineInfo
  if (localY < 0 || localY >= info.lineStarts.length) return null
  const charIndex = info.lineStarts[localY] + localX
  for (const zone of hitZones) {
    if (charIndex >= zone.charStart && charIndex < zone.charEnd) {
      return zone.segmentIndex
    }
  }
  return null
}

export function ArtifactRefLine({
  segments,
  foreground,
  onOpenArtifact,
}: {
  segments: ArtifactRefSegment[]
  foreground: string
  onOpenArtifact?: (name: string, section?: string) => void
}) {
  const theme = useTheme()
  const renderer = useRenderer()
  const artifactState = useArtifacts()
  const textRef = useRef<TextRenderable | null>(null)
  const [hoveredRef, setHoveredRef] = useState<number | null>(null)

  const spans: React.ReactNode[] = []
  const hitZones: RefHitZone[] = []
  let charOffset = 0

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i]

    if (seg.type === 'text') {
      spans.push(<React.Fragment key={`t-${i}`}>{seg.content}</React.Fragment>)
      charOffset += seg.content.length
      continue
    }

    const displayLabel = seg.section
      ? `${seg.artifactName}#${seg.section}`
      : seg.artifactName
    const exists = !!artifactState?.artifacts.get(seg.artifactName)

    if (!exists) {
      const label = `[≡ ${displayLabel} (not found)]`
      spans.push(
        <span key={`r-${i}`} style={{ fg: theme.muted }} attributes={TextAttributes.DIM}>
          {label}
        </span>
      )
      charOffset += label.length
      continue
    }

    const label = `[≡ ${displayLabel}]`
    const isHovered = hoveredRef === i
    spans.push(
      <span
        key={`r-${i}`}
        style={{ fg: isHovered ? theme.link : theme.primary }}
        attributes={TextAttributes.BOLD}
      >
        {label}
      </span>
    )
    hitZones.push({ segmentIndex: i, charStart: charOffset, charEnd: charOffset + label.length })
    charOffset += label.length
  }

  return (
    <text
      ref={(el: TextRenderable | null) => { textRef.current = el }}
      style={{ fg: foreground, wrapMode: 'word' }}
      onMouseDown={(event: OTMouseEvent) => {
        const el = textRef.current
        if (!el) return
        const hit = hitTestRef(event, el, hitZones)
        if (hit !== null) {
          renderer.clearSelection()
          const seg = segments[hit]
          if (seg.type === 'ref') {
            onOpenArtifact?.(seg.artifactName, seg.section)
          }
        }
      }}
      onMouseMove={(event: OTMouseEvent) => {
        const el = textRef.current
        if (!el) return
        const hit = hitTestRef(event, el, hitZones)
        setHoveredRef(hit)
      }}
      onMouseOut={() => setHoveredRef(null)}
    >
      {spans}
    </text>
  )
}