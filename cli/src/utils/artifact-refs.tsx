/**
 * Artifact Reference Parser
 *
 * Parses [[name]] and [[name#Section]] syntax.
 * Returns structured segments for the renderer to handle.
 */

import React, { type ReactNode } from 'react'

const ARTIFACT_REF_REGEX = /\[\[([^\]#|]+?)(?:#([^\]|]+?))?(?:\|([^\]]+?))?\]\]/g

export function extractSection(content: string, sectionName: string): string | null {
  const lines = content.split('\n')
  const lowerSection = sectionName.toLowerCase()
  let capturing = false
  let capturedLines: string[] = []
  let headingLevel = 0

  for (const line of lines) {
    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/)
    if (headingMatch) {
      if (capturing) {
        const currentLevel = headingMatch[1].length
        if (currentLevel <= headingLevel) {
          break
        }
      }
      if (!capturing && headingMatch[2].trim().toLowerCase() === lowerSection) {
        capturing = true
        headingLevel = headingMatch[1].length
        continue
      }
    }
    if (capturing) {
      capturedLines.push(line)
    }
  }

  if (!capturing) return null
  return capturedLines.join('\n').trim()
}

export type ArtifactRefSegment =
  | { readonly type: 'text'; readonly content: string }
  | { readonly type: 'ref'; readonly artifactName: string; readonly section?: string; readonly label?: string }

/** Split a string into text and artifact ref segments */
export function splitArtifactRefs(text: string): ArtifactRefSegment[] {
  const segments: ArtifactRefSegment[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null

  const regex = new RegExp(ARTIFACT_REF_REGEX.source, 'g')

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      segments.push({ type: 'text', content: text.slice(lastIndex, match.index) })
    }
    segments.push({
      type: 'ref',
      artifactName: match[1],
      section: match[2] || undefined,
      label: match[3] || undefined,
    })
    lastIndex = regex.lastIndex
  }

  if (segments.length === 0) return [{ type: 'text', content: text }]
  if (lastIndex < text.length) {
    segments.push({ type: 'text', content: text.slice(lastIndex) })
  }
  return segments
}

/** Check if a string contains any artifact references */
export function hasArtifactRefs(text: string): boolean {
  return new RegExp(ARTIFACT_REF_REGEX.source).test(text)
}