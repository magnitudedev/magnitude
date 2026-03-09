
/**
 * Artifact Reference Parser
 *
 * Parses [[name]] and [[name#Section]] syntax.
 * Returns structured segments for the renderer to handle.
 */

import React, { type ReactNode } from 'react'

const ARTIFACT_REF_REGEX = /\[\[([^\]#]+?)(?:#([^\]]+?))?\]\]/g

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
  | { readonly type: 'ref'; readonly artifactName: string; readonly section?: string }

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

/**
 * Flatten a ReactNode tree (from Bun's markdown renderer) into a single string.
 * Bun wraps text fragments in Fragments, so we unwrap those.
 * Returns null if the tree contains non-string-like elements.
 */
function flattenToString(node: ReactNode): string | null {
  if (typeof node === 'string') return node
  if (node === null || node === undefined) return ''
  if (typeof node === 'number') return String(node)
  if (typeof node === 'boolean') return ''

  if (Array.isArray(node)) {
    let result = ''
    for (const child of node) {
      const s = flattenToString(child)
      if (s === null) return null
      result += s
    }
    return result
  }

  if (React.isValidElement(node)) {
    const el = node as React.ReactElement<{ children?: ReactNode }>
    if (el.type === React.Fragment || typeof el.type === 'symbol') {
      return flattenToString(el.props.children)
    }
    if (el.props.children !== undefined) {
      return flattenToString(el.props.children)
    }
    return null
  }

  return null
}

/**
 * Extract artifact ref segments from chunk content (a ReactNode from markdown rendering).
 * Returns null if no artifact refs found — caller should render normally.
 * Returns segments array if refs found — caller should render via ArtifactRefLine.
 */
export function extractArtifactRefSegments(content: ReactNode): ArtifactRefSegment[] | null {
  const flat = flattenToString(content)
  if (flat === null || !hasArtifactRefs(flat)) return null
  return splitArtifactRefs(flat)
}
