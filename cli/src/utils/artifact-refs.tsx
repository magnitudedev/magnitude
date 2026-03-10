
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
export function extractArtifactRefSegments(content: ReactNode): ArtifactRefSegment[] {
  const flat = flattenToString(content)
  if (flat === null) return [{ type: 'text', content: '' }]
  return splitArtifactRefs(flat)
}

/**
 * Hit zone for a single artifact ref in the rendered text.
 * Character offsets are relative to the flattened text content of the <text> element.
 */
export interface RefHitZone {
  charStart: number
  charEnd: number
  artifactName: string
  section?: string
}

/**
 * Walk a React node tree, preserving all styling (fg, attributes, etc.),
 * while replacing [[artifact-ref]] patterns in text nodes with styled ref spans.
 *
 * Also tracks character offsets for each ref so the caller can do hit-testing
 * on mouse events (since <span> elements can't receive mouse events in OpenTUI).
 *
 * @param node - The styled React tree from the markdown renderer
 * @param refStyle - Style props to apply to ref label spans (fg, attributes)
 * @param refNotFoundStyle - Style props for refs that don't exist
 * @param isRefValid - Function to check if an artifact ref exists
 * @returns { node, hitZones } - Transformed tree + hit zone data for click handling
 */
export function injectArtifactRefsWithHitZones(
  node: ReactNode,
  refStyle: { fg: string; attributes?: number },
  refHoverStyle: { fg: string; attributes?: number },
  refNotFoundStyle: { fg: string; attributes?: number },
  isRefValid: (name: string) => boolean,
  hoveredRefIndex?: number | null,
): { node: ReactNode; hitZones: RefHitZone[] } {
  const hitZones: RefHitZone[] = []
  let charOffset = 0

  /**
   * Extract the text and styling from a node, unwrapping Fragments and spans.
   * Returns { text, fg, attributes } if the node is a simple styled/unstyled text node
   * (possibly wrapped in Fragment(s) and/or a single span), or null if it's complex.
   */
  function extractTextLeaf(node: ReactNode): { text: string; fg?: string; attributes?: number } | null {
    if (typeof node === 'string') return { text: node }

    if (!React.isValidElement(node)) return null
    const el = node as React.ReactElement<Record<string, unknown>>

    // Unwrap Fragment
    if (el.type === React.Fragment || typeof el.type === 'symbol') {
      const children = el.props.children
      if (typeof children === 'string') return { text: children }
      if (React.isValidElement(children)) return extractTextLeaf(children)
      return null
    }

    // Unwrap span with text child
    if (typeof el.type === 'string' && el.type === 'span') {
      const children = el.props.children
      if (typeof children === 'string') {
        return {
          text: children,
          fg: el.props.fg as string | undefined,
          attributes: el.props.attributes as number | undefined,
        }
      }
      return null
    }

    return null
  }

  /**
   * Merge adjacent text children that share the same styling, then walk each.
   * 
   * convertLinesToReactNodes wraps each segment in <Fragment key={...}>,
   * and styled text in <span fg={...} attributes={...}>. Bun's markdown parser
   * splits `[[ref]]` brackets across multiple text nodes.
   * 
   * We unwrap to extract text+style, merge adjacent nodes with identical style,
   * then reconstruct and walk.
   */
  function walkChildren(children: ReactNode[]): ReactNode[] {
    // Extract text leaves where possible
    type MergeEntry =
      | { kind: 'text'; text: string; fg?: string; attributes?: number }
      | { kind: 'node'; node: ReactNode }

    const entries: MergeEntry[] = []
    for (const child of children) {
      const leaf = extractTextLeaf(child)
      if (leaf) {
        const prev = entries.length > 0 ? entries[entries.length - 1] : null
        if (
          prev &&
          prev.kind === 'text' &&
          prev.fg === leaf.fg &&
          prev.attributes === leaf.attributes
        ) {
          prev.text += leaf.text
        } else {
          entries.push({ kind: 'text', ...leaf })
        }
      } else {
        entries.push({ kind: 'node', node: child })
      }
    }

    // Reconstruct and walk
    return entries.map((entry, i) => {
      if (entry.kind === 'text') {
        if (entry.fg || entry.attributes) {
          // Reconstruct as styled span, then walk its text content
          const walked = walk(entry.text)
          return (
            <React.Fragment key={i}>
              <span fg={entry.fg} attributes={entry.attributes}>
                {walked}
              </span>
            </React.Fragment>
          )
        }
        return <React.Fragment key={i}>{walk(entry.text)}</React.Fragment>
      }
      return <React.Fragment key={i}>{walk(entry.node)}</React.Fragment>
    })
  }

  function walk(n: ReactNode): ReactNode {
    if (typeof n === 'string') {
      if (!hasArtifactRefs(n)) {
        charOffset += n.length
        return n
      }
      const segments = splitArtifactRefs(n)
      return segments.map((seg, i) => {
        if (seg.type === 'text') {
          charOffset += seg.content.length
          return <React.Fragment key={i}>{seg.content}</React.Fragment>
        }
        const displayLabel = seg.label ?? (seg.section
          ? `${seg.artifactName}#${seg.section}`
          : seg.artifactName)
        const exists = isRefValid(seg.artifactName)
        const label = exists ? `[≡ ${displayLabel}]` : `[≡ ${displayLabel} (not found)]`
        const refIdx = hitZones.length

        const start = charOffset
        charOffset += label.length
        if (exists) {
          hitZones.push({
            charStart: start,
            charEnd: charOffset,
            artifactName: seg.artifactName,
            section: seg.section,
          })
        }
        const style = !exists
          ? refNotFoundStyle
          : (hoveredRefIndex === refIdx ? refHoverStyle : refStyle)
        return (
          <span key={i} style={{ fg: style.fg }} attributes={style.attributes}>
            {label}
          </span>
        )
      })
    }

    if (n === null || n === undefined || typeof n === 'boolean') return n
    if (typeof n === 'number') {
      charOffset += String(n).length
      return n
    }

    if (Array.isArray(n)) {
      return walkChildren(n)
    }

    if (React.isValidElement(n)) {
      const el = n as React.ReactElement<Record<string, unknown>>
      const children = el.props.children as ReactNode | undefined
      if (children === undefined) return n
      // Normalize children to array for merging
      const childArray = Array.isArray(children) ? children : [children]
      const newChildren = walkChildren(childArray)
      // If single child, unwrap from array
      const unwrapped = Array.isArray(newChildren) && newChildren.length === 1
        ? newChildren[0]
        : newChildren
      return React.cloneElement(el, {}, unwrapped)
    }

    return n
  }

  const result = walk(node)
  return { node: result, hitZones }
}
