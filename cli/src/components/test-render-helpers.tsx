import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { BlockRenderer } from './block-renderer'
import type { Block, HighlightRange } from '../utils/render-blocks'
import { buildMarkdownColorPalette, chatThemes } from '../utils/theme'

export interface RenderTreeNode {
  kind: 'element'
  type: string
  props: Record<string, unknown>
  children: RenderTree
}

export type RenderTree = Array<RenderTreeNode | string>

function normalizeType(type: unknown): string {
  if (typeof type === 'string') return type
  if (typeof type === 'function') return type.displayName || type.name || 'anonymous'
  if (type && typeof type === 'object') {
    const maybe = type as { displayName?: string; name?: string; $$typeof?: symbol; type?: unknown; render?: (...args: any[]) => any }
    if (maybe.displayName || maybe.name) return maybe.displayName || maybe.name || 'anonymous'
    if (maybe.type) return normalizeType(maybe.type)
    if (maybe.render) return normalizeType(maybe.render)
    if (typeof maybe.$$typeof === 'symbol') return String(maybe.$$typeof)
  }
  if (typeof type === 'symbol') return String(type)
  return 'unknown'
}

function resolveNode(node: React.ReactNode): React.ReactNode {
  if (node == null || typeof node === 'boolean') return null
  if (typeof node === 'string' || typeof node === 'number') return node
  if (Array.isArray(node)) return node.map(resolveNode)
  if (!React.isValidElement(node)) return String(node)

  const elementType = node.type as unknown

  if (typeof elementType === 'function') {
    const rendered = elementType(node.props)
    return resolveNode(rendered)
  }

  if (elementType && typeof elementType === 'object') {
    const maybe = elementType as { type?: unknown; render?: (...args: any[]) => any }
    if (typeof maybe.type === 'function') {
      const rendered = maybe.type(node.props)
      return resolveNode(rendered)
    }
    if (typeof maybe.render === 'function') {
      const rendered = maybe.render(node.props, null)
      return resolveNode(rendered)
    }
  }

  const props = node.props as Record<string, unknown>
  return React.cloneElement(
    node,
    props,
    resolveNode(props.children as React.ReactNode),
  )
}

function walkResolvedNode(node: React.ReactNode): RenderTree {
  if (node == null || typeof node === 'boolean') return []
  if (typeof node === 'string' || typeof node === 'number') return [String(node)]
  if (Array.isArray(node)) return node.flatMap((child) => walkResolvedNode(child))
  if (!React.isValidElement(node)) return [String(node)]

  const props = (node.props ?? {}) as Record<string, unknown>
  const { children, ...rest } = props

  return [
    {
      kind: 'element',
      type: normalizeType(node.type),
      props: rest,
      children: walkResolvedNode(children as React.ReactNode),
    },
  ]
}

type BlockRendererProps = {
  foreground?: string
  showCursor?: boolean
  highlightAnchorId?: string
  highlights?: HighlightRange[]
  onOpenArtifact?: (name: string, section?: string) => void
  contentWidth?: number
}

const palette = buildMarkdownColorPalette(chatThemes.dark)

export function renderBlocksToTree(blocks: Block[], props?: BlockRendererProps): RenderTree {
  const element = (
    <BlockRenderer
      blocks={blocks}
      foreground={props?.foreground ?? 'white'}
      palette={palette}
      contentWidth={props?.contentWidth ?? 79}
      showCursor={props?.showCursor}
      highlightAnchorId={props?.highlightAnchorId}
      highlights={props?.highlights}
      onOpenArtifact={props?.onOpenArtifact}
    />
  )

  return walkResolvedNode(resolveNode(element))
}

export function renderBlocksToStaticMarkup(blocks: Block[], props?: BlockRendererProps): string {
  return renderToStaticMarkup(
    <div>
      <BlockRenderer
        blocks={blocks}
        foreground={props?.foreground ?? 'white'}
        palette={palette}
        contentWidth={props?.contentWidth ?? 79}
        showCursor={props?.showCursor}
        highlightAnchorId={props?.highlightAnchorId}
        highlights={props?.highlights}
        onOpenArtifact={props?.onOpenArtifact}
      />
    </div>,
  )
}

export function extractAllText(tree: RenderTree | RenderTreeNode | string): string {
  if (typeof tree === 'string') return tree
  if (Array.isArray(tree)) return tree.map((child) => extractAllText(child)).join('')
  return extractAllText(tree.children)
}

export function findNodesByType(tree: RenderTree, type: string): RenderTreeNode[] {
  const matches: RenderTreeNode[] = []

  const visit = (node: RenderTreeNode | string) => {
    if (typeof node === 'string') return
    if (node.type === type) matches.push(node)
    node.children.forEach(visit)
  }

  tree.forEach(visit)
  return matches
}

export function findNodesWithStyle(tree: RenderTree, styleProp: string, value: unknown): RenderTreeNode[] {
  const matches: RenderTreeNode[] = []

  const visit = (node: RenderTreeNode | string) => {
    if (typeof node === 'string') return
    const style = node.props.style
    if (style && typeof style === 'object' && (style as Record<string, unknown>)[styleProp] === value) {
      matches.push(node)
    }
    node.children.forEach(visit)
  }

  tree.forEach(visit)
  return matches
}

export function extractTextFromStaticMarkup(html: string): string {
  // In OpenTUI, each <text> element occupies its own line.
  // Simulate this by inserting newlines between </text><text> boundaries.
  return html
    .replace(/<\/text>\s*<text[^>]*>/g, '\n')
    .replace(/<[^>]+>/g, '')
    .replace(/&gt;/g, '>')
    .replace(/&lt;/g, '<')
    .replace(/&amp;/g, '&')
}