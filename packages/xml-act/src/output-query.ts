import { evaluateXPath } from 'fontoxpath'
import type { ContentPart } from '@magnitudedev/tools'
import { Element, type Node } from 'slimdom'
import type { OutputImageNode, OutputNode } from './output-tree'
import { outputToDOM } from './output-tree'

export function observeOutput(tree: OutputNode, observe: string): ContentPart[] {
  if (observe === '.') {
    if (tree.tag === 'element') return renderChildrenParts(tree.children)
    return renderOutputParts(tree)
  }

  try {
    const { root, imageMap } = outputToDOM(tree)
    const result = evaluateXPath(
      observe,
      root,
      null,
      null,
      evaluateXPath.ALL_RESULTS_TYPE,
      { language: evaluateXPath.XQUERY_3_1_LANGUAGE },
    )
    if (result.length === 0) {
      if (tree.tag === 'element') return renderChildrenParts(tree.children)
      return renderOutputParts(tree)
    }

    const parts: ContentPart[] = []
    for (const value of result) {
      const rendered = renderXPathResult(value, imageMap)
      for (const part of rendered) {
        if (part.type === 'text') pushText(parts, part.text)
        else parts.push(part)
      }
    }
    return parts
  } catch {
    if (tree.tag === 'element') return renderChildrenParts(tree.children)
    return renderOutputParts(tree)
  }
}


function pushText(parts: ContentPart[], text: string): void {
  if (!text) return
  const last = parts[parts.length - 1]
  if (last?.type === 'text') {
    parts[parts.length - 1] = { type: 'text', text: last.text + text }
  } else {
    parts.push({ type: 'text', text })
  }
}

export function renderChildrenParts(children: readonly OutputNode[]): ContentPart[] {
  const parts: ContentPart[] = []
  for (const child of children) {
    const childParts = renderOutputParts(child)
    for (const part of childParts) {
      if (part.type === 'text') pushText(parts, part.text)
      else parts.push(part)
    }
  }
  return parts
}

export function renderOutputParts(node: OutputNode): ContentPart[] {
  const parts: ContentPart[] = []

  const walk = (current: OutputNode): void => {
    if (current.tag === 'text') {
      pushText(parts, current.value)
      return
    }
    if (current.tag === 'image') {
      parts.push({
        type: 'image',
        base64: current.base64,
        mediaType: current.mediaType,
        width: current.width,
        height: current.height,
      })
      return
    }

    const attrStr = Object.entries(current.attrs).map(([k, v]) => ` ${k}="${v}"`).join('')
    if (current.children.length === 0) {
      pushText(parts, `<${current.name}${attrStr} />`)
      return
    }

    pushText(parts, `<${current.name}${attrStr}>`)
    for (const child of current.children) walk(child)
    pushText(parts, `</${current.name}>`)
  }

  walk(node)
  return parts
}

function hasMappedImageDescendant(node: Node, imageMap: Map<Node, OutputImageNode>): boolean {
  if (imageMap.has(node)) return true
  for (let i = 0; i < node.childNodes.length; i++) {
    if (hasMappedImageDescendant(node.childNodes[i], imageMap)) return true
  }
  return false
}

function renderDomNodeParts(node: Node, imageMap: Map<Node, OutputImageNode>): ContentPart[] {
  const parts: ContentPart[] = []

  if (imageMap.has(node)) {
    const img = imageMap.get(node)!
    parts.push({
      type: 'image',
      base64: img.base64,
      mediaType: img.mediaType,
      width: img.width,
      height: img.height,
    })
    return parts
  }

  if (node.nodeType === 3) {
    pushText(parts, node.textContent ?? '')
    return parts
  }

  if (node.nodeType === 2) {
    pushText(parts, (node as { value?: string }).value ?? '')
    return parts
  }

  if (node.nodeType === 1) {
    const el = node as Element

    // Preserve previous observe behavior for text-only element matches:
    // return textContent directly unless the matched subtree contains images.
    if (!hasMappedImageDescendant(node, imageMap) && el.textContent) {
      pushText(parts, el.textContent)
      return parts
    }

    const attrs = Array.from(el.attributes).map(a => ` ${a.name}="${a.value}"`).join('')
    if (el.childNodes.length === 0) {
      pushText(parts, `<${el.tagName}${attrs} />`)
      return parts
    }
    pushText(parts, `<${el.tagName}${attrs}>`)
    for (let i = 0; i < el.childNodes.length; i++) {
      const childParts = renderDomNodeParts(el.childNodes[i], imageMap)
      for (const part of childParts) {
        if (part.type === 'text') pushText(parts, part.text)
        else parts.push(part)
      }
    }
    pushText(parts, `</${el.tagName}>`)
    return parts
  }

  pushText(parts, node.textContent ?? '')
  return parts
}

function renderXPathResult(value: unknown, imageMap: Map<Node, OutputImageNode>): ContentPart[] {
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return [{ type: 'text', text: String(value) }]
  }

  if (value !== null && typeof value === 'object' && 'nodeType' in value) {
    return renderDomNodeParts(value as Node, imageMap)
  }

  return [{ type: 'text', text: JSON.stringify(value) }]
}