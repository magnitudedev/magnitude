import { evaluateXPath } from 'fontoxpath'
import type { OutputNode } from './output-tree'
import { outputToDOM, outputToText } from './output-tree'

export function observeOutput(tree: OutputNode, observe: string): string {
  if (observe === '.') return outputToText(tree)

  try {
    const { root } = outputToDOM(tree)
    const result = evaluateXPath(
      observe,
      root,
      null,
      null,
      evaluateXPath.ALL_RESULTS_TYPE,
      { language: evaluateXPath.XQUERY_3_1_LANGUAGE },
    )
    if (result.length === 0) return outputToText(tree)
    if (result.length === 1) return renderOne(result[0], tree)
    return result.map(value => renderOne(value, tree)).join('\n')
  } catch {
    return outputToText(tree)
  }
}

function serializeElement(el: { tagName: string; attributes: ArrayLike<{ name: string; value: string }>; textContent?: string | null; childNodes: ArrayLike<unknown> }): string {
  const attrs = Array.from(el.attributes).map(a => ` ${a.name}="${a.value}"`).join('')
  if (el.childNodes.length === 0) return `<${el.tagName}${attrs} />`
  return `<${el.tagName}${attrs}>${el.textContent ?? ''}</${el.tagName}>`
}

function renderOne(value: unknown, tree: OutputNode): string {
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  if (value !== null && typeof value === 'object' && 'nodeType' in value) {
    const node = value as { nodeType: number; value?: string; textContent?: string | null; tagName: string; attributes: ArrayLike<{ name: string; value: string }>; childNodes: ArrayLike<unknown> }
    // Attribute node
    if (node.nodeType === 2) return node.value ?? ''
    // Element node
    if (node.nodeType === 1) {
      const text = node.textContent
      if (text) return text
      return serializeElement(node)
    }
    return node.textContent ?? outputToText(tree)
  }
  return JSON.stringify(value)
}