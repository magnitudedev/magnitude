/**
 * RefStore — tool result storage and reference resolution.
 *
 * Stores tool outputs as OutputNode trees keyed by ref ID.
 * Queries are XPath 3.1 expressions evaluated against a DOM built on demand.
 *
 * Examples:
 *   content          — child element text
 *   //item/@file     — attribute from descendant elements
 *   count(//item)    — count descendants
 *   parse-json(.)    — parse JSON body, then navigate with XQuery map syntax
 */

import { evaluateXPath } from 'fontoxpath'
import { type OutputNode, outputToText, outputToDOM, outputFromDOM } from '../output-tree'

export class RefStore {
  private readonly store = new Map<string, OutputNode[]>()

  set(tag: string, tree: OutputNode): void {
    const stack = this.store.get(tag) ?? []
    stack.push(tree)
    this.store.set(tag, stack)
  }

  has(tag: string, recency = 0): boolean {
    const stack = this.store.get(tag)
    if (!stack) return false
    return (stack.length - 1 - recency) >= 0
  }

  /**
   * Resolve a tool ref by tag + recency, optionally applying an XPath query.
   *
   * Without query: renders the full output tree to text.
   * With query: builds a DOM, evaluates XPath 3.1, converts result back
   * to OutputNode, then renders to text.
   */
  resolve(tag: string, recency = 0, query?: string): string | undefined {
    const stack = this.store.get(tag)
    if (!stack) return undefined
    const idx = stack.length - 1 - recency
    if (idx < 0) return undefined
    const tree = stack[idx]
    if (!query) return outputToText(tree)
    return evaluateQuery(query, tree)
  }
}

function evaluateQuery(query: string, tree: OutputNode): string {
  try {
    const { root } = outputToDOM(tree)
    const result = evaluateXPath(
      query, root, null, null,
      evaluateXPath.ALL_RESULTS_TYPE,
      { language: evaluateXPath.XQUERY_3_1_LANGUAGE },
    )
    if (result.length === 0) return outputToText(tree)
    if (result.length === 1) return renderOne(result[0])
    return result.map(renderOne).join('\n')
  } catch {
    return outputToText(tree)
  }
}

function renderOne(val: unknown): string {
  if (typeof val === 'string' || typeof val === 'number' || typeof val === 'boolean') {
    return String(val)
  }
  if (val !== null && typeof val === 'object' && 'nodeType' in val) {
    const node = val as { nodeType: number; value?: string; textContent?: string | null }
    // Attribute node — extract value
    if (node.nodeType === 2) return node.value ?? ''
    // Element or text node — extract text content (children only, not the element wrapper)
    return node.textContent ?? ''
  }
  return JSON.stringify(val)
}
