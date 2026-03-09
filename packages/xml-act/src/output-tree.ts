/**
 * Output Tree — structured AST for tool output.
 *
 * Replaces XML string serialization in the ref pipeline.
 * Values are stored as raw strings (no entity encoding).
 * DOM is built on demand only when XPath queries are needed.
 */

import { Document, Node, Element, Text, Attr } from 'slimdom'
import type { XmlBinding, XmlChildBinding } from '@magnitudedev/tools'

// =============================================================================
// AST Type
// =============================================================================

export type OutputNode =
  | { readonly tag: 'element'; readonly name: string; readonly attrs: Record<string, string>; readonly children: readonly OutputNode[] }
  | { readonly tag: 'text'; readonly value: string }

// =============================================================================
// Helpers
// =============================================================================

function el(name: string, attrs: Record<string, string>, children: readonly OutputNode[]): OutputNode {
  return { tag: 'element', name, attrs, children }
}

function text(value: string): OutputNode {
  return { tag: 'text', value }
}

// =============================================================================
// Rendering
// =============================================================================

/**
 * Render OutputNode to plain text for injection into tool bodies and LLM context.
 * Elements render their tag structure; text nodes render their raw value.
 * No entity escaping — this is intentionally not valid XML.
 */
export function outputToText(node: OutputNode): string {
  if (node.tag === 'text') return node.value

  const attrStr = Object.entries(node.attrs)
    .map(([k, v]) => ` ${k}="${v}"`)
    .join('')

  if (node.children.length === 0) {
    return `<${node.name}${attrStr} />`
  }

  const inner = node.children.map(outputToText).join('')
  return `<${node.name}${attrStr}>${inner}</${node.name}>`
}

// =============================================================================
// DOM Conversion (for XPath)
// =============================================================================

/**
 * Convert OutputNode to slimdom DOM node for fontoxpath XPath evaluation.
 * Returns the document element (root element).
 */
export function outputToDOM(node: OutputNode): { doc: Document; root: Node } {
  const doc = new Document()
  const root = nodeToDOM(node, doc)
  doc.appendChild(root)
  return { doc, root }
}

function nodeToDOM(node: OutputNode, doc: Document): Node {
  if (node.tag === 'text') {
    return doc.createTextNode(node.value)
  }

  const elem = doc.createElement(node.name)
  for (const [k, v] of Object.entries(node.attrs)) {
    elem.setAttribute(k, v)
  }
  for (const child of node.children) {
    elem.appendChild(nodeToDOM(child, doc))
  }
  return elem
}

/**
 * Convert a slimdom DOM node back to OutputNode after XPath evaluation.
 */
export function outputFromDOM(node: Node): OutputNode {
  if (node instanceof Text) {
    return text(node.data)
  }

  if (node instanceof Attr) {
    return text(node.value)
  }

  if (node instanceof Element) {
    const attrs: Record<string, string> = {}
    for (let i = 0; i < node.attributes.length; i++) {
      const attr = node.attributes[i]
      attrs[attr.name] = attr.value
    }
    const children: OutputNode[] = []
    for (let i = 0; i < node.childNodes.length; i++) {
      children.push(outputFromDOM(node.childNodes[i]))
    }
    return el(node.tagName, attrs, children)
  }

  if (node instanceof Document) {
    if (node.documentElement) return outputFromDOM(node.documentElement)
    return text('')
  }

  // Fallback
  return text(node.textContent ?? '')
}

// =============================================================================
// Build from tool output
// =============================================================================

/**
 * Build an OutputNode tree from a tool's raw output and its XML binding.
 * Replaces serializeOutput in the ref pipeline.
 */
export function buildOutputTree(
  tagName: string,
  output: unknown,
  binding: XmlBinding<unknown>,
  echoAttrs?: Record<string, string>,
): OutputNode {
  const attrs: Record<string, string> = { ...(echoAttrs ?? {}) }

  // Scalar outputs (string, number, boolean)
  if (typeof output === 'string' || typeof output === 'number' || typeof output === 'boolean') {
    return el(tagName, attrs, [text(String(output))])
  }

  // Void outputs (undefined, null)
  if (output === undefined || output === null) {
    return el(tagName, attrs, [])
  }

  // For objects and arrays, require binding
  if (binding.type !== 'tag') {
    throw new Error(`buildOutputTree: tool output <${tagName}> is missing required xmlOutput binding (type: 'tag')`)
  }

  return buildWithBinding(tagName, output, binding, attrs)
}

function buildWithBinding(
  tagName: string,
  output: unknown,
  binding: Extract<XmlBinding<unknown>, { type: 'tag' }>,
  attrs: Record<string, string>,
): OutputNode {
  // Array outputs with items binding
  if (Array.isArray(output) && binding.items) {
    const children = output.map(item => buildItem(item, binding.items!))
    return el(tagName, attrs, children)
  }

  // Object outputs
  const obj = (output && typeof output === 'object') ? output as Record<string, unknown> : {}

  // Binding attributes from output
  if (binding.attributes) {
    for (const attr of binding.attributes) {
      const val = obj[attr]
      if (val !== undefined && val !== null) {
        attrs[attr] = String(val)
      }
    }
  }

  const hasBody = binding.body && obj[binding.body] !== undefined
  const hasChildTags = binding.childTags && binding.childTags.length > 0
  const hasChildren = binding.children && binding.children.length > 0
  const hasChildRecord = !!binding.childRecord
  const hasNested = hasChildTags || hasChildren || hasChildRecord

  // Self-closing
  if (!hasBody && !hasNested) {
    return el(tagName, attrs, [])
  }

  // Body only
  if (hasBody && !hasNested) {
    return el(tagName, attrs, [text(String(obj[binding.body!]))])
  }

  // Has nested content
  const children: OutputNode[] = []

  if (hasChildTags) {
    for (const ct of binding.childTags!) {
      const val = resolveNestedField(obj, ct.field)
      if (val !== undefined) {
        children.push(el(ct.tag, {}, [text(String(val))]))
      }
    }
  }

  if (hasChildren) {
    for (const child of binding.children!) {
      const arr = obj[child.field as string]
      if (Array.isArray(arr)) {
        buildChildArray(children, arr, child)
      }
    }
  }

  if (hasChildRecord) {
    const { field, tag: childTag, keyAttr } = binding.childRecord!
    const record = obj[field as string]
    if (record && typeof record === 'object') {
      for (const [key, val] of Object.entries(record as Record<string, unknown>)) {
        children.push(el(childTag, { [keyAttr]: key }, [text(String(val))]))
      }
    }
  }

  if (hasBody) {
    children.push(text(String(obj[binding.body!])))
  }

  return el(tagName, attrs, children)
}

function resolveNestedField(obj: Record<string, unknown>, path: string): unknown {
  const segments = path.split('.')
  let current: unknown = obj
  for (const seg of segments) {
    if (!current || typeof current !== 'object') return undefined
    current = (current as Record<string, unknown>)[seg]
  }
  return current
}

function buildItem(
  item: unknown,
  itemBinding: { tag: string; attributes?: readonly (string | number | symbol)[]; body?: string | number | symbol },
): OutputNode {
  const itemTag = itemBinding.tag

  // Scalar items
  if (typeof item === 'string' || typeof item === 'number' || typeof item === 'boolean') {
    return el(itemTag, {}, [text(String(item))])
  }

  const itemObj = (item && typeof item === 'object') ? item as Record<string, unknown> : {}

  // Build attributes
  const itemAttrs: Record<string, string> = {}
  if (itemBinding.attributes) {
    for (const attr of itemBinding.attributes) {
      const attrKey = String(attr)
      const val = itemObj[attrKey]
      if (val !== undefined && val !== null) {
        itemAttrs[attrKey] = String(val)
      }
    }
  }

  // Build body
  const bodyKey = itemBinding.body ? String(itemBinding.body) : undefined
  const hasBody = bodyKey && itemObj[bodyKey] !== undefined

  if (hasBody) {
    return el(itemTag, itemAttrs, [text(String(itemObj[bodyKey!]))])
  }
  return el(itemTag, itemAttrs, [])
}

function buildChildArray(children: OutputNode[], arr: unknown[], child: XmlChildBinding): void {
  const childTag = child.tag ?? child.field
  for (const item of arr) {
    const itemObj = (item && typeof item === 'object') ? item as Record<string, unknown> : {}
    const childAttrs: Record<string, string> = {}
    if (child.attributes) {
      for (const attr of child.attributes) {
        const val = itemObj[attr]
        if (val !== undefined && val !== null) {
          childAttrs[attr] = String(val)
        }
      }
    }
    if (child.body && itemObj[child.body] !== undefined) {
      children.push(el(childTag, childAttrs, [text(String(itemObj[child.body]))]))
    } else {
      children.push(el(childTag, childAttrs, []))
    }
  }
}
