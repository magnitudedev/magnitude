/**
 * Output Tree — structured AST for tool output.
 *
 * Replaces XML string serialization in the ref pipeline.
 * Values are stored as raw strings (no entity encoding).
 * DOM is built on demand only when XPath queries are needed.
 */

import { Document, Node, Element, Text, Attr } from 'slimdom'
import { AST } from '@effect/schema'
import type { Schema } from '@effect/schema'
import type { XmlBinding, XmlChildBinding, ImageMediaType } from '@magnitudedev/tools'

// =============================================================================
// AST Type
// =============================================================================

export type OutputImageNode =
  { readonly tag: 'image'; readonly base64: string; readonly mediaType: ImageMediaType; readonly width: number; readonly height: number }

export type OutputNode =
  | { readonly tag: 'element'; readonly name: string; readonly attrs: Record<string, string>; readonly children: readonly OutputNode[] }
  | { readonly tag: 'text'; readonly value: string }
  | OutputImageNode

// =============================================================================
// Helpers
// =============================================================================

function el(name: string, attrs: Record<string, string>, children: readonly OutputNode[]): OutputNode {
  return { tag: 'element', name, attrs, children }
}

function text(value: string): OutputNode {
  return { tag: 'text', value }
}

function image(base64: string, mediaType: ImageMediaType, width: number, height: number): OutputImageNode {
  return { tag: 'image', base64, mediaType, width, height }
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
  if (node.tag === 'image') return `[image ${node.mediaType} ${node.width}x${node.height}]`

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
export function outputToDOM(node: OutputNode): { doc: Document; root: Node; imageMap: Map<Node, OutputImageNode> } {
  const doc = new Document()
  const imageMap = new Map<Node, OutputImageNode>()
  const root = nodeToDOM(node, doc, imageMap)
  doc.appendChild(root)
  return { doc, root, imageMap }
}

function nodeToDOM(node: OutputNode, doc: Document, imageMap: Map<Node, OutputImageNode>): Node {
  if (node.tag === 'text') {
    return doc.createTextNode(node.value)
  }

  if (node.tag === 'image') {
    // Images are represented as placeholder DOM nodes for XPath, then recovered via imageMap.
    const elem = doc.createElement('image')
    elem.setAttribute('mediaType', node.mediaType)
    elem.setAttribute('width', String(node.width))
    elem.setAttribute('height', String(node.height))
    imageMap.set(elem, node)
    return elem
  }

  const elem = doc.createElement(node.name)
  for (const [k, v] of Object.entries(node.attrs)) {
    elem.setAttribute(k, v)
  }
  for (const child of node.children) {
    elem.appendChild(nodeToDOM(child, doc, imageMap))
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
  options?: { readonly outputSchema?: Schema.Schema<any> },
): OutputNode {
  const attrs: Record<string, string> = { ...(echoAttrs ?? {}) }

  if ((isToolImageSchema(options?.outputSchema) || (!options?.outputSchema && hasImageValueShape(output))) && hasImageValueShape(output)) {
    return el(tagName, attrs, [image(output.base64, output.mediaType, output.width, output.height)])
  }

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

  return buildWithBinding(tagName, output, binding, attrs, options)
}

function unwrapAst(ast: AST.AST): AST.AST {
  if (ast._tag === 'Transformation') return unwrapAst(ast.from)
  if (ast._tag === 'Refinement') return unwrapAst(ast.from)
  return ast
}

function isToolImageAst(ast: AST.AST): boolean {
  const unwrapped = unwrapAst(ast)
  const identifier = unwrapped.annotations?.[AST.IdentifierAnnotationId]
  if (identifier === 'ToolImage') return true
  if (unwrapped._tag === 'Union') return unwrapped.types.some(isToolImageAst)
  return false
}

function isToolImageSchema(schema?: Schema.Schema<any>): boolean {
  return !!schema && isToolImageAst(schema.ast)
}

function getFieldAstByPath(ast: AST.AST, path: string): AST.AST | undefined {
  const segments = path.split('.')
  let current = unwrapAst(ast)

  for (let i = 0; i < segments.length; i++) {
    if (current._tag !== 'TypeLiteral') return undefined
    const prop = current.propertySignatures.find(p => String(p.name) === segments[i])
    if (!prop) return undefined
    if (i === segments.length - 1) return prop.type
    current = unwrapAst(prop.type)
  }

  return undefined
}

function getArrayElementAstByField(ast: AST.AST, fieldName: string): AST.AST | undefined {
  const unwrapped = unwrapAst(ast)
  if (unwrapped._tag !== 'TypeLiteral') return undefined

  const prop = unwrapped.propertySignatures.find(p => String(p.name) === fieldName)
  if (!prop) return undefined

  const propAst = unwrapAst(prop.type)
  if (propAst._tag === 'TupleType' && propAst.rest.length > 0) {
    return propAst.rest[0].type
  }
  return undefined
}

function isImageMediaType(value: unknown): value is ImageMediaType {
  return value === 'image/png' || value === 'image/jpeg' || value === 'image/webp' || value === 'image/gif'
}

function hasImageValueShape(value: unknown): value is { readonly base64: string; readonly mediaType: ImageMediaType; readonly width: number; readonly height: number } {
  return !!value
    && typeof value === 'object'
    && 'base64' in value
    && 'mediaType' in value
    && 'width' in value
    && 'height' in value
    && typeof (value as Record<string, unknown>).base64 === 'string'
    && isImageMediaType((value as Record<string, unknown>).mediaType)
    && typeof (value as Record<string, unknown>).width === 'number'
    && typeof (value as Record<string, unknown>).height === 'number'
}

function buildWithBinding(
  tagName: string,
  output: unknown,
  binding: Extract<XmlBinding<unknown>, { type: 'tag' }>,
  attrs: Record<string, string>,
  options?: { readonly outputSchema?: Schema.Schema<any> },
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
      const val = resolveNestedField(obj, attr.field)
      if (val !== undefined && val !== null) {
        attrs[attr.attr] = String(val)
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
    const bodyAst = options?.outputSchema ? getFieldAstByPath(options.outputSchema.ast, String(binding.body)) : undefined
    const bodyValue = obj[binding.body!]
    if (((bodyAst && isToolImageAst(bodyAst)) || (!bodyAst && hasImageValueShape(bodyValue))) && hasImageValueShape(bodyValue)) {
      return el(tagName, attrs, [image(bodyValue.base64, bodyValue.mediaType, bodyValue.width, bodyValue.height)])
    }
    return el(tagName, attrs, [text(String(bodyValue))])
  }

  // Has nested content
  const children: OutputNode[] = []

  if (hasChildTags) {
    for (const ct of binding.childTags!) {
      const val = resolveNestedField(obj, ct.field)
      if (val !== undefined) {
        const fieldAst = options?.outputSchema ? getFieldAstByPath(options.outputSchema.ast, ct.field) : undefined
        children.push(buildFieldNode(ct.tag, val, fieldAst, {}, { outputSchema: options?.outputSchema }))
      }
    }
  }

  if (hasChildren) {
    for (const child of binding.children!) {
      const arr = obj[child.field as string]
      if (Array.isArray(arr)) {
        const elementAst = options?.outputSchema ? getArrayElementAstByField(options.outputSchema.ast, String(child.field)) : undefined
        buildChildArray(children, arr, child, elementAst, options)
      }
    }
  }

  if (hasChildRecord) {
    const { field, tag: childTag, keyAttr } = binding.childRecord!
    const record = obj[field as string]
    const valueAst = options?.outputSchema ? getFieldAstByPath(options.outputSchema.ast, String(field)) : undefined
    if (record && typeof record === 'object') {
      for (const [key, val] of Object.entries(record as Record<string, unknown>)) {
        children.push(buildFieldNode(childTag, val, valueAst, { [keyAttr]: key }, { outputSchema: options?.outputSchema }))
      }
    }
  }

  if (hasBody) {
    const bodyValue = obj[binding.body!]
    const bodyAst = options?.outputSchema ? getFieldAstByPath(options.outputSchema.ast, String(binding.body)) : undefined
    if (((bodyAst && isToolImageAst(bodyAst)) || (!bodyAst && hasImageValueShape(bodyValue))) && hasImageValueShape(bodyValue)) {
      children.push(image(bodyValue.base64, bodyValue.mediaType, bodyValue.width, bodyValue.height))
    } else {
      children.push(text(String(bodyValue)))
    }
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
  itemBinding: { tag: string; attributes?: readonly { attr: string; field: string }[]; body?: string | number | symbol },
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
    for (const attrSpec of itemBinding.attributes) {
      const val = itemObj[attrSpec.field]
      if (val !== undefined && val !== null) {
        itemAttrs[attrSpec.attr] = String(val)
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

function buildFieldNode(
  tagName: string,
  value: unknown,
  fieldAst?: AST.AST,
  attrs: Record<string, string> = {},
  options?: { readonly outputSchema?: Schema.Schema<any> },
): OutputNode {
  if (((fieldAst && isToolImageAst(fieldAst)) || (!fieldAst && hasImageValueShape(value))) && hasImageValueShape(value)) {
    return el(tagName, attrs, [image(value.base64, value.mediaType, value.width, value.height)])
  }
  return el(tagName, attrs, [text(String(value))])
}

function buildChildArray(
  children: OutputNode[],
  arr: unknown[],
  child: XmlChildBinding,
  elementAst?: AST.AST,
  options?: { readonly outputSchema?: Schema.Schema<any> },
): void {
  const childTag = child.tag ?? child.field
  for (const item of arr) {
    const itemObj = (item && typeof item === 'object') ? item as Record<string, unknown> : {}
    const childAttrs: Record<string, string> = {}
    if (child.attributes) {
      for (const attr of child.attributes) {
        const val = itemObj[attr.field]
        if (val !== undefined && val !== null) {
          childAttrs[attr.attr] = String(val)
        }
      }
    }
    if (child.body && itemObj[child.body] !== undefined) {
      const bodyAst =
        elementAst && unwrapAst(elementAst)._tag === 'TypeLiteral'
          ? getFieldAstByPath(elementAst, String(child.body))
          : undefined
      children.push(buildFieldNode(childTag, itemObj[child.body], bodyAst, childAttrs, options))
    } else {
      children.push(el(childTag, childAttrs, []))
    }
  }
}
