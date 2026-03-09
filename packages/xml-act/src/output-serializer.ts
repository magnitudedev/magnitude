/**
 * XML Output Serializer
 *
 * Serializes tool output to XML string based on either an explicit
 * XmlBinding<TOutput> or schema-derived defaults from the tool's outputSchema.
 */

import { AST } from '@effect/schema'
import type { XmlBinding, XmlChildBinding } from '@magnitudedev/tools'

// =============================================================================
// Schema Introspection Helpers
// =============================================================================

function unwrapAst(ast: AST.AST): AST.AST {
  if (ast._tag === 'Transformation') return unwrapAst(ast.from)
  if (ast._tag === 'Refinement') return unwrapAst(ast.from)
  return ast
}

/**
 * Classify a schema AST into a shape category for default rendering.
 */
type OutputShape =
  | { readonly kind: 'void' }
  | { readonly kind: 'scalar'; readonly type: 'string' | 'number' | 'boolean' }
  | { readonly kind: 'struct'; readonly fields: ReadonlyArray<StructFieldInfo> }
  | { readonly kind: 'array-struct'; readonly fields: ReadonlyArray<StructFieldInfo> }
  | { readonly kind: 'array-scalar'; readonly elementType: string }
  | { readonly kind: 'unknown' }

interface StructFieldInfo {
  readonly name: string
  readonly type: 'string' | 'number' | 'boolean' | 'unknown'
  readonly optional: boolean
}

function classifyOutputShape(schemaAst: AST.AST): OutputShape {
  const ast = unwrapAst(schemaAst)

  // Void (UndefinedKeyword, VoidKeyword, NeverKeyword, or Unit-like)
  if (ast._tag === 'UndefinedKeyword' || ast._tag === 'VoidKeyword' || ast._tag === 'NeverKeyword') {
    return { kind: 'void' }
  }

  // String
  if (ast._tag === 'StringKeyword') return { kind: 'scalar', type: 'string' }

  // Number
  if (ast._tag === 'NumberKeyword') return { kind: 'scalar', type: 'number' }

  // Boolean
  if (ast._tag === 'BooleanKeyword') return { kind: 'scalar', type: 'boolean' }

  // Struct (TypeLiteral with property signatures, no index signatures)
  if (ast._tag === 'TypeLiteral' && ast.propertySignatures.length > 0 && ast.indexSignatures.length === 0) {
    const fields: StructFieldInfo[] = []
    for (const prop of ast.propertySignatures) {
      fields.push({
        name: String(prop.name),
        type: classifyScalarType(unwrapAst(prop.type)),
        optional: prop.isOptional,
      })
    }
    return { kind: 'struct', fields }
  }

  // Array (TupleType with rest elements)
  if (ast._tag === 'TupleType' && ast.rest.length > 0) {
    const elemAst = unwrapAst(ast.rest[0].type)

    // Array<Struct>
    if (elemAst._tag === 'TypeLiteral' && elemAst.propertySignatures.length > 0) {
      const fields: StructFieldInfo[] = []
      for (const prop of elemAst.propertySignatures) {
        fields.push({
          name: String(prop.name),
          type: classifyScalarType(unwrapAst(prop.type)),
          optional: prop.isOptional,
        })
      }
      return { kind: 'array-struct', fields }
    }

    // Array<scalar>
    return { kind: 'array-scalar', elementType: classifyScalarType(elemAst) }
  }

  return { kind: 'unknown' }
}

function classifyScalarType(ast: AST.AST): 'string' | 'number' | 'boolean' | 'unknown' {
  if (ast._tag === 'StringKeyword') return 'string'
  if (ast._tag === 'NumberKeyword') return 'number'
  if (ast._tag === 'BooleanKeyword') return 'boolean'
  // Literal types
  if (ast._tag === 'Literal') {
    if (typeof ast.literal === 'string') return 'string'
    if (typeof ast.literal === 'number') return 'number'
    if (typeof ast.literal === 'boolean') return 'boolean'
  }
  // Unions of scalars (e.g. "file" | "dir")
  if (ast._tag === 'Union') {
    const types = ast.types.map(t => classifyScalarType(unwrapAst(t)))
    // Filter out undefined types from optional unions
    const nonUnknown = types.filter(t => t !== 'unknown')
    if (nonUnknown.length > 0 && nonUnknown.every(t => t === nonUnknown[0])) return nonUnknown[0]
  }
  return 'unknown'
}

// =============================================================================
// XML Escaping
// =============================================================================

function escapeXmlAttr(value: string): string {
  return value.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function escapeXmlBody(value: string): string {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

// =============================================================================
// Serializer
// =============================================================================

/**
 * Serialize tool output to XML string.
 *
 * @param tagName - The XML tag name for the result element
 * @param output - The raw tool output value
 * @param binding - Explicit output binding (XmlBinding) - required for objects/arrays
 * @param echoAttrs - Input attributes to echo on the result tag for context
 */
export function serializeOutput(
  tagName: string,
  output: unknown,
  binding: XmlBinding<unknown>,
  echoAttrs?: Record<string, string>,
): string {
  // Handle scalar outputs (string, number, boolean)
  if (typeof output === 'string' || typeof output === 'number' || typeof output === 'boolean') {
    const attrs = buildAttrString(echoAttrs)
    return `<${tagName}${attrs}>${escapeXmlBody(String(output))}</${tagName}>`
  }

  // Handle void outputs (undefined, null)
  if (output === undefined || output === null) {
    const attrs = buildAttrString(echoAttrs)
    return `<${tagName}${attrs} />`
  }

  // For objects and arrays, require binding
  if (binding.type !== 'tag') {
    throw new Error(`serializeOutput: tool output <${tagName}> is missing required xmlOutput binding (type: 'tag')`)
  }

  return serializeWithBinding(tagName, output, binding, echoAttrs)
}

function buildAttrString(echoAttrs?: Record<string, string>): string {
  if (!echoAttrs || Object.keys(echoAttrs).length === 0) return ''
  return Object.entries(echoAttrs).map(([k, v]) => ` ${k}="${escapeXmlAttr(v)}"`).join('')
}

function serializeWithBinding(
  tagName: string,
  output: unknown,
  binding: Extract<XmlBinding<unknown>, { type: 'tag' }>,
  echoAttrs?: Record<string, string>,
): string {
  const attrs = buildAttrString(echoAttrs)

  // Handle array outputs with items binding
  if (Array.isArray(output) && binding.items) {
    const items = output.map(item => serializeItem(item, binding.items!))
    const parts: string[] = [`<${tagName}${attrs}>`]
    parts.push(...items)
    parts.push(`</${tagName}>`)
    return parts.join('\n')
  }

  // Handle object outputs
  const obj = (output && typeof output === 'object') ? output as Record<string, unknown> : {}

  // Build binding attributes from output
  let bindingAttrs = ''
  if (binding.attributes) {
    for (const attr of binding.attributes) {
      const val = obj[attr]
      if (val !== undefined && val !== null) {
        bindingAttrs += ` ${attr}="${escapeXmlAttr(String(val))}"`
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
    return `<${tagName}${attrs}${bindingAttrs} />`
  }

  // Body only
  if (hasBody && !hasNested) {
    return `<${tagName}${attrs}${bindingAttrs}>${escapeXmlBody(String(obj[binding.body!]))}</${tagName}>`
  }

  // Has nested content
  const parts: string[] = [`<${tagName}${attrs}${bindingAttrs}>`]

  if (hasChildTags) {
    for (const ct of binding.childTags!) {
      const val = resolveNestedField(obj, ct.field)
      if (val !== undefined) {
        parts.push(`<${ct.tag}>${escapeXmlBody(String(val))}</${ct.tag}>`)
      }
    }
  }

  if (hasChildren) {
    for (const child of binding.children!) {
      const arr = obj[child.field as string]
      if (Array.isArray(arr)) {
        serializeChildArray(parts, arr, child)
      }
    }
  }

  if (hasChildRecord) {
    const { field, tag, keyAttr } = binding.childRecord!
    const record = obj[field as string]
    if (record && typeof record === 'object') {
      for (const [key, val] of Object.entries(record as Record<string, unknown>)) {
        parts.push(`<${tag} ${keyAttr}="${escapeXmlAttr(key)}">${escapeXmlBody(String(val))}</${tag}>`)
      }
    }
  }

  if (hasBody) {
    parts.push(escapeXmlBody(String(obj[binding.body!])))
  }

  parts.push(`</${tagName}>`)
  return parts.join('\n')
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

function serializeItem(item: unknown, itemBinding: { tag: string; attributes?: readonly (string | number | symbol)[]; body?: string | number | symbol }): string {
  const itemTag = itemBinding.tag

  // Scalar items (string, number, boolean) — render as body text
  if (typeof item === 'string' || typeof item === 'number' || typeof item === 'boolean') {
    return `<${itemTag}>${escapeXmlBody(String(item))}</${itemTag}>`
  }

  const itemObj = (item && typeof item === 'object') ? item as Record<string, unknown> : {}

  // Build attributes
  let itemAttrs = ''
  if (itemBinding.attributes) {
    for (const attr of itemBinding.attributes) {
      const attrKey = String(attr)
      const val = itemObj[attrKey]
      if (val !== undefined && val !== null) {
        itemAttrs += ` ${attrKey}="${escapeXmlAttr(String(val))}"`
      }
    }
  }

  // Build body
  const bodyKey = itemBinding.body ? String(itemBinding.body) : undefined
  const hasBody = bodyKey && itemObj[bodyKey] !== undefined

  if (hasBody) {
    return `<${itemTag}${itemAttrs}>${escapeXmlBody(String(itemObj[bodyKey!]))}</${itemTag}>`
  } else {
    return `<${itemTag}${itemAttrs} />`
  }
}

function serializeChildArray(parts: string[], arr: unknown[], child: XmlChildBinding): void {
  const childTag = child.tag ?? child.field
  for (const item of arr) {
    const itemObj = (item && typeof item === 'object') ? item as Record<string, unknown> : {}
    let childAttrs = ''
    if (child.attributes) {
      for (const attr of child.attributes) {
        const val = itemObj[attr]
        if (val !== undefined && val !== null) {
          childAttrs += ` ${attr}="${escapeXmlAttr(String(val))}"`
        }
      }
    }
    if (child.body && itemObj[child.body] !== undefined) {
      parts.push(`<${childTag}${childAttrs}>${escapeXmlBody(String(itemObj[child.body]))}</${childTag}>`)
    } else {
      parts.push(`<${childTag}${childAttrs} />`)
    }
  }
}

function serializeDefault(
  tagName: string,
  output: unknown,
  shape: OutputShape,
  echoAttrs?: Record<string, string>,
): string {
  const attrs = buildAttrString(echoAttrs)

  switch (shape.kind) {
    case 'void':
      return `<${tagName}${attrs} />`

    case 'scalar':
      return `<${tagName}${attrs}>${escapeXmlBody(String(output))}</${tagName}>`

    case 'struct': {
      const obj = (output && typeof output === 'object') ? output as Record<string, unknown> : {}
      const parts: string[] = [`<${tagName}${attrs}>`]
      for (const field of shape.fields) {
        const val = obj[field.name]
        if (val !== undefined) {
          parts.push(`<${field.name}>${escapeXmlBody(String(val))}</${field.name}>`)
        }
      }
      parts.push(`</${tagName}>`)
      return parts.join('\n')
    }

    case 'array-struct': {
      const arr = Array.isArray(output) ? output : []
      const parts: string[] = [`<${tagName}${attrs}>`]
      for (const item of arr) {
        const itemObj = (item && typeof item === 'object') ? item as Record<string, unknown> : {}
        // Scalar fields as attributes
        const itemAttrs = shape.fields
          .filter(f => itemObj[f.name] !== undefined)
          .map(f => ` ${f.name}="${escapeXmlAttr(String(itemObj[f.name]))}"`)
          .join('')
        parts.push(`<item${itemAttrs} />`)
      }
      parts.push(`</${tagName}>`)
      return parts.join('\n')
    }

    case 'array-scalar': {
      const arr = Array.isArray(output) ? output : []
      const parts: string[] = [`<${tagName}${attrs}>`]
      for (const item of arr) {
        parts.push(`<item>${escapeXmlBody(String(item))}</item>`)
      }
      parts.push(`</${tagName}>`)
      return parts.join('\n')
    }

    case 'unknown':
      return `<${tagName}${attrs}>${escapeXmlBody(JSON.stringify(output))}</${tagName}>`
  }
}
