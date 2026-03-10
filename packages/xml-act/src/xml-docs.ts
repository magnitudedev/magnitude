/**
 * XML Tool Documentation Generator
 *
 * Generates unified annotated XML documentation from tool definitions.
 * Reads `tool.bindings?.xmlInput` and `tool.bindings?.xmlOutput` to produce
 * inline-annotated XML examples showing both invocation syntax and return shape.
 */

import { Option } from 'effect'
import { AST } from '@effect/schema'
import type { Tool, XmlBinding } from '@magnitudedev/tools'
import type { XmlTagBinding } from './types'

// =============================================================================
// Schema Introspection Helpers
// =============================================================================

/**
 * Unwrap AST to get the structural input type (handles Transformation/Refinement).
 */
export function unwrapAst(ast: AST.AST): AST.AST {
  if (ast._tag === 'Transformation') return unwrapAst(ast.from)
  if (ast._tag === 'Refinement') return unwrapAst(ast.from)
  return ast
}

/**
 * Extract description annotation from a property signature's own annotations.
 */
function getPropDescription(prop: AST.PropertySignature): string | undefined {
  const annotations = prop.annotations
  const desc = annotations[AST.DescriptionAnnotationId]
  return typeof desc === 'string' ? desc : undefined
}

/**
 * Extract description annotation from an AST node, unwrapping if needed.
 */
export function getDescription(propType: AST.AST): string | undefined {
  const desc = AST.getDescriptionAnnotation(propType)
  if (Option.isSome(desc)) return desc.value
  const unwrapped = unwrapAst(propType)
  const desc2 = AST.getDescriptionAnnotation(unwrapped)
  if (Option.isSome(desc2)) return desc2.value
  if (unwrapped._tag === 'Union') {
    for (const t of unwrapped.types) {
      const d = AST.getDescriptionAnnotation(t)
      if (Option.isSome(d)) return d.value
    }
  }
  return undefined
}

interface FieldInfo {
  readonly name: string
  readonly optional: boolean
  readonly description: string | undefined
  readonly ast: AST.AST
}

/**
 * Extract field info from a struct schema's AST.
 */
export function getFieldInfos(schemaAst: AST.AST): Map<string, FieldInfo> {
  const result = new Map<string, FieldInfo>()
  const ast = unwrapAst(schemaAst)
  if (ast._tag === 'TypeLiteral') {
    for (const prop of ast.propertySignatures) {
      const name = String(prop.name)
      const typeDesc = getDescription(prop.type)
      const propDesc = getPropDescription(prop)
      result.set(name, {
        name,
        optional: prop.isOptional,
        description: typeDesc ?? propDesc,
        ast: prop.type,
      })
    }
  }
  return result
}

/**
 * Find the first Record-typed field name in a struct schema.
 */
function findRecordFieldName(schemaAst: AST.AST): string | undefined {
  const ast = unwrapAst(schemaAst)
  if (ast._tag !== 'TypeLiteral') return undefined
  for (const prop of ast.propertySignatures) {
    if (isRecordAst(unwrapAst(prop.type))) return String(prop.name)
  }
  return undefined
}

function isRecordAst(ast: AST.AST): boolean {
  if (ast._tag === 'TypeLiteral' && ast.indexSignatures.length > 0) return true
  if (ast._tag === 'Union') {
    return ast.types.some(t => {
      const u = unwrapAst(t)
      return u._tag === 'TypeLiteral' && u.indexSignatures.length > 0
    })
  }
  return false
}

/**
 * Resolve field info by dotted path (e.g. 'options.type' or 'name').
 */
export function resolveFieldInfoByPath(schemaAst: AST.AST, path: string): FieldInfo | undefined {
  const segments = path.split('.')
  let currentAst = unwrapAst(schemaAst)

  for (let i = 0; i < segments.length; i++) {
    if (currentAst._tag !== 'TypeLiteral') return undefined
    const prop = currentAst.propertySignatures.find(p => String(p.name) === segments[i])
    if (!prop) return undefined

    if (i === segments.length - 1) {
      return {
        name: segments[i],
        optional: prop.isOptional,
        description: getDescription(prop.type),
        ast: prop.type,
      }
    }
    currentAst = unwrapAst(prop.type)
  }
  return undefined
}

/**
 * Extract field info from an array field's element type.
 */
export function getArrayElementFieldInfos(schemaAst: AST.AST, arrayFieldName: string): Map<string, FieldInfo> {
  const ast = unwrapAst(schemaAst)
  if (ast._tag !== 'TypeLiteral') return new Map()

  for (const prop of ast.propertySignatures) {
    if (String(prop.name) !== arrayFieldName) continue
    const propAst = unwrapAst(prop.type)
    if (propAst._tag === 'TupleType' && propAst.rest.length > 0) {
      const elemAst = unwrapAst(propAst.rest[0].type)
      return getFieldInfos(elemAst)
    }
  }
  return new Map()
}

// =============================================================================
// Type Comment Resolution
// =============================================================================

/**
 * Determine if a field needs a type comment in the annotated example.
 * Returns the comment string or null if the type is self-evident (string).
 */
export function needsTypeComment(ast: AST.AST): string | null {
  const unwrapped = unwrapAst(ast)

  if (unwrapped._tag === 'NumberKeyword') return 'number'
  if (unwrapped._tag === 'BooleanKeyword') return 'boolean'

  if (unwrapped._tag === 'Union') {
    // Filter out undefined from optional unions
    const nonUndef = unwrapped.types.filter(t => unwrapAst(t)._tag !== 'UndefinedKeyword')
    const literals = nonUndef.filter(t => unwrapAst(t)._tag === 'Literal')
    if (literals.length === nonUndef.length && literals.length > 0) {
      return literals.map(t => JSON.stringify((unwrapAst(t) as AST.Literal).literal)).join(' | ')
    }
    // Check if all members are the same scalar type
    if (nonUndef.length > 0) {
      const first = unwrapAst(nonUndef[0])
      if (first._tag === 'NumberKeyword') return 'number'
      if (first._tag === 'BooleanKeyword') return 'boolean'
    }
  }

  return null
}

// =============================================================================
// Annotation Building
// =============================================================================

/**
 * Build inline annotation comment for a field.
 * Combines optionality, type comments, and descriptions.
 */
function buildAnnotation(info: FieldInfo | undefined, fieldAst?: AST.AST): string {
  const parts: string[] = []

  if (info?.optional) parts.push('optional')
  else parts.push('required')

  const ast = fieldAst ?? info?.ast
  if (ast) {
    const typeComment = needsTypeComment(ast)
    if (typeComment) parts.push(typeComment)
  }

  if (info?.description) parts.push('— ' + info.description)

  return `<!-- ${parts.join('. ')} -->`
}

function buildAttrAnnotation(info: FieldInfo | undefined): string {
  const parts: string[] = []
  if (info?.optional) parts.push('optional')

  const ast = info?.ast
  if (ast) {
    const typeComment = needsTypeComment(ast)
    if (typeComment) parts.push(typeComment)
  }

  if (info?.description) parts.push('— ' + info.description)

  if (parts.length === 0) return ''
  return ` <!-- ${parts.join('. ')} -->`
}

// =============================================================================
// Output Shape Classification (for doc generation)
// =============================================================================

type OutputDocShape =
  | { readonly kind: 'void' }
  | { readonly kind: 'string' }
  | { readonly kind: 'scalar'; readonly type: string }
  | { readonly kind: 'struct'; readonly fields: Map<string, FieldInfo> }
  | { readonly kind: 'array-struct'; readonly fields: Map<string, FieldInfo> }
  | { readonly kind: 'array-scalar'; readonly elementType: string }
  | { readonly kind: 'unknown' }

function classifyOutputDoc(schemaAst: AST.AST): OutputDocShape {
  const ast = unwrapAst(schemaAst)

  if (ast._tag === 'UndefinedKeyword' || ast._tag === 'VoidKeyword' || ast._tag === 'NeverKeyword') {
    return { kind: 'void' }
  }
  if (ast._tag === 'StringKeyword') return { kind: 'string' }
  if (ast._tag === 'NumberKeyword') return { kind: 'scalar', type: 'number' }
  if (ast._tag === 'BooleanKeyword') return { kind: 'scalar', type: 'boolean' }

  if (ast._tag === 'TypeLiteral' && ast.propertySignatures.length > 0 && ast.indexSignatures.length === 0) {
    return { kind: 'struct', fields: getFieldInfos(schemaAst) }
  }

  if (ast._tag === 'TupleType' && ast.rest.length > 0) {
    const elemAst = unwrapAst(ast.rest[0].type)
    if (elemAst._tag === 'TypeLiteral' && elemAst.propertySignatures.length > 0) {
      return { kind: 'array-struct', fields: getFieldInfos(elemAst) }
    }
    const scalarType = needsTypeComment(elemAst)
    return { kind: 'array-scalar', elementType: scalarType ?? 'string' }
  }

  return { kind: 'unknown' }
}

// =============================================================================
// XML Tag Doc Generation
// =============================================================================

/**
 * Derive a default XML tag name from a tool.
 */
export function defaultXmlTagName(tool: { name: string; group?: string }): string {
  const group = tool.group
  if (!group || group === 'default') return tool.name
  return `${group}-${tool.name}`
}

/**
 * Generate unified annotated XML documentation for a single tool.
 * Returns null if the tool has no XML input binding or is omitted.
 */
export function generateXmlToolDoc(tool: Tool.Any): string | null {
  const binding = tool.bindings?.xmlInput as XmlBinding<unknown> | undefined
  if (!binding) return null

  const tagName = binding.tag ?? defaultXmlTagName(tool)
  const fields = getFieldInfos(tool.inputSchema.ast)
  const lines: string[] = []

  // 1. Description
  if (tool.description) {
    lines.push(tool.description)
    lines.push('')
  }

  // 2. Annotated input example
  buildAnnotatedInput(lines, tagName, binding, fields, tool.inputSchema.ast)

  // 3. Output documentation
  const outputBinding = tool.bindings?.xmlOutput as XmlBinding<unknown> | undefined
  buildAnnotatedOutput(lines, tagName, tool.outputSchema.ast, outputBinding)

  return lines.join('\n')
}

// =============================================================================
// Annotated Input Builder
// =============================================================================

function buildAnnotatedInput(
  lines: string[],
  tagName: string,
  binding: XmlTagBinding,
  fields: Map<string, FieldInfo>,
  schemaAst: AST.AST,
): void {
  const hasBody = !!binding.body
  const hasChildren = binding.children && binding.children.length > 0
  const hasChildTags = binding.childTags && binding.childTags.length > 0
  const hasChildRecord = !!binding.childRecord
  const hasNested = hasChildren || hasChildTags || hasChildRecord
  const hasAttrs = binding.attributes && binding.attributes.length > 0

  // Multi-line format when we have attrs to annotate
  if (hasAttrs && (binding.attributes!.length > 1 || fields.get(binding.attributes![0])?.description)) {
    // Opening tag on own line
    lines.push(`<${tagName}`)

    // Each attribute on its own line with inline annotation
    for (const attr of binding.attributes!) {
      const info = fields.get(attr)
      lines.push(`${attr}="..."${buildAttrAnnotation(info)}`)
    }

    if (!hasBody && !hasNested) {
      lines.push('/>')
    } else if (hasBody && !hasNested) {
      lines.push(`>${binding.body}</${tagName}>`)
      const bodyInfo = fields.get(binding.body!)
      if (bodyInfo?.description) {
        lines.push(`<!-- ${binding.body} (${bodyInfo.optional ? 'optional' : 'required'}, body) — ${bodyInfo.description} -->`)
      }
    } else {
      lines.push('>')
    }
  } else {
    // Single-line opening tag
    let open = `<${tagName}`
    if (hasAttrs) {
      for (const attr of binding.attributes!) {
        open += ` ${attr}="..."`
      }
      const info = fields.get(binding.attributes![0])
      open += buildAttrAnnotation(info)
    }

    if (!hasBody && !hasNested) {
      lines.push(`${open} />`)
    } else if (hasBody && !hasNested) {
      lines.push(`${open}>${binding.body}</${tagName}>`)
      const bodyInfo = fields.get(binding.body!)
      if (bodyInfo?.description) {
        lines.push(`<!-- ${binding.body} (${bodyInfo.optional ? 'optional' : 'required'}, body) — ${bodyInfo.description} -->`)
      }
    } else {
      lines.push(`${open}>`)
    }
  }

  // Nested content
  if (hasNested) {
    // ChildTags
    if (hasChildTags) {
      for (const ct of binding.childTags!) {
        const info = resolveFieldInfoByPath(schemaAst, ct.field)
        const typeComment = info?.ast ? needsTypeComment(info.ast) : null
        const annotations: string[] = []
        if (info?.optional) annotations.push('optional')
        if (typeComment) annotations.push(typeComment)
        if (info?.description) annotations.push('— ' + info.description)
        const comment = annotations.length > 0 ? ` <!-- ${annotations.join('. ')} -->` : ''
        lines.push(`<${ct.tag}>${ct.tag}</${ct.tag}>${comment}`)
      }
    }

    // Children (repeated)
    if (hasChildren) {
      for (const child of binding.children!) {
        const childTag = child.tag ?? child.field
        const elemFields = getArrayElementFieldInfos(schemaAst, child.field)

        if (child.attributes && child.attributes.length > 0) {
          // Multi-line child with annotated attributes
          lines.push(`<${childTag}`)
          for (const attr of child.attributes) {
            const info = elemFields.get(attr)
            lines.push(`${attr}="..."${buildAttrAnnotation(info)}`)
          }
          if (child.body) {
            lines.push(`>${child.body}</${childTag}>`)
            const bodyInfo = elemFields.get(child.body)
            if (bodyInfo?.description) {
              lines.push(`<!-- ${child.body} (${bodyInfo.optional ? 'optional' : 'required'}, body) — ${bodyInfo.description} -->`)
            }
          } else {
            lines.push(`/>`)
          }
        } else {
          // Simple child
          if (child.body) {
            lines.push(`<${childTag}>${child.body}</${childTag}>`)
          } else {
            lines.push(`<${childTag} />`)
          }
        }
        lines.push(`<!-- ...more ${childTag}s -->`)
      }
    }

    // ChildRecord
    if (hasChildRecord) {
      const { tag: childTag, keyAttr } = binding.childRecord!
      const recordFieldName = findRecordFieldName(schemaAst)
      const recordDesc = recordFieldName ? fields.get(recordFieldName)?.description : undefined
      lines.push(`<${childTag} ${keyAttr}="...">value</${childTag}>`)
      lines.push(`<${childTag} ${keyAttr}="...">value</${childTag}>`)
      const comment = recordDesc
        ? `<!-- ...more. ${keyAttr} (required, attr) — key. body — ${recordDesc} -->`
        : `<!-- ...more. ${keyAttr} (required, attr) — key. body — value -->`
      lines.push(comment)
    }

    // Body after nested content (rare but possible)
    if (hasBody) {
      const bodyInfo = fields.get(binding.body!)
      if (bodyInfo?.description) {
        lines.push(`<!-- ${binding.body} (body) — ${bodyInfo.description} -->`)
      }
    }

    lines.push(`</${tagName}>`)
  }
}

// =============================================================================
// Annotated Output Builder
// =============================================================================

function buildAnnotatedOutput(
  lines: string[],
  tagName: string,
  outputSchemaAst: AST.AST,
  outputBinding?: XmlBinding<unknown>,
): void {
  const shape = classifyOutputDoc(outputSchemaAst)

  // Void — no returns section
  if (shape.kind === 'void') return

  lines.push('')

  if (shape.kind === 'string') {
    lines.push('Returns: string')
    lines.push(`  <${tagName}>...</${tagName}>`)
    return
  }

  if (shape.kind === 'scalar') {
    lines.push(`Returns: ${shape.type}`)
    lines.push(`  <${tagName}>...</${tagName}>`)
    return
  }

  // If explicit output binding exists
  if (outputBinding && outputBinding.type === 'tag') {
    lines.push('Returns:')
    buildOutputWithBinding(lines, tagName, outputBinding, outputSchemaAst)
    return
  }

  // Schema-derived defaults
  if (shape.kind === 'struct') {
    lines.push('Returns:')
    lines.push(`  <${tagName}>`)
    for (const [, field] of shape.fields) {
      const typeComment = needsTypeComment(field.ast)
      const annotations: string[] = []
      if (field.optional) annotations.push('optional')
      if (typeComment) annotations.push(typeComment)
      const comment = annotations.length > 0 ? ` <!-- ${annotations.join('. ')} -->` : ''
      lines.push(`    <${field.name}>${field.name}</${field.name}>${comment}`)
    }
    lines.push(`  </${tagName}>`)
    return
  }

  if (shape.kind === 'array-struct') {
    lines.push('Returns:')
    lines.push(`  <${tagName}>`)
    // Show one example item with fields as attributes
    const attrStr = [...shape.fields.values()].map(f => ` ${f.name}="..."`).join('')
    lines.push(`    <item${attrStr} />`)
    // Type annotations for non-string fields
    const typeAnnotations: string[] = []
    for (const [, field] of shape.fields) {
      const tc = needsTypeComment(field.ast)
      if (tc) typeAnnotations.push(`${field.name}: ${tc}`)
    }
    const extra = typeAnnotations.length > 0 ? ` ${typeAnnotations.join('. ')}` : ''
    lines.push(`    <!-- ...more items.${extra} -->`)
    lines.push(`  </${tagName}>`)
    return
  }

  if (shape.kind === 'array-scalar') {
    lines.push('Returns:')
    lines.push(`  <${tagName}>`)
    lines.push(`    <item>value</item>`)
    lines.push(`    <!-- ...more items -->`)
    lines.push(`  </${tagName}>`)
    return
  }

  // Unknown
  lines.push('Returns: (unstructured)')
  lines.push(`  <${tagName}>JSON output...</${tagName}>`)
}

function buildOutputWithBinding(
  lines: string[],
  tagName: string,
  binding: Extract<XmlBinding<unknown>, { type: 'tag' }>,
  schemaAst: AST.AST,
): void {
  const fields = getFieldInfos(schemaAst)

  let attrs = ''
  if (binding.attributes) {
    for (const attr of binding.attributes) {
      attrs += ` ${attr}="..."`
    }
  }

  const hasBody = !!binding.body
  const hasChildTags = binding.childTags && binding.childTags.length > 0
  const hasChildren = binding.children && binding.children.length > 0
  const hasItems = !!binding.items

  if (!hasBody && !hasChildTags && !hasChildren && !hasItems) {
    lines.push(`  <${tagName}${attrs} />`)
    return
  }

  if (hasBody && !hasChildTags && !hasChildren && !hasItems) {
    lines.push(`  <${tagName}${attrs}>${binding.body}</${tagName}>`)
    return
  }

  lines.push(`  <${tagName}${attrs}>`)

  if (hasItems) {
    const itemsBinding = binding.items!
    const itemTag = itemsBinding.tag
    let itemAttrs = ''
    if (itemsBinding.attributes) {
      for (const attr of itemsBinding.attributes) {
        itemAttrs += ` ${String(attr)}="..."`
      }
    }

    const schema = unwrapAst(schemaAst)
    const elemFields =
      schema._tag === 'TupleType' && schema.rest.length > 0
        ? getFieldInfos(unwrapAst(schema.rest[0].type))
        : new Map<string, FieldInfo>()

    if (itemsBinding.body) {
      const bodyKey = String(itemsBinding.body)
      const bodyInfo = elemFields.get(bodyKey)
      const bodyText =
        bodyInfo?.description
        ?? (needsTypeComment(bodyInfo?.ast ?? unwrapAst(schemaAst)) ?? (bodyKey ? `${bodyKey} text` : 'value'))
      lines.push(`    <${itemTag}${itemAttrs}>${bodyText}</${itemTag}>`)
    } else {
      lines.push(`    <${itemTag}${itemAttrs} />`)
    }
  }

  if (hasChildTags) {
    for (const ct of binding.childTags!) {
      const info = resolveFieldInfoByPath(schemaAst, ct.field)
      const typeComment = info?.ast ? needsTypeComment(info.ast) : null
      const annotations: string[] = []
      if (info?.optional) annotations.push('optional')
      if (typeComment) annotations.push(typeComment)
      const comment = annotations.length > 0 ? ` <!-- ${annotations.join('. ')} -->` : ''
      lines.push(`    <${ct.tag}>${ct.tag}</${ct.tag}>${comment}`)
    }
  }

  if (hasChildren) {
    for (const child of binding.children!) {
      const childTag = child.tag ?? child.field
      let childAttrs = ''
      if (child.attributes) {
        for (const attr of child.attributes) {
          childAttrs += ` ${attr}="..."`
        }
      }
      if (child.body) {
        lines.push(`    <${childTag}${childAttrs}>${child.body}</${childTag}>`)
      } else {
        lines.push(`    <${childTag}${childAttrs} />`)
      }
      lines.push(`    <!-- ...more ${childTag}s -->`)
    }
  }

  if (hasBody) {
    const info = fields.get(binding.body!)
    if (info?.description) {
      lines.push(`    <!-- ${binding.body} (body) — ${info.description} -->`)
    }
  }

  lines.push(`  </${tagName}>`)
}

/**
 * Generate XML documentation for a group of tools.
 */
export function generateXmlToolGroupDoc(
  groupName: string,
  tools: ReadonlyArray<Tool.Any>,
  implicitDefKeys?: ReadonlyArray<string>,
  defKeyLookup?: ReadonlyMap<Tool.Any, string>,
): string {
  const docs: string[] = []

  for (const tool of tools) {
    if (implicitDefKeys && defKeyLookup) {
      const defKey = defKeyLookup.get(tool)
      if (defKey && implicitDefKeys.includes(defKey)) continue
    }

    const doc = generateXmlToolDoc(tool)
    if (!doc) continue
    docs.push(doc)
  }

  if (docs.length === 0) return ''

  return `### ${groupName}\n\n${docs.join('\n\n')}`
}
