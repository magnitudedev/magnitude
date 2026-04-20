/**
 * Tool Docs Renderer
 *
 * Generates compact tool reference documentation from ToolDefinition schemas.
 * Walks Effect Schema AST directly — no TypeScript package dependency.
 */

import { Schema, AST } from '@effect/schema'
import { Option } from 'effect'
import type { ToolDefinition } from '../tool-definition'

// =============================================================================
// AST Helpers
// =============================================================================

/** Walk AST to find description annotation, handling Transformation/Union wrapping */
function walkForDescription(a: AST.AST, depth = 0): string | undefined {
  if (depth > 5) return undefined
  const d = AST.getDescriptionAnnotation(a)
  if (Option.isSome(d)) return d.value
  if (a._tag === 'Transformation') {
    const from = walkForDescription(a.from, depth + 1)
    if (from) return from
    return walkForDescription(a.to, depth + 1)
  }
  if (a._tag === 'Union') {
    for (const t of a.types) {
      const r = walkForDescription(t, depth + 1)
      if (r) return r
    }
  }
  if (a._tag === 'Refinement') return walkForDescription(a.from, depth + 1)
  return undefined
}

/** Get default value from annotation */
function getDefaultValue(node: AST.Annotated): unknown {
  const annotation = AST.getDefaultAnnotation(node)
  if (Option.isSome(annotation)) {
    const thunk = annotation.value as () => unknown
    return thunk()
  }
  return undefined
}

/** Format a default value for display */
function formatDefaultValue(value: unknown): string {
  return JSON.stringify(value)
}

/** Extract defaults from TypeLiteralTransformation (used by Schema.optionalWith) */
function extractDefaultsFromTransformation(ast: AST.AST): Map<string, unknown> {
  const defaults = new Map<string, unknown>()
  if (ast._tag !== 'Transformation') return defaults
  if (ast.transformation._tag !== 'TypeLiteralTransformation') return defaults

  for (const pst of ast.transformation.propertySignatureTransformations) {
    const propName = String(pst.from)
    try {
      const result = pst.decode(Option.none())
      if (Option.isSome(result)) {
        defaults.set(propName, result.value)
      }
    } catch {
      // decode failed, skip
    }
  }
  return defaults
}

/** Unwrap AST to get the structural type (strip Transformation/Refinement) */
function unwrapAst(ast: AST.AST): AST.AST {
  if (ast._tag === 'Transformation') return unwrapAst(ast.from)
  if (ast._tag === 'Refinement') return unwrapAst(ast.from)
  return ast
}

/** Unwrap to find TypeLiteral - Schema.optionalWith can wrap the whole struct in a Transformation */
function unwrapToTypeLiteral(ast: AST.AST): AST.TypeLiteral | null {
  if (ast._tag === 'TypeLiteral') return ast
  if (ast._tag === 'Transformation') {
    const from = unwrapToTypeLiteral(ast.from)
    if (from) return from
    return unwrapToTypeLiteral(ast.to)
  }
  if (ast._tag === 'Refinement') return unwrapToTypeLiteral(ast.from)
  return null
}

/** Check for identifier annotation (named types like ToolImage) */
function getIdentifier(ast: AST.AST): string | undefined {
  const id = AST.getIdentifierAnnotation(ast)
  return Option.isSome(id) ? id.value : undefined
}

/** Check if a description is auto-generated noise from Effect Schema */
function isNoiseDescription(desc: string | undefined): boolean {
  if (!desc) return true
  // Effect Schema auto-generates descriptions like "a string", "a number"
  return /^a (string|number|boolean|unknown|void|never|object|array)/.test(desc)
}

// =============================================================================
// Type String Conversion
// =============================================================================

/**
 * Convert Effect Schema AST to a type string.
 * depth controls formatting: depth=0 means top-level (multi-line), depth>0 means inline.
 */
function typeToString(ast: AST.AST, isOptional: boolean = false, depth: number = 0): string {
  const unwrapped = unwrapAst(ast)

  // Check for identifier annotation — use the name instead of expanding
  const identifier = getIdentifier(ast) || getIdentifier(unwrapped)
  if (identifier) {
    const nameMap: Record<string, string> = {
      'ToolImage': 'image',
    }
    return nameMap[identifier] ?? identifier
  }

  switch (unwrapped._tag) {
    case 'StringKeyword':
      return 'string'

    case 'NumberKeyword':
      return 'number'

    case 'BooleanKeyword':
      return 'boolean'

    case 'VoidKeyword':
      return 'void'

    case 'NeverKeyword':
      return 'never'

    case 'UnknownKeyword':
      return 'unknown'

    case 'AnyKeyword':
      return 'any'

    case 'UndefinedKeyword':
      return 'undefined'

    case 'Literal':
      return JSON.stringify(unwrapped.literal)

    case 'Union': {
      const nonUndefined = unwrapped.types.filter(t => unwrapAst(t)._tag !== 'UndefinedKeyword')
      if (nonUndefined.length === 1 && isOptional) {
        return typeToString(nonUndefined[0], false, depth)
      }
      const allStringLit = nonUndefined.every(t => {
        const u = unwrapAst(t)
        return u._tag === 'Literal' && typeof u.literal === 'string'
      })
      if (allStringLit) {
        return nonUndefined.map(t => JSON.stringify((unwrapAst(t) as AST.Literal).literal)).join(' | ')
      }
      return nonUndefined.map(t => typeToString(t, false, depth)).join(' | ')
    }

    case 'TypeLiteral': {
      if (depth > 0) {
        const props = unwrapped.propertySignatures.map(p => {
          const opt = p.isOptional ? '?' : ''
          return `${String(p.name)}${opt}: ${typeToString(p.type, p.isOptional, depth + 1)}`
        })
        return `{ ${props.join(', ')} }`
      }
      const props = unwrapped.propertySignatures.map(p => {
        const opt = p.isOptional ? '?' : ''
        return `\t${String(p.name)}${opt}: ${typeToString(p.type, p.isOptional, 1)}`
      })
      return `{\n${props.join(',\n')}\n}`
    }

    case 'TupleType': {
      if (unwrapped.elements.length === 0 && unwrapped.rest.length > 0) {
        return `${typeToString(unwrapped.rest[0].type, false, depth)}[]`
      }
      const elements = unwrapped.elements.map(e => typeToString(e.type, false, depth))
      const rest = unwrapped.rest.length > 0
        ? [`...${typeToString(unwrapped.rest[0].type, false, depth)}[]`]
        : []
      return `[${[...elements, ...rest].join(', ')}]`
    }

    case 'Declaration': {
      const id = getIdentifier(unwrapped)
      if (id === 'Array' || id === 'ReadonlyArray') {
        if (unwrapped.typeParameters.length > 0) {
          return `${typeToString(unwrapped.typeParameters[0], false, depth)}[]`
        }
        return 'unknown[]'
      }
      if (id === 'Record' || id === 'ReadonlyMap') {
        if (unwrapped.typeParameters.length >= 2) {
          return `Record<${typeToString(unwrapped.typeParameters[0], false, depth)}, ${typeToString(unwrapped.typeParameters[1], false, depth)}>`
        }
        return 'Record<string, unknown>'
      }
      if (id) {
        const typeArgs = unwrapped.typeParameters.map(p => typeToString(p, false, depth))
        return typeArgs.length > 0 ? `${id}<${typeArgs.join(', ')}>` : id
      }
      return 'unknown'
    }

    case 'Enums': {
      return unwrapped.enums.map(([_, v]) => JSON.stringify(v)).join(' | ')
    }

    case 'Suspend': {
      return typeToString(unwrapped.f(), isOptional, depth)
    }

    default:
      return 'unknown'
  }
}

// =============================================================================
// Comment Building
// =============================================================================

/** Build comment string from description and default value */
function buildComment(description: string | undefined, defaultValue: unknown): string {
  const cleanDesc = isNoiseDescription(description) ? undefined : description
  
  if (!cleanDesc && defaultValue === undefined) return ''
  
  const parts: string[] = []
  if (cleanDesc) parts.push(cleanDesc)
  if (defaultValue !== undefined) {
    parts.push(`(default: ${formatDefaultValue(defaultValue)})`)
  }
  
  return ` // ${parts.join(' ')}`
}

// =============================================================================
// Parameter & Return Type Extraction
// =============================================================================

interface ParamInfo {
  name: string
  optional: boolean
  type: string
  description: string | undefined
  defaultValue: unknown
}

function getParams(tool: ToolDefinition): ParamInfo[] {
  // Extract defaults from the top-level Transformation (if any)
  const transformDefaults = extractDefaultsFromTransformation(tool.inputSchema.ast)
  
  const inputAst = unwrapToTypeLiteral(tool.inputSchema.ast)
  if (!inputAst) return []

  // For optionalWith, descriptions live on the `from` side's Union members.
  // Build a lookup: propName -> description from the from-side TypeLiteral.
  const fromDescriptions = new Map<string, string>()
  const topAst = tool.inputSchema.ast
  if (topAst._tag === 'Transformation' && topAst.from._tag === 'TypeLiteral') {
    for (const p of topAst.from.propertySignatures) {
      const desc = walkForDescription(p.type)
      if (desc && !isNoiseDescription(desc)) {
        fromDescriptions.set(String(p.name), desc)
      }
    }
  }

  return inputAst.propertySignatures.map(p => {
    const name = String(p.name)
    const optional = p.isOptional
    const type = typeToString(p.type, optional, 1)
    const description = walkForDescription(p.type) || walkForDescription(p as AST.Annotated) || fromDescriptions.get(name)
    const defaultValue = getDefaultValue(p) || transformDefaults.get(name)

    return { name, optional, type, description, defaultValue }
  })
}

function getReturnType(tool: ToolDefinition): string {
  return typeToString(tool.outputSchema.ast, false, 0)
}

// =============================================================================
// Rendering
// =============================================================================

/**
 * Render a single tool in the docs format.
 */
function renderOneTool(tool: ToolDefinition): string {
  const params = getParams(tool)
  const returnType = getReturnType(tool)
  const lines: string[] = []

  // Heading
  lines.push(`### ${tool.name}`)

  // Description
  if (tool.description) {
    lines.push(tool.description)
  }

  // Blank line
  lines.push('')

  // Signature: toolname({
  // Each param on its own line with comma BEFORE the comment
  const paramLines = params.map(p => {
    const opt = p.optional ? '?' : ''
    const comment = buildComment(p.description, p.defaultValue)
    return `\t${p.name}${opt}: ${p.type}${comment}`
  })

  lines.push(`${tool.name}({`)
  lines.push(paramLines.join('\n'))
  lines.push(`}) -> ${returnType}`)

  return lines.join('\n')
}

// =============================================================================
// Public API
// =============================================================================

/**
 * Render tool documentation for a set of tools.
 */
export function renderToolDocs(tools: readonly ToolDefinition[]): string {
  return tools.map(renderOneTool).join('\n\n')
}
