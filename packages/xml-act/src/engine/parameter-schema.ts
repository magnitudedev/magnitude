/**
 * Parameter Schema Derivation — derives parameter schema from tool input schema AST.
 *
 * Walks the Effect Schema AST to determine:
 * - What parameters a tool accepts
 * - Each parameter's type (scalar or json)
 * - Whether each parameter is required
 *
 * Top-level scalar fields become individual parameters by name.
 * Top-level complex fields (nested structs, arrays) become a single json parameter.
 * No dotted-path recursion into nested structs.
 *
 * This replaces the old binding-validator approach. No manual binding needed —
 * the schema has all the information.
 */

import { AST } from '@effect/schema'

// =============================================================================
// AST Helpers
// =============================================================================

function unwrapAst(ast: AST.AST): AST.AST {
  if (ast._tag === 'Transformation') return unwrapAst(ast.from)
  if (ast._tag === 'Refinement') return unwrapAst(ast.from)
  return ast
}

export type ScalarType = 'string' | 'number' | 'boolean' | { readonly _tag: 'enum'; readonly values: readonly string[] }

function getScalarType(ast: AST.AST): ScalarType | undefined {
  const unwrapped = unwrapAst(ast)
  switch (unwrapped._tag) {
    case 'StringKeyword': return 'string'
    case 'NumberKeyword': return 'number'
    case 'BooleanKeyword': return 'boolean'
    case 'Literal': {
      const lit = unwrapped.literal
      if (typeof lit === 'string') return 'string'
      if (typeof lit === 'number') return 'number'
      if (typeof lit === 'boolean') return 'boolean'
      return undefined
    }
    case 'Union': {
      // Check if all members are string literals → enum
      const allStringLiterals = unwrapped.types.every(t => {
        const u = unwrapAst(t)
        return u._tag === 'Literal' && typeof u.literal === 'string'
      })
      if (allStringLiterals && unwrapped.types.length > 0) {
        const values = unwrapped.types.map(t => (unwrapAst(t) as AST.Literal).literal as string)
        return { _tag: 'enum', values }
      }
      // Union of literals sharing one scalar type
      const memberTypes = unwrapped.types
        .map(t => getScalarType(t))
        .filter((t): t is ScalarType => t !== undefined)
      if (memberTypes.length === 0) return undefined
      const first = memberTypes[0]
      if (typeof first === 'string' && memberTypes.every(t => t === first)) return first
      return undefined
    }
    default: return undefined
  }
}

function describeType(ast: AST.AST): string {
  const unwrapped = unwrapAst(ast)
  switch (unwrapped._tag) {
    case 'StringKeyword': return 'string'
    case 'NumberKeyword': return 'number'
    case 'BooleanKeyword': return 'boolean'
    case 'BigIntKeyword': return 'bigint'
    case 'TupleType': return 'array'
    case 'TypeLiteral': return 'object'
    case 'Union': {
      const types = unwrapped.types
        .map(t => describeType(t))
        .filter(t => t !== 'undefined')
      return types.join(' | ')
    }
    case 'UndefinedKeyword': return 'undefined'
    default: return unwrapped._tag
  }
}

/** Get property signatures from a struct schema AST */
function getPropertySignatures(schemaAst: AST.AST): ReadonlyArray<AST.PropertySignature> {
  const ast = unwrapAst(schemaAst)
  if (ast._tag !== 'TypeLiteral') return []
  return ast.propertySignatures
}

/** Look up a property by name */
function findProperty(schemaAst: AST.AST, name: string): AST.PropertySignature | undefined {
  return getPropertySignatures(schemaAst).find(p => String(p.name) === name)
}

/**
 * Get the scalar type for a property, handling optional (Union with UndefinedKeyword).
 * Returns undefined if the property is not scalar.
 */
function getPropertyScalarType(prop: AST.PropertySignature): ScalarType | undefined {
  const unwrapped = unwrapAst(prop.type)

  // Direct scalar
  const direct = getScalarType(unwrapped)
  if (direct) return direct

  // Optional scalar: Union with UndefinedKeyword
  if (unwrapped._tag === 'Union') {
    const nonUndefined = unwrapped.types.filter(t => unwrapAst(t)._tag !== 'UndefinedKeyword')
    if (nonUndefined.length === 1) {
      return getScalarType(nonUndefined[0])
    }
  }

  return undefined
}

/** Check if a property is complex (object or array) */
function isComplexType(ast: AST.AST): boolean {
  const unwrapped = unwrapAst(ast)
  
  // Handle optional: Union with UndefinedKeyword
  let typeToCheck = unwrapped
  if (unwrapped._tag === 'Union') {
    const nonUndefined = unwrapped.types.filter(t => unwrapAst(t)._tag !== 'UndefinedKeyword')
    if (nonUndefined.length === 1) {
      typeToCheck = unwrapAst(nonUndefined[0])
    }
  }

  return typeToCheck._tag === 'TypeLiteral' || typeToCheck._tag === 'TupleType' ||
    (typeToCheck._tag === 'Union' && typeToCheck.types.some(t => {
      const u = unwrapAst(t)
      return u._tag === 'TypeLiteral' || u._tag === 'TupleType'
    }))
}

// =============================================================================
// Public Types
// =============================================================================

export interface ParameterSchema {
  /** Parameter name (top-level field name only, no dotted paths) */
  readonly name: string
  /** Parameter type — scalar types are raw text, json types are parsed */
  readonly type: ScalarType | 'json_object' | 'json_array'
  /** Whether this parameter must be provided */
  readonly required: boolean
}

export interface ToolSchema {
  /** Valid parameters with their types */
  readonly parameters: ReadonlyMap<string, ParameterSchema>
  /** Whether this tool can be self-closing (no required parameters) */
  readonly selfClosing: boolean
}

// =============================================================================
// Derivation
// =============================================================================

/**
 * Derive parameter schema from a tool's input schema AST.
 * Walks top-level properties to build the parameter map.
 *
 * Rules:
 * - Each top-level property becomes a parameter with name = property name
 * - Scalar types (string, number, boolean, enum) → type is the scalar type
 * - Complex types (object, array, nested struct) → type is 'json_object' or 'json_array'
 * - Required/optional derived from schema
 * - No dotted-path recursion into nested structs
 */
export function deriveParameters(schemaAst: AST.AST, prefix?: string): ToolSchema {
  const parameters = new Map<string, ParameterSchema>()

  function walkProperties(ast: AST.AST, currentPrefix: string): void {
    const props = getPropertySignatures(ast)
    for (const prop of props) {
      const name = String(prop.name)
      const fullName = currentPrefix ? `${currentPrefix}.${name}` : name

      const scalarType = getPropertyScalarType(prop)

      if (scalarType) {
        // Scalar parameter
        parameters.set(fullName, {
          name: fullName,
          type: scalarType,
          required: !prop.isOptional,
        })
      } else if (isComplexType(prop.type)) {
        // Check if this is a nested struct that should be flattened into dotted params
        const unwrapped = unwrapAst(prop.type)
        // Handle optional
        let typeToInspect = unwrapped
        if (unwrapped._tag === 'Union') {
          const nonUndefined = unwrapped.types.filter(t => unwrapAst(t)._tag !== 'UndefinedKeyword')
          if (nonUndefined.length === 1) {
            typeToInspect = unwrapAst(nonUndefined[0])
          }
        }

        if (typeToInspect._tag === 'TypeLiteral') {
          parameters.set(fullName, {
            name: fullName,
            type: 'json_object',
            required: !prop.isOptional,
          })
        } else {
          // TupleType or other complex type — array
          parameters.set(fullName, {
            name: fullName,
            type: 'json_array',
            required: !prop.isOptional,
          })
        }
      } else {
        // Unknown type — treat as object fallback
        parameters.set(fullName, {
          name: fullName,
          type: 'json_object',
          required: !prop.isOptional,
        })
      }
    }
  }

  walkProperties(schemaAst, prefix ?? '')

  const hasRequired = [...parameters.values()].some(p => p.required)

  return {
    parameters,
    selfClosing: !hasRequired,
  }
}
