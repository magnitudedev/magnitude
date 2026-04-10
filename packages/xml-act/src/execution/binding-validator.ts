/**
 * BindingValidator — validates XmlTagBinding against schema AST at registration time.
 *
 * Catches tool definition bugs eagerly:
 * - Array/record fields declared as attributes (must use children/childTags)
 * - Non-string fields declared as body (body is always text)
 * - Binding references fields that don't exist in the schema
 * - Child binding attributes that aren't scalar
 */

import { AST } from '@effect/schema'
import type { XmlTagBinding, XmlChildBinding } from '../types'

// =============================================================================
// AST Helpers
// =============================================================================

function unwrapAst(ast: AST.AST): AST.AST {
  if (ast._tag === 'Transformation') return unwrapAst(ast.from)
  if (ast._tag === 'Refinement') return unwrapAst(ast.from)
  return ast
}

type ScalarType = 'string' | 'number' | 'boolean'

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
      // Union of literals (e.g., Schema.Literal('a', 'b', 'c')) — resolve if all members share one scalar type
      const memberTypes = unwrapped.types
        .map(t => getScalarType(t))
        .filter((t): t is ScalarType => t !== undefined)
      if (memberTypes.length === 0) return undefined
      const first = memberTypes[0]
      if (memberTypes.every(t => t === first)) return first
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

/** Resolve a dotted field path like 'options.type' through the schema AST. */
function resolveFieldPath(schemaAst: AST.AST, path: string): AST.PropertySignature | undefined {
  const segments = path.split('.')
  let currentAst = schemaAst
  for (let i = 0; i < segments.length; i++) {
    const prop = findProperty(currentAst, segments[i])
    if (!prop) return undefined
    if (i < segments.length - 1) {
      // Intermediate segment — drill into the struct
      currentAst = unwrapAst(prop.type)
    } else {
      return prop
    }
  }
  return undefined
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

/**
 * Get the element struct AST for an array property.
 * Handles both required and optional arrays (Union with UndefinedKeyword).
 * Returns undefined if not an array of structs.
 */
function getArrayElementAst(prop: AST.PropertySignature): AST.AST | undefined {
  const unwrapped = unwrapAst(prop.type)

  // Direct array
  if (unwrapped._tag === 'TupleType' && unwrapped.rest.length > 0) {
    const elemAst = unwrapAst(unwrapped.rest[0].type)
    if (elemAst._tag === 'TypeLiteral') return elemAst
  }

  // Optional array: Union(TupleType, UndefinedKeyword)
  if (unwrapped._tag === 'Union') {
    const nonUndefined = unwrapped.types.filter(t => unwrapAst(t)._tag !== 'UndefinedKeyword')
    if (nonUndefined.length === 1) {
      const inner = unwrapAst(nonUndefined[0])
      if (inner._tag === 'TupleType' && inner.rest.length > 0) {
        const elemAst = unwrapAst(inner.rest[0].type)
        if (elemAst._tag === 'TypeLiteral') return elemAst
      }
    }
  }

  return undefined
}

// =============================================================================
// Public API
// =============================================================================

export interface AttributeSchema {
  readonly type: ScalarType
  readonly required: boolean
}

export interface ChildTagSchema {
  readonly attributes: ReadonlyMap<string, AttributeSchema>
  readonly acceptsBody: boolean
}

export interface TagSchema {
  /** Valid attributes with their types */
  readonly attributes: ReadonlyMap<string, AttributeSchema>
  /** Whether this tag accepts body content */
  readonly acceptsBody: boolean
  /** Canonical body field path if this tool binds body content */
  readonly bodyField?: string
  /** Valid child tag names → child schema */
  readonly children: ReadonlyMap<string, ChildTagSchema>
}

/**
 * Validate a binding against a schema AST and return the derived TagSchema.
 * Throws on binding errors (developer bugs).
 */
export function validateBinding(
  tagName: string,
  binding: XmlTagBinding,
  schemaAst: AST.AST,
): TagSchema {
  const attributes = new Map<string, AttributeSchema>()
  const children = new Map<string, ChildTagSchema>()

  // --- Validate duplicate input field mappings across sections ---
  const seenInputFields = new Map<string, string>()
  const addInputField = (field: string, section: string) => {
    const existing = seenInputFields.get(field)
    if (existing) {
      throw new Error(
        `Binding error on <${tagName}>: field '${field}' is mapped multiple times across input sections (${existing} and ${section})`
      )
    }
    seenInputFields.set(field, section)
  }

  if (binding.attributes) {
    for (const attrSpec of binding.attributes) addInputField(attrSpec.field, 'attributes')
  }
  if (binding.body) addInputField(binding.body, 'body')
  if (binding.childTags) {
    for (const ct of binding.childTags) addInputField(ct.field, 'childTags')
  }
  if (binding.children) {
    for (const child of binding.children) addInputField(child.field, 'children')
  }
  if (binding.childRecord) addInputField(binding.childRecord.field, 'childRecord')

  // --- Validate no XML name collision between explicitly-declared attributes and child tags ---
  if (binding.attributes && binding.childTags) {
    const attrNames = new Set(binding.attributes.map(a => a.attr))
    for (const ct of binding.childTags) {
      if (attrNames.has(ct.tag)) {
        throw new Error(
          `Binding error on <${tagName}>: '${ct.tag}' is used as both an attribute name and a child tag name`
        )
      }
    }
  }

  // --- Validate attribute fields ---
  if (binding.attributes) {
    for (const attrSpec of binding.attributes) {
      const prop = resolveFieldPath(schemaAst, attrSpec.field)
      if (!prop) {
        throw new Error(
          `Binding error on <${tagName}>: attribute field '${attrSpec.field}' does not exist in the schema`
        )
      }

      const scalarType = getPropertyScalarType(prop)
      if (!scalarType) {
        throw new Error(
          `Binding error on <${tagName}>: attribute field '${attrSpec.field}' has type '${describeType(prop.type)}' — ` +
          `attributes must be scalar (string, number, or boolean)`
        )
      }

      attributes.set(attrSpec.attr, {
        type: scalarType,
        required: !prop.isOptional,
      })
    }
  }

  // --- Validate body field ---
  // A tool accepts body content if it has a body binding OR any children bindings
  // (children naturally have whitespace body text between child elements)
  const hasChildren = !!(binding.children?.length || binding.childTags?.length || binding.childRecord)
  const acceptsBody = binding.body !== undefined || hasChildren
  if (binding.body) {
    const prop = resolveFieldPath(schemaAst, binding.body)
    if (!prop) {
      throw new Error(
        `Binding error on <${tagName}>: body field '${binding.body}' does not exist in the schema`
      )
    }
    const unwrapped = unwrapAst(prop.type)
    const isString = unwrapped._tag === 'StringKeyword' ||
      (unwrapped._tag === 'Union' && unwrapped.types.some(t => unwrapAst(t)._tag === 'StringKeyword'))
    if (!isString) {
      throw new Error(
        `Binding error on <${tagName}>: body field '${binding.body}' has type '${describeType(prop.type)}' — ` +
        `body must be a string field`
      )
    }
  }

  // --- Validate children bindings ---
  if (binding.children) {
    for (const childBinding of binding.children) {
      const childSchema = validateChildBinding(tagName, childBinding, schemaAst)
      const childTag = childBinding.tag ?? childBinding.field
      children.set(childTag, childSchema)
    }
  }

  // --- Validate childTags (scalar child elements) ---
  if (binding.childTags) {
    for (const ct of binding.childTags) {
      const prop = resolveFieldPath(schemaAst, ct.field)
      if (!prop) {
        throw new Error(
          `Binding error on <${tagName}>: childTag field '${ct.field}' does not exist in the schema`
        )
      }
      const xmlTag = ct.tag
      // childTags map to scalar string fields (the body of the child element)
      children.set(xmlTag, {
        attributes: new Map(),
        acceptsBody: true,
      })
    }
  }

  // --- Validate childRecord ---
  if (binding.childRecord) {
    const { field, tag: childTag, keyAttr } = binding.childRecord
    const prop = findProperty(schemaAst, field)
    if (!prop) {
      throw new Error(
        `Binding error on <${tagName}>: childRecord field '${field}' does not exist in the schema`
      )
    }
    // childRecord children have a key attribute (string) and body text
    const keySchema: AttributeSchema = { type: 'string', required: true }
    children.set(childTag, {
      attributes: new Map([[keyAttr, keySchema]]),
      acceptsBody: true,
    })
  }

  // Tool tags always accept an optional `id` attribute for stable toolCall identity.
  if (!attributes.has('id')) {
    attributes.set('id', { type: 'string', required: false })
  }

  return { attributes, acceptsBody, bodyField: binding.body, children }
}

function validateChildBinding(
  parentTagName: string,
  childBinding: XmlChildBinding,
  parentSchemaAst: AST.AST,
): ChildTagSchema {
  const childTag = childBinding.tag ?? childBinding.field

  // The field must exist and be an array
  const prop = findProperty(parentSchemaAst, childBinding.field)
  if (!prop) {
    throw new Error(
      `Binding error on <${parentTagName}>: children field '${childBinding.field}' does not exist in the schema`
    )
  }

  const elemAst = getArrayElementAst(prop)
  if (!elemAst) {
    throw new Error(
      `Binding error on <${parentTagName}>: children field '${childBinding.field}' has type '${describeType(prop.type)}' — ` +
      `children fields must be arrays of structs`
    )
  }

  const attributes = new Map<string, AttributeSchema>()

  // Validate child attributes
  if (childBinding.attributes) {
    for (const attrSpec of childBinding.attributes) {
      const childProp = findProperty(elemAst, attrSpec.field)
      if (!childProp) {
        throw new Error(
          `Binding error on <${parentTagName}>: child <${childTag}> attribute field '${attrSpec.field}' ` +
          `does not exist in the element schema`
        )
      }

      const scalarType = getPropertyScalarType(childProp)
      if (!scalarType) {
        throw new Error(
          `Binding error on <${parentTagName}>: child <${childTag}> attribute field '${attrSpec.field}' ` +
          `has type '${describeType(childProp.type)}' — attributes must be scalar`
        )
      }

      attributes.set(attrSpec.attr, {
        type: scalarType,
        required: !childProp.isOptional,
      })
    }
  }

  // Validate child body
  const acceptsBody = childBinding.body !== undefined
  if (childBinding.body) {
    const bodyProp = findProperty(elemAst, childBinding.body)
    if (!bodyProp) {
      throw new Error(
        `Binding error on <${parentTagName}>: child <${childTag}> body field '${childBinding.body}' ` +
        `does not exist in the element schema`
      )
    }
    const unwrapped = unwrapAst(bodyProp.type)
    const isString = unwrapped._tag === 'StringKeyword' ||
      (unwrapped._tag === 'Union' && unwrapped.types.some(t => unwrapAst(t)._tag === 'StringKeyword'))
    if (!isString) {
      throw new Error(
        `Binding error on <${parentTagName}>: child <${childTag}> body field '${childBinding.body}' ` +
        `has type '${describeType(bodyProp.type)}' — body must be a string field`
      )
    }
  }

  return { attributes, acceptsBody }
}
