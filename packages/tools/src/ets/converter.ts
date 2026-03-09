/**
 * Effect Schema to TypeScript AST Converter
 *
 * Converts Effect Schema definitions to TypeScript AST nodes.
 * Similar to zts but for Effect Schema instead of Zod.
 */

import ts from "typescript"
import { Schema } from "@effect/schema"
import { AST } from "@effect/schema"
import { Option } from "effect"

// =============================================================================
// Helper Functions (merged from sage ETS)
// =============================================================================

/**
 * Get the identifier/typeName annotation from an AST node
 */
function getIdentifier(ast: AST.AST): string | undefined {
  const id = AST.getIdentifierAnnotation(ast)
  return Option.isSome(id) ? id.value : undefined
}

/**
 * Get the description annotation from an annotated node
 * Works with both AST nodes and PropertySignature
 */
function getDescription(node: AST.Annotated): string | undefined {
  const desc = AST.getDescriptionAnnotation(node)
  return Option.isSome(desc) ? desc.value : undefined
}

/**
 * Get the default value annotation from an annotated node (if present)
 * Works with both AST nodes and PropertySignature
 */
function getDefaultValue(node: AST.Annotated): unknown | undefined {
  const defaultAnnotation = AST.getDefaultAnnotation(node)
  if (Option.isSome(defaultAnnotation)) {
    // Default annotation is a thunk that returns the default value
    const thunk = defaultAnnotation.value as () => unknown
    return thunk()
  }
  return undefined
}

/**
 * Check if a property signature is optional
 */
function isPropertyOptional(prop: AST.PropertySignature): boolean {
  return prop.isOptional
}

/**
 * Format a default value for display in JSDoc comments
 */
function formatDefaultValue(value: unknown): string {
  return JSON.stringify(value)
}

// =============================================================================
// AST Conversion
// =============================================================================

/**
 * Unwrap optional union types by removing the undefined variant.
 * For `string | undefined` returns `string`.
 * For `string | number | undefined` returns `string | number` (as union).
 */
export function unwrapOptionalUnion(ast: AST.AST): AST.AST {
  if (ast._tag !== "Union") return ast

  const union = ast as AST.Union
  const nonUndefined = union.types.filter(t => t._tag !== "UndefinedKeyword")

  if (nonUndefined.length === union.types.length) {
    // No undefined was present
    return ast
  }

  if (nonUndefined.length === 1) {
    return nonUndefined[0]
  }

  // Multiple non-undefined types remain
  return { ...union, types: nonUndefined } as unknown as AST.Union
}

/**
 * Known entity info for structural matching
 */
export interface KnownEntityInfo {
  schema: Schema.Schema.All
  fields: Set<string>
  isUnion: boolean
  /** For unions: fields of each member for structural matching */
  memberFields?: Set<string>[]
}

/**
 * Options for AST generation
 */
export interface AstGenOptions {
  /**
   * Mode determines how to handle schemas with identifier annotation:
   * - 'expression': Return type reference (for use in properties/params)
   * - 'definition': Return structure (for type alias body)
   */
  mode: "expression" | "definition"
  /** Track referenced entity types */
  referencedEntities?: Map<string, Schema.Schema.All>
  /** Known entity schemas for structural matching (read-only input) */
  knownEntities?: Map<string, KnownEntityInfo>
  /** Track if Resources are used */
  hasResources?: { value: boolean }
  /** Whether this is an input schema (affects Resource handling) */
  isInput?: boolean
  /** Stack for cycle detection */
  processingStack?: Set<AST.AST>
  /** Default values extracted from Transformation */
  defaultValues?: Map<string, unknown>
  /** Descriptions extracted from Transformation's "to" side */
  descriptions?: Map<string, string>
}

/**
 * Metadata extracted from a TypeLiteralTransformation for input schemas.
 * Contains defaults (from decode functions) and descriptions (from "to" side annotations).
 */
interface TransformationMetadata {
  defaults: Map<string, unknown>
  descriptions: Map<string, string>
}

/**
 * Extract metadata from a TypeLiteralTransformation.
 * - Defaults are captured in decode function closures - extracted by calling decode(None)
 * - Descriptions are on the "to" side property signatures
 */
function extractTransformationMetadata(ast: AST.AST): TransformationMetadata {
  const defaults = new Map<string, unknown>()
  const descriptions = new Map<string, string>()

  if (ast._tag !== "Transformation") return { defaults, descriptions }

  const transformation = ast as AST.Transformation
  if (transformation.transformation._tag !== "TypeLiteralTransformation") return { defaults, descriptions }

  // Extract defaults from PropertySignatureTransformations
  for (const pst of transformation.transformation.propertySignatureTransformations) {
    const propName = String(pst.from)
    try {
      // Call decode with None - if there's a default, it returns Some(defaultValue)
      const result = pst.decode(Option.none())
      if (Option.isSome(result)) {
        defaults.set(propName, result.value)
      }
    } catch {
      // decode failed, skip
    }
  }

  // Extract descriptions from "to" side
  const toAst = transformation.to
  if (toAst._tag === "TypeLiteral") {
    for (const prop of toAst.propertySignatures) {
      const propName = String(prop.name)
      const desc = getDescription(prop)
      if (desc) {
        descriptions.set(propName, desc)
      }
    }
  }

  return { defaults, descriptions }
}

// Note: getIdentifier and getDescription are defined above in the helpers section

/**
 * Core conversion function - converts Effect Schema AST to TypeScript AST
 */
function getSchemaStructure(
  ast: AST.AST,
  referencedEntities: Map<string, Schema.Schema.All> | undefined,
  knownEntities: Map<string, KnownEntityInfo>,
  hasResources: { value: boolean },
  isInput: boolean,
  processingStack: Set<AST.AST>,
  defaultValues: Map<string, unknown> = new Map(),
  descriptions: Map<string, string> = new Map()
): ts.TypeNode {
  // Handle different AST types
  switch (ast._tag) {
    case "StringKeyword":
      return ts.factory.createKeywordTypeNode(ts.SyntaxKind.StringKeyword)

    case "NumberKeyword":
      return ts.factory.createKeywordTypeNode(ts.SyntaxKind.NumberKeyword)

    case "BooleanKeyword":
      return ts.factory.createKeywordTypeNode(ts.SyntaxKind.BooleanKeyword)

    case "BigIntKeyword":
      return ts.factory.createKeywordTypeNode(ts.SyntaxKind.BigIntKeyword)

    case "SymbolKeyword":
      return ts.factory.createKeywordTypeNode(ts.SyntaxKind.SymbolKeyword)

    case "VoidKeyword":
      return ts.factory.createKeywordTypeNode(ts.SyntaxKind.VoidKeyword)

    case "NeverKeyword":
      return ts.factory.createKeywordTypeNode(ts.SyntaxKind.NeverKeyword)

    case "UnknownKeyword":
      return ts.factory.createKeywordTypeNode(ts.SyntaxKind.UnknownKeyword)

    case "AnyKeyword":
      return ts.factory.createKeywordTypeNode(ts.SyntaxKind.AnyKeyword)

    case "UndefinedKeyword":
      return ts.factory.createKeywordTypeNode(ts.SyntaxKind.UndefinedKeyword)

    case "ObjectKeyword":
      return ts.factory.createKeywordTypeNode(ts.SyntaxKind.ObjectKeyword)

    case "Literal": {
      const value = ast.literal
      if (typeof value === "string") {
        return ts.factory.createLiteralTypeNode(ts.factory.createStringLiteral(value))
      } else if (typeof value === "number") {
        return ts.factory.createLiteralTypeNode(ts.factory.createNumericLiteral(value))
      } else if (typeof value === "boolean") {
        return ts.factory.createLiteralTypeNode(
          value ? ts.factory.createTrue() : ts.factory.createFalse()
        )
      } else if (typeof value === "bigint") {
        return ts.factory.createLiteralTypeNode(
          ts.factory.createBigIntLiteral(value.toString())
        )
      } else if (value === null) {
        return ts.factory.createLiteralTypeNode(ts.factory.createNull())
      }
      return ts.factory.createKeywordTypeNode(ts.SyntaxKind.UnknownKeyword)
    }

    case "UniqueSymbol":
      return ts.factory.createKeywordTypeNode(ts.SyntaxKind.SymbolKeyword)

    case "TupleType": {
      // Check if this is actually an Array (no elements, just rest)
      if (ast.elements.length === 0 && ast.rest.length > 0) {
        const elementType = getTypeNode(ast.rest[0].type, {
          mode: "expression",
          referencedEntities,
          knownEntities,
          hasResources,
          isInput,
          processingStack
        })
        return ts.factory.createArrayTypeNode(elementType)
      }

      // Otherwise it's a real tuple
      const elements = ast.elements.map((element) =>
        getTypeNode(element.type, {
          mode: "expression",
          referencedEntities,
          knownEntities,
          hasResources,
          isInput,
          processingStack
        })
      )
      // Handle rest element if present
      if (ast.rest.length > 0) {
        const restType = getTypeNode(ast.rest[0].type, {
          mode: "expression",
          referencedEntities,
          knownEntities,
          hasResources,
          isInput,
          processingStack
        })
        elements.push(ts.factory.createRestTypeNode(ts.factory.createArrayTypeNode(restType)))
      }
      return ts.factory.createTupleTypeNode(elements)
    }

    case "TypeLiteral": {
      const properties: ts.TypeElement[] = []

      for (const prop of ast.propertySignatures) {
        const propName = String(prop.name)

        // Get default value - check both AST annotation and passed-in defaults from Transformation
        let defaultValue = getDefaultValue(prop)
        if (defaultValue === undefined && defaultValues.has(propName)) {
          defaultValue = defaultValues.get(propName)
        }
        const isOptional = isPropertyOptional(prop) || defaultValue !== undefined

        // If property is optional, unwrap the union to remove redundant | undefined
        const typeAst = isOptional ? unwrapOptionalUnion(prop.type) : prop.type

        const propType = getTypeNode(typeAst, {
          mode: "expression",
          referencedEntities,
          knownEntities,
          hasResources,
          isInput,
          processingStack
        })

        const propSignature = ts.factory.createPropertySignature(
          undefined,
          propName,
          isOptional ? ts.factory.createToken(ts.SyntaxKind.QuestionToken) : undefined,
          propType
        )

        // Build JSDoc comment with description and default value
        // Check prop annotations first, then fall back to passed-in descriptions from Transformation
        let description = getDescription(prop)
        if (!description && descriptions.has(propName)) {
          description = descriptions.get(propName)
        }

        if (description || defaultValue !== undefined) {
          const parts: string[] = []
          if (description) parts.push(description)
          if (defaultValue !== undefined) {
            parts.push(`(default: ${formatDefaultValue(defaultValue)})`)
          }
          const commentText = parts.join(" ")
          ts.addSyntheticLeadingComment(
            propSignature,
            ts.SyntaxKind.MultiLineCommentTrivia,
            `* ${commentText} `,
            true
          )
        }

        properties.push(propSignature)
      }

      // Handle index signatures
      for (const indexSig of ast.indexSignatures) {
        const keyType = getTypeNode(indexSig.parameter, {
          mode: "expression",
          referencedEntities,
          knownEntities,
          hasResources,
          isInput,
          processingStack
        })
        const valueType = getTypeNode(indexSig.type, {
          mode: "expression",
          referencedEntities,
          knownEntities,
          hasResources,
          isInput,
          processingStack
        })

        properties.push(
          ts.factory.createIndexSignature(
            undefined,
            [ts.factory.createParameterDeclaration(undefined, undefined, "key", undefined, keyType)],
            valueType
          )
        )
      }

      return ts.factory.createTypeLiteralNode(properties)
    }

    case "Union": {
      const types = ast.types.map((type) =>
        getTypeNode(type, {
          mode: "expression",
          referencedEntities,
          knownEntities,
          hasResources,
          isInput,
          processingStack
        })
      )
      return ts.factory.createUnionTypeNode(types)
    }

    case "Enums": {
      const types = ast.enums.map(([_, value]) => {
        if (typeof value === "string") {
          return ts.factory.createLiteralTypeNode(ts.factory.createStringLiteral(value))
        } else {
          return ts.factory.createLiteralTypeNode(ts.factory.createNumericLiteral(value))
        }
      })
      return ts.factory.createUnionTypeNode(types)
    }

    case "Refinement":
      // Refinements don't change the type, just add validation
      return getSchemaStructure(ast.from, referencedEntities, knownEntities, hasResources, isInput, processingStack)

    case "Transformation": {
      // For input schemas, use the "from" type but extract defaults and descriptions from transformation
      // For output schemas, use the "to" type
      if (isInput) {
        const metadata = extractTransformationMetadata(ast)
        return getSchemaStructure(ast.from, referencedEntities, knownEntities, hasResources, isInput, processingStack, metadata.defaults, metadata.descriptions)
      }
      return getSchemaStructure(ast.to, referencedEntities, knownEntities, hasResources, isInput, processingStack, defaultValues, descriptions)
    }

    case "Declaration": {
      // Handle Declaration AST nodes (Array, Map, Set, ReadonlyArray, etc.)
      const identifier = getIdentifier(ast)

      // Check for common built-in declarations
      if (identifier === "Array" || identifier === "ReadonlyArray") {
        if (ast.typeParameters.length > 0) {
          const itemType = getTypeNode(ast.typeParameters[0], {
            mode: "expression",
            referencedEntities,
            knownEntities,
            hasResources,
            isInput,
            processingStack
          })
          return ts.factory.createArrayTypeNode(itemType)
        }
        return ts.factory.createArrayTypeNode(
          ts.factory.createKeywordTypeNode(ts.SyntaxKind.UnknownKeyword)
        )
      }

      if (identifier === "Map" || identifier === "ReadonlyMap") {
        const typeArgs = ast.typeParameters.map((p: AST.AST) =>
          getTypeNode(p, { mode: "expression", referencedEntities, knownEntities, hasResources, isInput, processingStack })
        )
        return ts.factory.createTypeReferenceNode("Map", typeArgs.length > 0 ? typeArgs : undefined)
      }

      if (identifier === "Set" || identifier === "ReadonlySet") {
        const typeArgs = ast.typeParameters.map((p: AST.AST) =>
          getTypeNode(p, { mode: "expression", referencedEntities, knownEntities, hasResources, isInput, processingStack })
        )
        return ts.factory.createTypeReferenceNode("Set", typeArgs.length > 0 ? typeArgs : undefined)
      }

      if (identifier === "Record") {
        const typeArgs = ast.typeParameters.map((p: AST.AST) =>
          getTypeNode(p, { mode: "expression", referencedEntities, knownEntities, hasResources, isInput, processingStack })
        )
        return ts.factory.createTypeReferenceNode("Record", typeArgs.length > 0 ? typeArgs : undefined)
      }

      if (identifier === "Date") {
        return ts.factory.createTypeReferenceNode("Date")
      }

      // For other declarations, try to use the identifier or fall back to unknown
      if (identifier) {
        const typeArgs = ast.typeParameters.map((p: AST.AST) =>
          getTypeNode(p, { mode: "expression", referencedEntities, knownEntities, hasResources, isInput, processingStack })
        )
        return ts.factory.createTypeReferenceNode(identifier, typeArgs.length > 0 ? typeArgs : undefined)
      }

      return ts.factory.createKeywordTypeNode(ts.SyntaxKind.UnknownKeyword)
    }

    case "Suspend": {
      // Suspend is for recursive types - evaluate the thunk and go through getTypeNode for cycle detection
      const suspended = ast.f()
      return getTypeNode(suspended, {
        mode: "expression",
        referencedEntities,
        knownEntities,
        hasResources,
        isInput,
        processingStack
      })
    }

    default:
      console.warn(`Unhandled AST type: ${(ast as AST.AST)._tag}`)
      return ts.factory.createKeywordTypeNode(ts.SyntaxKind.UnknownKeyword)
  }
}

/**
 * Result of union-level matching
 */
interface UnionMatchResult {
  entityName: string
  schema: Schema.Schema.All
  extraFields: Map<string, AST.PropertySignature>
}

/**
 * Try to match an anonymous union against known union entities.
 * Returns match info if the union's members collectively match a known union with uniform extra fields.
 */
function tryMatchUnion(
  anonUnion: AST.Union,
  knownEntities: Map<string, KnownEntityInfo>
): UnionMatchResult | undefined {
  // Get fields for each anonymous member
  const anonMemberFields: Set<string>[] = []
  const anonMembers: AST.TypeLiteral[] = []

  for (const member of anonUnion.types) {
    // Unwrap to TypeLiteral
    let unwrapped = member
    while (unwrapped._tag === "Refinement" || unwrapped._tag === "Transformation") {
      unwrapped = unwrapped._tag === "Refinement" ? unwrapped.from : unwrapped.to
    }
    if (unwrapped._tag !== "TypeLiteral") return undefined

    anonMembers.push(unwrapped)
    anonMemberFields.push(new Set(unwrapped.propertySignatures.map(p => String(p.name))))
  }

  // Try each known union entity
  for (const [name, info] of knownEntities) {
    if (!info.isUnion || !info.memberFields) continue
    if (info.memberFields.length !== anonMemberFields.length) continue

    // For each anonymous member, find the BEST matching known member (most fields = fewest extras)
    const extraFieldSets: Set<string>[] = []
    let allMatched = true

    for (const anonFields of anonMemberFields) {
      let bestExtras: Set<string> | undefined
      let bestMatchSize = -1

      for (const knownMemberFields of info.memberFields) {
        const isSubset = [...knownMemberFields].every(f => anonFields.has(f))
        if (isSubset && knownMemberFields.size > bestMatchSize) {
          bestExtras = new Set([...anonFields].filter(f => !knownMemberFields.has(f)))
          bestMatchSize = knownMemberFields.size
        }
      }

      if (!bestExtras) {
        allMatched = false
        break
      }
      extraFieldSets.push(bestExtras)
    }

    if (!allMatched) continue

    // All members must have identical extra fields
    const firstExtras = [...extraFieldSets[0]]
    const allSame = extraFieldSets.every(s =>
      s.size === firstExtras.length && firstExtras.every(f => s.has(f))
    )
    if (!allSame) continue

    // Must have at least one extra field
    if (firstExtras.length === 0) continue

    // Build extra fields map from first anonymous member (they're all the same)
    const extraFields = new Map<string, AST.PropertySignature>()
    for (const fieldName of firstExtras) {
      const prop = anonMembers[0].propertySignatures.find(p => String(p.name) === fieldName)
      if (prop) extraFields.set(fieldName, prop)
    }

    return { entityName: name, schema: info.schema, extraFields }
  }

  return undefined
}

/**
 * Convert an Effect Schema to a TypeScript AST type node
 */
export function getTypeNode(ast: AST.AST, options: AstGenOptions): ts.TypeNode {
  const {
    mode,
    knownEntities = new Map(),
    hasResources = { value: false },
    isInput = false,
    processingStack = new Set()
  } = options

  // Whether entity extraction is enabled (caller explicitly provided referencedEntities)
  const trackEntities = options.referencedEntities !== undefined
  const referencedEntities = options.referencedEntities ?? new Map<string, Schema.Schema.All>()

  // In 'expression' mode, check for identifier and return references
  if (mode === "expression") {
    const identifier = getIdentifier(ast)
    if (identifier) {
      // Special handling for built-in types (always use references)
      if (identifier === "Resource") {
        hasResources.value = true
        if (isInput) {
          return ts.factory.createUnionTypeNode([
            ts.factory.createTypeReferenceNode("Resource"),
            ts.factory.createTypeReferenceNode("Content")
          ])
        }
        return ts.factory.createTypeReferenceNode("Resource")
      }

      if (identifier === "DateTime") {
        return ts.factory.createTypeReferenceNode("DateTime")
      }

      if (identifier === "Duration") {
        return ts.factory.createTypeReferenceNode("Duration")
      }

      // Check for cycles
      if (processingStack.has(ast)) {
        return ts.factory.createTypeReferenceNode(identifier)
      }

      // When tracking entities, store reference and return type ref.
      // When not tracking (extractCommon: false), fall through to inline the full structure.
      if (trackEntities) {
        referencedEntities.set(identifier, Schema.make(ast))
        return ts.factory.createTypeReferenceNode(identifier)
      }
    }

    // For anonymous Unions, try union-level matching against known union entities
    if (trackEntities && ast._tag === "Union" && knownEntities.size > 0) {
      const match = tryMatchUnion(ast, knownEntities)
      if (match) {
        referencedEntities.set(match.entityName, match.schema)

        // Build intersection: KnownUnion & { extraFields }
        const baseRef = ts.factory.createTypeReferenceNode(match.entityName)
        const extraProps: ts.TypeElement[] = []

        for (const [fieldName, fieldAst] of match.extraFields) {
          const propType = getTypeNode(fieldAst.type, {
            mode: "expression",
            referencedEntities,
            knownEntities,
            hasResources,
            isInput,
            processingStack
          })
          const propSignature = ts.factory.createPropertySignature(
            undefined,
            fieldName,
            fieldAst.isOptional ? ts.factory.createToken(ts.SyntaxKind.QuestionToken) : undefined,
            propType
          )
          extraProps.push(propSignature)
        }

        const extraType = ts.factory.createTypeLiteralNode(extraProps)
        return ts.factory.createIntersectionTypeNode([baseRef, extraType])
      }
    }

    // For anonymous TypeLiterals, try struct-level matching against known non-union entities
    if (trackEntities && ast._tag === "TypeLiteral" && knownEntities.size > 0) {
      const currentFields = new Set(ast.propertySignatures.map(p => String(p.name)))

      // Find best matching known entity (most fields matched, prefer non-unions)
      let bestMatch: { name: string; extraFields: string[] } | undefined
      let bestMatchSize = 0
      let bestMatchIsUnion = true

      for (const [name, info] of knownEntities) {
        // Known entity must be a strict subset (all its fields present in current)
        const isSubset = [...info.fields].every(f => currentFields.has(f))
        if (!isSubset) continue

        const extraFields = [...currentFields].filter(f => !info.fields.has(f))
        // Only match if there are extra fields (otherwise it would be exact match)
        if (extraFields.length === 0) continue

        // Prefer: more fields > non-union > first found
        const isBetter = info.fields.size > bestMatchSize ||
          (info.fields.size === bestMatchSize && bestMatchIsUnion && !info.isUnion)

        if (isBetter) {
          bestMatch = { name, extraFields }
          bestMatchSize = info.fields.size
          bestMatchIsUnion = info.isUnion
        }
      }

      if (bestMatch) {
        // Add matched entity to referenced entities
        const matchedInfo = knownEntities.get(bestMatch.name)
        if (matchedInfo) {
          referencedEntities.set(bestMatch.name, matchedInfo.schema)
        }

        // Build intersection type: KnownEntity & { extraFields... }
        const baseRef = ts.factory.createTypeReferenceNode(bestMatch.name)

        // Build the extra fields as a type literal
        const extraProps: ts.TypeElement[] = []
        for (const extraFieldName of bestMatch.extraFields) {
          const prop = ast.propertySignatures.find(p => String(p.name) === extraFieldName)
          if (prop) {
            const propType = getTypeNode(prop.type, {
              mode: "expression",
              referencedEntities,
              knownEntities,
              hasResources,
              isInput,
              processingStack
            })
            const propSignature = ts.factory.createPropertySignature(
              undefined,
              extraFieldName,
              prop.isOptional ? ts.factory.createToken(ts.SyntaxKind.QuestionToken) : undefined,
              propType
            )
            extraProps.push(propSignature)
          }
        }

        const extraType = ts.factory.createTypeLiteralNode(extraProps)
        return ts.factory.createIntersectionTypeNode([baseRef, extraType])
      }
    }
  }

  // Get the actual structure — pass options.referencedEntities (preserves undefined for no-tracking mode)
  return getSchemaStructure(ast, options.referencedEntities, knownEntities, hasResources, isInput, processingStack)
}

/**
 * Convert a Schema to a TypeScript AST type node
 */
export function schemaToTypeNode(schema: Schema.Schema.All, options: AstGenOptions): ts.TypeNode {
  return getTypeNode(schema.ast, options)
}

/**
 * Print an AST node to a string
 */
export function printAst(node: ts.Node, printerOptions?: ts.PrinterOptions): string {
  const printer = ts.createPrinter(printerOptions)
  const sourceFile = ts.createSourceFile("", "", ts.ScriptTarget.Latest, false, ts.ScriptKind.TS)
  return printer.printNode(ts.EmitHint.Unspecified, node, sourceFile)
}

/**
 * Extract fields from a TypeLiteral, unwrapping Refinement/Transformation
 */
function getTypeFields(ast: AST.AST): Set<string> | undefined {
  if (ast._tag === "TypeLiteral") {
    return new Set(ast.propertySignatures.map(p => String(p.name)))
  }
  if (ast._tag === "Refinement") {
    return getTypeFields(ast.from)
  }
  if (ast._tag === "Transformation") {
    return getTypeFields(ast.to)
  }
  return undefined
}

/**
 * Extract entity info from an AST for structural matching
 */
function extractEntityInfo(ast: AST.AST): KnownEntityInfo | undefined {
  if (ast._tag === "TypeLiteral") {
    return {
      schema: Schema.make(ast),
      fields: new Set(ast.propertySignatures.map(p => String(p.name))),
      isUnion: false
    }
  }

  if (ast._tag === "Union" && ast.types.length > 0) {
    const memberFields: Set<string>[] = []
    for (const member of ast.types) {
      const fields = getTypeFields(member)
      if (!fields) return undefined
      memberFields.push(fields)
    }
    return {
      schema: Schema.make(ast),
      fields: memberFields[0],
      isUnion: true,
      memberFields
    }
  }

  if (ast._tag === "Refinement") {
    return extractEntityInfo(ast.from)
  }
  if (ast._tag === "Transformation") {
    return extractEntityInfo(ast.to)
  }

  return undefined
}

/**
 * Build a knownEntities map from a list of entity schemas.
 */
export function buildKnownEntities(
  entities: ReadonlyArray<Schema.Schema.All>
): Map<string, KnownEntityInfo> {
  const result = new Map<string, KnownEntityInfo>()

  for (const schema of entities) {
    const identifier = getIdentifier(schema.ast)
    if (!identifier) continue

    const info = extractEntityInfo(schema.ast)
    if (!info) continue

    result.set(identifier, info)
  }

  return result
}
