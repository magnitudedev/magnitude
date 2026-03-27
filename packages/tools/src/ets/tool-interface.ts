/**
 * Tool Interface Generator
 *
 * Generates TypeScript function signatures from Tool definitions.
 * Errors are shown as @throws JSDoc annotations, not return types.
 */

import ts from "typescript"
import { Schema } from "@effect/schema"
import { AST } from "@effect/schema"
import type { ToolDefinition } from "../tool-definition"
import type { ToolErrorBase } from "../errors"
import { schemaToTypeNode, getTypeNode, printAst, unwrapOptionalUnion, type KnownEntityInfo } from "./converter"

/**
 * Unwrap AST to get the structural input type (what the user provides).
 * - Transformation: use "from" (pre-transform input)
 * - Refinement: use "from" (pre-validation input)
 */
function unwrapInputAst(ast: AST.AST): AST.AST {
  if (ast._tag === "Transformation") {
    return unwrapInputAst(ast.from)
  }
  if (ast._tag === "Refinement") {
    return unwrapInputAst(ast.from)
  }
  return ast
}

/**
 * Unwrap AST to get the structural output type (what the tool returns).
 * - Transformation: use "to" (post-transform output)
 * - Refinement: use "from" (refinements don't change type)
 */
function unwrapOutputAst(ast: AST.AST): AST.AST {
  if (ast._tag === "Transformation") {
    return unwrapOutputAst(ast.to)
  }
  if (ast._tag === "Refinement") {
    return unwrapOutputAst(ast.from)
  }
  return ast
}

/**
 * Collect known entities from tool schemas by scanning for types with identifiers.
 */
function collectKnownEntities(tools: ReadonlyArray<ToolDefinition>): Map<string, KnownEntityInfo> {
  const result = new Map<string, KnownEntityInfo>()

  function getStructFields(ast: AST.AST): Set<string> | undefined {
    if (ast._tag === "TypeLiteral") {
      return new Set(ast.propertySignatures.map(p => String(p.name)))
    }
    if (ast._tag === "Refinement") return getStructFields(ast.from)
    if (ast._tag === "Transformation") return getStructFields(ast.to)
    return undefined
  }

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
        const fields = getStructFields(member)
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

    if (ast._tag === "Refinement") return extractEntityInfo(ast.from)
    if (ast._tag === "Transformation") return extractEntityInfo(ast.to)
    return undefined
  }

  function visit(ast: AST.AST, visited: Set<AST.AST>): void {
    if (visited.has(ast)) return
    visited.add(ast)

    const id = AST.getIdentifierAnnotation(ast)
    if (id._tag === "Some" && !result.has(id.value)) {
      const info = extractEntityInfo(ast)
      if (info) result.set(id.value, info)
    }

    if (ast._tag === "TypeLiteral") {
      for (const prop of ast.propertySignatures) visit(prop.type, visited)
    } else if (ast._tag === "Union") {
      for (const t of ast.types) visit(t, visited)
    } else if (ast._tag === "TupleType") {
      for (const el of ast.elements) visit(el.type, visited)
      for (const rest of ast.rest) visit(rest.type, visited)
    } else if (ast._tag === "Refinement") {
      visit(ast.from, visited)
    } else if (ast._tag === "Transformation") {
      visit(ast.from, visited)
      visit(ast.to, visited)
    } else if (ast._tag === "Suspend") {
      visit(ast.f(), visited)
    }
  }

  const visited = new Set<AST.AST>()
  for (const tool of tools) {
    if (tool.inputSchema) visit(tool.inputSchema.ast, visited)
    if (tool.outputSchema) visit(tool.outputSchema.ast, visited)
  }

  return result
}

/**
 * Result of generating a tool interface
 */
export interface ToolInterfaceResult {
  /** The function signature as a string */
  signature: string
  /** Names of referenced entity types */
  referencedEntities: string[]
  /** Whether Resources are used */
  hasResources: boolean
  /** Generated type definitions for entities */
  entityDefinitions: string[]
  /** Error types that can be thrown (for documentation) */
  errorTypes: string[]
}

/**
 * Options for controlling tool interface generation output.
 */
export interface ToolInterfaceOptions {
  /**
   * Extract common named types into separate type definitions.
   * When false, all types are inlined in the function signature.
   * @default true
   */
  readonly extractCommon?: boolean
  /**
   * Show error schemas as @throws JSDoc annotations and error class declarations.
   * When false, errors are omitted from the output entirely.
   * @default true
   */
  readonly showErrors?: boolean
  /**
   * Override the function name in the generated signature.
   * When set, uses this instead of tool.name.
   */
  readonly nameOverride?: string
}

/**
 * Generate a TypeScript function interface from a Tool definition
 *
 * @param tool - The tool to generate an interface for
 * @param errorNamespace - Namespace for error types (e.g., "reddit" for reddit.RateLimitError)
 * @param knownEntities - Optional map of known entity schemas for structural matching
 * @param options - Options to control output format
 */
export function generateToolInterface(
  tool: ToolDefinition,
  errorNamespace: string,
  knownEntities?: Map<string, KnownEntityInfo>,
  options?: ToolInterfaceOptions,
): ToolInterfaceResult {
  const extractCommon = options?.extractCommon ?? true
  const showErrors = options?.showErrors ?? true
  // When extractCommon is false, don't track entities — everything inlines
  const referencedEntities = extractCommon ? new Map<string, Schema.Schema.All>() : new Map<string, Schema.Schema.All>()
  const effectiveKnownEntities = extractCommon ? knownEntities : undefined
  const hasResourcesRef = { value: false }
  const errorTypes: string[] = []

  // Generate input parameter type
  const inputType = schemaToTypeNode(tool.inputSchema, {
    mode: "expression",
    referencedEntities: extractCommon ? referencedEntities : undefined,
    knownEntities: effectiveKnownEntities,
    hasResources: hasResourcesRef,
    isInput: true
  })

  // Generate output type
  const outputType = schemaToTypeNode(tool.outputSchema, {
    mode: "expression",
    referencedEntities: extractCommon ? referencedEntities : undefined,
    knownEntities: effectiveKnownEntities,
    hasResources: hasResourcesRef,
    isInput: false
  })

  // Process error schema if present
  if (showErrors && tool.errorSchema) {
    const errorAst = tool.errorSchema.ast

    // Extract error type names from union or single error
    if (errorAst._tag === "Union") {
      for (const errorType of errorAst.types) {
        const errorName = extractErrorName(errorType)
        if (errorName) {
          const qualifiedName = `${errorNamespace}.${errorName}`
          errorTypes.push(qualifiedName)
        }
      }
    } else {
      const errorName = extractErrorName(errorAst)
      if (errorName) {
        const qualifiedName = `${errorNamespace}.${errorName}`
        errorTypes.push(qualifiedName)
      }
    }
  }

  // Build parameter declarations
  let params: ts.ParameterDeclaration[]

  // Unwrap to get the structural input type (handles Transformation from optionalWith, etc.)
  const inputAst = unwrapInputAst(tool.inputSchema.ast)
  const hasNoProperties = inputAst._tag === "TypeLiteral" && inputAst.propertySignatures.length === 0

  if (hasNoProperties) {
    // No parameters needed for empty input
    params = []
  } else {
    // Use single params object
    params = [
      ts.factory.createParameterDeclaration(
        undefined,
        undefined,
        "params",
        undefined,
        inputType,
        undefined
      )
    ]
  }

  // Build function signature
  const funcDecl = ts.factory.createFunctionDeclaration(
    undefined,
    undefined,
    options?.nameOverride ?? tool.name,
    undefined,
    params,
    outputType,
    undefined
  )

  // Add JSDoc with description and @throws
  const jsDocParts: string[] = []

  if (tool.description) {
    jsDocParts.push(tool.description)
  }

  for (const errorType of errorTypes) {
    jsDocParts.push(`@throws {${errorType}}`)
  }

  if (jsDocParts.length > 0) {
    const jsDocText = jsDocParts.length === 1
      ? `* ${jsDocParts[0]} `
      : `*\n * ${jsDocParts.join("\n * ")}\n `

    ts.addSyntheticLeadingComment(
      funcDecl,
      ts.SyntaxKind.MultiLineCommentTrivia,
      jsDocText,
      true
    )
  }

  // Generate entity definitions (skip when extractCommon is false — all types are inlined)
  const entityDefinitions = extractCommon
    ? generateEntityDefinitions(referencedEntities, hasResourcesRef, knownEntities)
    : []

  return {
    signature: printAst(funcDecl),
    referencedEntities: Array.from(referencedEntities.keys()),
    hasResources: hasResourcesRef.value,
    entityDefinitions,
    errorTypes
  }
}

/**
 * Extract the error name from a TypeLiteral AST (looks for _tag literal)
 */
function extractErrorName(ast: AST.AST): string | undefined {
  if (ast._tag === "TypeLiteral") {
    for (const prop of ast.propertySignatures) {
      if (prop.name === "_tag" && prop.type._tag === "Literal") {
        const value = prop.type.literal
        if (typeof value === "string") {
          return value
        }
      }
    }
  }
  return undefined
}

/**
 * Info about an error type for generating class declaration
 */
interface ErrorClassInfo {
  name: string
  properties: Array<{ name: string; type: ts.TypeNode; optional: boolean }>
}

/**
 * Extract error class info from a TypeLiteral AST
 */
function extractErrorClassInfo(
  ast: AST.AST,
  referencedEntities: Map<string, Schema.Schema.All>,
  hasResourcesRef: { value: boolean }
): ErrorClassInfo | undefined {
  if (ast._tag !== "TypeLiteral") return undefined

  let name: string | undefined
  const properties: ErrorClassInfo["properties"] = []

  for (const prop of ast.propertySignatures) {
    const propName = typeof prop.name === "string" ? prop.name : String(prop.name)

    if (propName === "_tag" && prop.type._tag === "Literal") {
      const value = prop.type.literal
      if (typeof value === "string") {
        name = value
      }
    } else {
      const typeNode = schemaToTypeNode(Schema.make(prop.type), {
        mode: "expression",
        referencedEntities,
        hasResources: hasResourcesRef,
        isInput: false
      })
      properties.push({ name: propName, type: typeNode, optional: prop.isOptional })
    }
  }

  if (!name) return undefined
  return { name, properties }
}

/**
 * Generate a class declaration string for an error type
 */
function generateErrorClassDeclaration(info: ErrorClassInfo): string {
  const members: string[] = []
  // _tag is internal discriminator, not useful for LLM
  // members.push(`  _tag: "${info.name}"`)

  for (const prop of info.properties) {
    const typeStr = printAst(prop.type)
    const optional = prop.optional ? "?" : ""
    members.push(`  ${prop.name}${optional}: ${typeStr}`)
  }

  return `class ${info.name} extends Error {\n${members.join("\n")}\n}`
}

/**
 * Generate type definitions for all referenced entities
 */
function generateEntityDefinitions(
  referencedEntities: Map<string, Schema.Schema.All>,
  hasResourcesRef: { value: boolean },
  knownEntities?: Map<string, KnownEntityInfo>
): string[] {
  // Skip built-in types and primitives
  const skipTypes = new Set([
    "Resource", "Content", "DateTime", "Duration",
    "string", "number", "boolean", "undefined", "null", "unknown", "any", "void", "never",
    "bigint", "symbol", "object"
  ])
  const entityDefinitions: string[] = []
  const processedEntities = new Set<string>()

  let entitiesToProcess = Array.from(referencedEntities.entries()).filter(
    ([name]) => !skipTypes.has(name)
  )

  while (entitiesToProcess.length > 0) {
    const [typeName, schema] = entitiesToProcess.shift()!

    if (processedEntities.has(typeName)) {
      continue
    }

    processedEntities.add(typeName)

    const sizeBefore = referencedEntities.size

    // Generate type definition
    const typeNode = schemaToTypeNode(schema, {
      mode: "definition",
      referencedEntities,
      knownEntities,
      hasResources: hasResourcesRef,
      isInput: false
    })

    const typeAlias = ts.factory.createTypeAliasDeclaration(
      undefined,
      typeName,
      undefined,
      typeNode
    )

    entityDefinitions.push(printAst(typeAlias))

    // Check for newly discovered entities
    if (referencedEntities.size > sizeBefore) {
      entitiesToProcess = Array.from(referencedEntities.entries()).filter(
        ([name]) => !skipTypes.has(name) && !processedEntities.has(name)
      )
    }
  }

  return entityDefinitions
}

/**
 * Options for controlling tool group interface generation output.
 */
export interface ToolGroupInterfaceOptions extends ToolInterfaceOptions {
  /**
   * Wrap tools in a `declare namespace` block.
   * When false, tools are rendered as standalone functions prefixed with `groupName.`.
   * @default true
   */
  readonly useNamespace?: boolean
}

/**
 * Generate interfaces for a tool group
 */
export function generateToolGroupInterface(
  groupName: string,
  tools: ReadonlyArray<ToolDefinition>,
  options?: ToolGroupInterfaceOptions,
): string {
  const extractCommon = options?.extractCommon ?? true
  const showErrors = options?.showErrors ?? true
  const useNamespace = options?.useNamespace ?? true

  // Build known entities from all tool schemas (skip when extractCommon is false)
  const knownEntities = extractCommon ? collectKnownEntities(tools) : undefined

  const allResults = tools.map(tool => generateToolInterface(tool, groupName, knownEntities, {
    extractCommon,
    showErrors,
    nameOverride: useNamespace ? undefined : `${groupName}.${tool.name}`,
  }))

  // Collect all unique entity definitions
  const seenEntities = new Set<string>()
  const allEntityDefs: string[] = []

  for (const result of allResults) {
    for (const def of result.entityDefinitions) {
      // Extract type name from definition
      const match = def.match(/^type (\w+)/)
      if (match && !seenEntities.has(match[1])) {
        seenEntities.add(match[1])
        allEntityDefs.push(def)
      }
    }
  }

  // Collect all unique error classes from tools
  const errorClassDecls: string[] = []

  if (showErrors) {
    const errorReferencedEntities = new Map<string, Schema.Schema.All>()
    const errorHasResourcesRef = { value: false }
    const seenErrors = new Set<string>()

    for (const tool of tools) {
      if (!tool.errorSchema) continue
      const errorAst = tool.errorSchema.ast

      const processErrorAst = (ast: AST.AST) => {
        const info = extractErrorClassInfo(ast, errorReferencedEntities, errorHasResourcesRef)
        if (info && !seenErrors.has(info.name)) {
          seenErrors.add(info.name)
          errorClassDecls.push(generateErrorClassDeclaration(info))
        }
      }

      if (errorAst._tag === "Union") {
        for (const member of errorAst.types) {
          processErrorAst(member)
        }
      } else {
        processErrorAst(errorAst)
      }
    }
  }

  // Build output
  const parts: string[] = []

  // Entity definitions first (outside namespace)
  if (allEntityDefs.length > 0) {
    parts.push("// Types")
    parts.push(...allEntityDefs)
    parts.push("")
  }

  if (useNamespace) {
    // Namespace with error classes and functions
    parts.push(`declare namespace ${groupName} {`)

    // Error classes first
    for (const decl of errorClassDecls) {
      const indented = decl.split("\n").map(line => "  " + line).join("\n")
      parts.push(indented)
    }

    // Then functions
    for (const result of allResults) {
      const indented = result.signature
        .split("\n")
        .map((line) => "  " + line)
        .join("\n")
      parts.push(indented)
    }
    parts.push("}")
  } else {
    // Flat mode: individual functions prefixed with groupName.
    // Error classes rendered standalone
    for (const decl of errorClassDecls) {
      parts.push(decl)
    }

    for (const result of allResults) {
      parts.push(result.signature)
    }
  }

  return parts.join("\n")
}
