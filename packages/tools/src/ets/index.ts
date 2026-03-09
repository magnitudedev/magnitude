/**
 * Effect Schema to TypeScript (ETS)
 *
 * Converts Effect Schema definitions to TypeScript AST nodes or type strings.
 * Similar to zts but for Effect Schema, with errors as @throws annotations.
 */

// Core AST conversion
export { getTypeNode, schemaToTypeNode, printAst, buildKnownEntities, type AstGenOptions, type KnownEntityInfo } from "./converter"

// Tool interface generation
export { generateToolInterface, generateToolGroupInterface, generateToolGroupInterface as generateNamespaceInterface, type ToolInterfaceResult, type ToolInterfaceOptions, type ToolGroupInterfaceOptions } from "./tool-interface"
