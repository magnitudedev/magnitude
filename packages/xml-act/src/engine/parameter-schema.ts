/**
 * Parameter Schema Derivation — re-exported from execution/parameter-schema.ts.
 *
 * Derives parameter metadata (name, type, required) from an Effect Schema AST.
 * Used by the parser for jsonish decisions and by the engine for tool schema lookup.
 */

export type { ParameterSchema, ToolSchema, ScalarType } from '../execution/parameter-schema'
export { deriveParameters } from '../execution/parameter-schema'
