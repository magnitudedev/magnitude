
/**
 * SchemaCoercer — Maps parsed JSON values against Effect Schema AST to produce StreamingPartial.
 * 
 * Ported from BAML's coercer architecture. Handles two-phase coercion:
 * 1. tryCast — strict matching for union disambiguation
 * 2. coerce — lenient matching with heuristics for LLM output recovery
 */

import { AST } from "@effect/schema";
import type { ParsedValue, CompletionState } from "./types";
import type { StreamingPartial, StreamingLeaf } from "@magnitudedev/tools";

// ============================================================================
// Types
// ============================================================================

/**
 * Coercion flags track what transformations were applied during coercion.
 * Each flag has a score — lower is better. Score 0 = perfect match.
 */
export type CoercionFlag =
  | { readonly _tag: "incomplete" }
  | { readonly _tag: "stringToNumber"; readonly raw: string }
  | { readonly _tag: "stringToBoolean"; readonly raw: string }
  | { readonly _tag: "floatToInt"; readonly raw: number }
  | { readonly _tag: "extraKey"; readonly key: string }
  | { readonly _tag: "missingRequired"; readonly key: string }
  | { readonly _tag: "impliedKey"; readonly key: string }
  | { readonly _tag: "singleToArray" }
  | { readonly _tag: "arrayItemError"; readonly index: number; readonly error: string }
  | { readonly _tag: "unionMatch"; readonly variantIndex: number }
  | { readonly _tag: "defaultFromNoValue" }
  | { readonly _tag: "optionalDefault" }
  | { readonly _tag: "stringToEnum"; readonly raw: string; readonly expected: string }
  | { readonly _tag: "jsonToString"; readonly raw: ParsedValue };

/**
 * Result of coercion — the coerced value plus diagnostic flags and score.
 */
export type CoercedResult = {
  readonly value: unknown;
  readonly flags: CoercionFlag[];
  readonly score: number;
};

/**
 * Context for coercion — tracks scope path for error reporting and
 * visited class-value pairs to prevent infinite recursion.
 */
type CoerceContext = {
  readonly scope: string[];
  readonly visitedCoerce: Set<string>;
  readonly visitedTryCast: Set<string>;
};

// ============================================================================
// Scoring
// ============================================================================

/**
 * Get the score for a single flag. Lower is better.
 */
function flagScore(flag: CoercionFlag): number {
  switch (flag._tag) {
    case "incomplete":
    case "unionMatch":
      return 0;
    case "optionalDefault":
    case "extraKey":
    case "singleToArray":
    case "stringToBoolean":
    case "stringToNumber":
    case "stringToEnum":
    case "floatToInt":
      return 1;
    case "impliedKey":
    case "jsonToString":
      return 2;
    case "arrayItemError":
      return 1 + flag.index;
    case "missingRequired":
      return 100;
    case "defaultFromNoValue":
      return 100;
  }
}

/**
 * Calculate total score for a set of flags.
 */
function totalScore(flags: CoercionFlag[]): number {
  return flags.reduce((sum, f) => sum + flagScore(f), 0);
}

// ============================================================================
// Context Helpers
// ============================================================================

function createContext(): CoerceContext {
  return {
    scope: [],
    visitedCoerce: new Set(),
    visitedTryCast: new Set(),
  };
}

function enterScope(ctx: CoerceContext, name: string): CoerceContext {
  return {
    ...ctx,
    scope: [...ctx.scope, name],
  };
}

function markVisitedCoerce(ctx: CoerceContext, key: string): CoerceContext {
  const newVisited = new Set(ctx.visitedCoerce);
  newVisited.add(key);
  return { ...ctx, visitedCoerce: newVisited };
}

function markVisitedTryCast(ctx: CoerceContext, key: string): CoerceContext {
  const newVisited = new Set(ctx.visitedTryCast);
  newVisited.add(key);
  return { ...ctx, visitedTryCast: newVisited };
}

function isVisitedCoerce(ctx: CoerceContext, key: string): boolean {
  return ctx.visitedCoerce.has(key);
}

function isVisitedTryCast(ctx: CoerceContext, key: string): boolean {
  return ctx.visitedTryCast.has(key);
}

function displayScope(ctx: CoerceContext): string {
  return ctx.scope.length === 0 ? "<root>" : ctx.scope.join(".");
}

// ============================================================================
// AST Helpers
// ============================================================================

/**
 * Unwrap Transformation, Refinement, and PropertySignature to get the underlying type.
 */
function unwrapAst(ast: AST.AST): AST.AST {
  if (ast._tag === "Transformation") {
    return unwrapAst(ast.from);
  }
  if (ast._tag === "Refinement") {
    return unwrapAst(ast.from);
  }
  if (ast instanceof AST.PropertySignature) {
    return unwrapAst(ast.type);
  }
  return ast;
}

/**
 * Check if AST represents a string type.
 */
function isStringKeyword(ast: AST.AST): boolean {
  const unwrapped = unwrapAst(ast);
  return unwrapped._tag === "StringKeyword";
}

/**
 * Check if AST represents a number type.
 */
function isNumberKeyword(ast: AST.AST): boolean {
  const unwrapped = unwrapAst(ast);
  return unwrapped._tag === "NumberKeyword";
}

/**
 * Check if AST represents a boolean type.
 */
function isBooleanKeyword(ast: AST.AST): boolean {
  const unwrapped = unwrapAst(ast);
  return unwrapped._tag === "BooleanKeyword";
}

/**
 * Check if AST represents undefined (for optional fields).
 */
function isUndefinedKeyword(ast: AST.AST): boolean {
  const unwrapped = unwrapAst(ast);
  return unwrapped._tag === "UndefinedKeyword";
}

/**
 * Check if AST is a literal type.
 */
function isLiteral(ast: AST.AST): boolean {
  const unwrapped = unwrapAst(ast);
  return unwrapped._tag === "Literal";
}

/**
 * Get literal value if it is one.
 */
function getLiteralValue(ast: AST.AST): string | number | boolean | null | undefined {
  const unwrapped = unwrapAst(ast);
  if (unwrapped._tag !== "Literal") return undefined;
  const lit = unwrapped.literal;
  if (typeof lit === 'string' || typeof lit === 'number' || typeof lit === 'boolean' || lit === null) return lit
  return undefined
}

/**
 * Check if AST is a struct/object type.
 */
function isTypeLiteral(ast: AST.AST): boolean {
  const unwrapped = unwrapAst(ast);
  return unwrapped._tag === "TypeLiteral";
}

/**
 * Get struct fields if this is a type literal.
 */
function getStructFields(ast: AST.AST): Array<{ name: string; type: AST.AST; isOptional: boolean }> {
  const unwrapped = unwrapAst(ast);
  if (unwrapped._tag !== "TypeLiteral") return [];
  
  return unwrapped.propertySignatures.map((prop) => ({
    name: String(prop.name),
    type: prop.type,
    isOptional: prop.isOptional,
  }));
}

/**
 * Check if AST is a union type.
 */
function isUnion(ast: AST.AST): boolean {
  const unwrapped = unwrapAst(ast);
  return unwrapped._tag === "Union";
}

/**
 * Get union members if this is a union.
 */
function getUnionMembers(ast: AST.AST): readonly AST.AST[] {
  const unwrapped = unwrapAst(ast);
  if (unwrapped._tag !== "Union") return [];
  return unwrapped.types;
}

/**
 * Check if AST is an array type.
 * Effect Schema uses TupleType with rest element for arrays.
 */
function isArrayType(ast: AST.AST): boolean {
  const unwrapped = unwrapAst(ast);
  if (unwrapped._tag === "TupleType") {
    return unwrapped.rest.length > 0;
  }
  return false;
}

/**
 * Get element type for array.
 */
function getElementType(ast: AST.AST): AST.AST | undefined {
  const unwrapped = unwrapAst(ast);
  if (unwrapped._tag === "TupleType" && unwrapped.rest.length > 0) {
    return unwrapped.rest[0].type;
  }
  return undefined;
}

/**
 * Check if AST is an enum (union of string literals).
 */
function isEnumType(ast: AST.AST): boolean {
  const unwrapped = unwrapAst(ast);
  if (unwrapped._tag !== "Union") return false;
  
  // All members must be string literals
  return unwrapped.types.every((t) => {
    const member = unwrapAst(t);
    return member._tag === "Literal" && typeof member.literal === "string";
  });
}

/**
 * Get enum values if this is an enum type.
 */
function getEnumValues(ast: AST.AST): string[] {
  const unwrapped = unwrapAst(ast);
  if (unwrapped._tag !== "Union") return [];
  
  return unwrapped.types
    .map((t) => {
      const member = unwrapAst(t);
      if (member._tag === "Literal" && typeof member.literal === "string") {
        return member.literal;
      }
      return undefined;
    })
    .filter((v): v is string => v !== undefined);
}

/**
 * Check if a type is optional.
 * Effect Schema represents optionals as Union(String, UndefinedKeyword)
 * or through PropertySignature with isOptional flag.
 */
function isOptionalType(ast: AST.AST): boolean {
  // Check for PropertySignatureDeclaration (from Schema.optional)
  // These extend OptionalType which is not in the AST union type
  if (ast instanceof AST.OptionalType) {
    return true;
  }
  
  const unwrapped = unwrapAst(ast);
  
  // Check for Union with undefined
  if (unwrapped._tag === "Union") {
    return unwrapped.types.some((t) => isUndefinedKeyword(t));
  }
  
  return false;
}

/**
 * Get the non-undefined part of an optional type.
 */
function getOptionalInnerType(ast: AST.AST): AST.AST | undefined {
  // Handle PropertySignatureDeclaration / OptionalType from Schema.optional
  if (ast instanceof AST.OptionalType && 'type' in ast) {
    return (ast as any).type as AST.AST;
  }
  
  const unwrapped = unwrapAst(ast);
  if (unwrapped._tag !== "Union") return undefined;
  
  const nonUndefined = unwrapped.types.find((t) => !isUndefinedKeyword(t));
  return nonUndefined;
}

// ============================================================================
// Core Coercion Functions
// ============================================================================

/**
 * Coerce a ParsedValue to match the schema AST, producing a StreamingPartial-compatible value.
 * This is the lenient path — applies heuristics to recover from LLM errors.
 */
export function coerceToStreamingPartial(
  parsed: ParsedValue | undefined,
  schemaAst: AST.AST,
  ctx?: CoerceContext
): CoercedResult | undefined {
  const context = ctx ?? createContext();
  
  if (parsed === undefined) {
    // No value provided
    if (isOptionalType(schemaAst)) {
      return {
        value: undefined,
        flags: [{ _tag: "optionalDefault" }],
        score: 1,
      };
    }
    return {
      value: undefined,
      flags: [{ _tag: "defaultFromNoValue" }],
      score: 100,
    };
  }

  // Handle optional types (including null -> undefined conversion)
  // This must come before union handling because optional types ARE unions
  if (isOptionalType(schemaAst)) {
    const inner = getOptionalInnerType(schemaAst);
    if (inner) {
      if (parsed._tag === "null") {
        return {
          value: undefined,
          flags: [{ _tag: "optionalDefault" }],
          score: 1,
        };
      }
      return coerceToStreamingPartial(parsed, inner, context);
    }
  }

  // Handle enum types (unions of string literals) before general unions
  if (isEnumType(schemaAst)) {
    return coerceEnum(parsed, schemaAst, context, false);
  }

  // Handle union types (non-optional, non-enum unions)
  if (isUnion(schemaAst)) {
    return coerceUnion(parsed, schemaAst, context, false);
  }

  // Handle array types
  if (isArrayType(schemaAst)) {
    return coerceArray(parsed, schemaAst, context, false);
  }

  // Handle struct/object types
  if (isTypeLiteral(schemaAst)) {
    return coerceStruct(parsed, schemaAst, context, false);
  }

  // Handle literal types (single literals, not unions of them)
  if (isLiteral(schemaAst)) {
    return coerceLiteral(parsed, schemaAst, context, false);
  }

  // Handle primitive types
  if (isStringKeyword(schemaAst)) {
    return coerceString(parsed, context, false);
  }

  if (isNumberKeyword(schemaAst)) {
    return coerceNumber(parsed, context, false);
  }

  if (isBooleanKeyword(schemaAst)) {
    return coerceBoolean(parsed, context, false);
  }

  // Unknown type — try to convert to string
  if (parsed._tag === "string" || parsed._tag === "number" || parsed._tag === "boolean" || parsed._tag === "null") {
    return coerceString(parsed, context, false);
  }

  // For objects/arrays that don't match struct/array expectations, convert to string
  return {
    value: JSON.stringify(parsedToRaw(parsed)),
    flags: [{ _tag: "jsonToString", raw: parsed }],
    score: 2,
  };
}

/**
 * Try to cast a ParsedValue to match the schema AST strictly.
 * Returns undefined if ANY mismatch occurs. Used for union disambiguation.
 */
export function tryCastToStreamingPartial(
  parsed: ParsedValue | undefined,
  schemaAst: AST.AST,
  ctx?: CoerceContext
): CoercedResult | undefined {
  const context = ctx ?? createContext();
  
  if (parsed === undefined) {
    // For try_cast, undefined only matches optional types
    if (isOptionalType(schemaAst)) {
      return {
        value: undefined,
        flags: [],
        score: 0,
      };
    }
    return undefined;
  }

  // Handle optional types by unwrapping (before union check)
  if (isOptionalType(schemaAst)) {
    const inner = getOptionalInnerType(schemaAst);
    if (inner) {
      if (parsed._tag === "null") {
        return {
          value: undefined,
          flags: [],
          score: 0,
        };
      }
      return tryCastToStreamingPartial(parsed, inner, context);
    }
  }

  // Handle enum types (unions of string literals) before general unions
  if (isEnumType(schemaAst)) {
    return coerceEnum(parsed, schemaAst, context, true);
  }

  // Handle union types (non-optional, non-enum unions)
  if (isUnion(schemaAst)) {
    return coerceUnion(parsed, schemaAst, context, true);
  }

  // Handle array types
  if (isArrayType(schemaAst)) {
    return coerceArray(parsed, schemaAst, context, true);
  }

  // Handle struct/object types
  if (isTypeLiteral(schemaAst)) {
    return coerceStruct(parsed, schemaAst, context, true);
  }

  // Handle enum types
  if (isEnumType(schemaAst)) {
    return coerceEnum(parsed, schemaAst, context, true);
  }

  // Handle literal types
  if (isLiteral(schemaAst)) {
    return coerceLiteral(parsed, schemaAst, context, true);
  }

  // Handle primitive types
  if (isStringKeyword(schemaAst)) {
    return coerceString(parsed, context, true);
  }

  if (isNumberKeyword(schemaAst)) {
    return coerceNumber(parsed, context, true);
  }

  if (isBooleanKeyword(schemaAst)) {
    return coerceBoolean(parsed, context, true);
  }

  // Unknown type — fail try_cast
  return undefined;
}

// ============================================================================
// Primitive Coercion
// ============================================================================

function coerceString(
  parsed: ParsedValue,
  ctx: CoerceContext,
  strict: boolean
): CoercedResult | undefined {
  switch (parsed._tag) {
    case "string":
      return {
        value: parsed.value,
        flags: parsed.state === "incomplete" ? [{ _tag: "incomplete" }] : [],
        score: parsed.state === "incomplete" ? 0 : 0,
      };
    
    case "null":
      // Null never coerces to string - it only matches null type or optional
      return undefined;
    
    case "number":
    case "boolean":
      // Convert to string representation
      if (strict) return undefined;
      const strValue = parsed._tag === "number" ? parsed.value : String(parsed.value);
      return {
        value: strValue,
        flags: [{ _tag: "jsonToString", raw: parsed }],
        score: 2,
      };
    
    case "object":
    case "array":
      if (strict) return undefined;
      return {
        value: JSON.stringify(parsedToRaw(parsed)),
        flags: [{ _tag: "jsonToString", raw: parsed }],
        score: 2,
      };
  }
}

function coerceNumber(
  parsed: ParsedValue,
  ctx: CoerceContext,
  strict: boolean
): CoercedResult | undefined {
  switch (parsed._tag) {
    case "number": {
      const numStr = parsed.value;
      const num = parseFloat(numStr);
      
      if (isNaN(num)) {
        return undefined;
      }

      const flags: CoercionFlag[] = [];
      
      // Check if this was a float that rounds to int
      if (Number.isInteger(num) && numStr.includes(".")) {
        flags.push({ _tag: "floatToInt", raw: num });
      }

      if (parsed.state === "incomplete") {
        flags.push({ _tag: "incomplete" });
      }

      return {
        value: num,
        flags,
        score: totalScore(flags),
      };
    }
    
    case "string": {
      if (strict) return undefined;
      
      const trimmed = parsed.value.trim().replace(/,$/, "");
      
      // Try parsing as number
      const num = parseFloat(trimmed);
      if (!isNaN(num)) {
        const flags: CoercionFlag[] = [{ _tag: "stringToNumber", raw: parsed.value }];
        
        // Check if this was a float that rounds to int
        if (Number.isInteger(num) && trimmed.includes(".")) {
          flags.push({ _tag: "floatToInt", raw: num });
        }

        if (parsed.state === "incomplete") {
          flags.push({ _tag: "incomplete" });
        }

        return {
          value: num,
          flags,
          score: totalScore(flags),
        };
      }
      
      return undefined;
    }
    
    case "null":
      if (strict) return undefined;
      return undefined; // Can't coerce null to number
    
    case "boolean":
    case "object":
    case "array":
      if (strict) return undefined;
      return undefined;
  }
}

function coerceBoolean(
  parsed: ParsedValue,
  ctx: CoerceContext,
  strict: boolean
): CoercedResult | undefined {
  switch (parsed._tag) {
    case "boolean":
      return {
        value: parsed.value,
        flags: [],
        score: 0,
      };
    
    case "string": {
      const lower = parsed.value.toLowerCase();
      
      // In strict mode, only accept exact boolean strings with no coercion flag
      if (strict) {
        if (lower === "true") {
          // Even "true" string is a coercion from string to boolean
          // In strict mode, we reject this - the type is wrong
          return undefined;
        }
        if (lower === "false") {
          return undefined;
        }
        return undefined;
      }
      
      // Non-strict mode: apply heuristics
      if (lower === "true") {
        return {
          value: true,
          flags: [{ _tag: "stringToBoolean", raw: parsed.value }],
          score: 1,
        };
      }
      
      if (lower === "false") {
        return {
          value: false,
          flags: [{ _tag: "stringToBoolean", raw: parsed.value }],
          score: 1,
        };
      }
      
      // Fuzzy matching for non-strict mode
      if (lower.startsWith("t") || lower.includes("yes") || lower.includes("1")) {
        return {
          value: true,
          flags: [{ _tag: "stringToBoolean", raw: parsed.value }],
          score: 1,
        };
      }
      if (lower.startsWith("f") || lower.includes("no") || lower.includes("0")) {
        return {
          value: false,
          flags: [{ _tag: "stringToBoolean", raw: parsed.value }],
          score: 1,
        };
      }
      
      return undefined;
    }
    
    case "null":
      if (strict) return undefined;
      return undefined;
    
    case "number":
      if (strict) return undefined;
      const num = parseFloat(parsed.value);
      if (!isNaN(num)) {
        return {
          value: num !== 0,
          flags: [{ _tag: "stringToBoolean", raw: parsed.value }],
          score: 1,
        };
      }
      return undefined;
    
    case "object":
    case "array":
      if (strict) return undefined;
      return undefined;
  }
}

// ============================================================================
// Array Coercion
// ============================================================================

function coerceArray(
  parsed: ParsedValue,
  schemaAst: AST.AST,
  ctx: CoerceContext,
  strict: boolean
): CoercedResult | undefined {
  const elementType = getElementType(schemaAst);
  if (!elementType) {
    return undefined;
  }

  switch (parsed._tag) {
    case "array": {
      const items: unknown[] = [];
      const flags: CoercionFlag[] = [];
      
      if (parsed.state === "incomplete") {
        flags.push({ _tag: "incomplete" });
      }

      for (let i = 0; i < parsed.items.length; i++) {
        const itemCtx = enterScope(ctx, String(i));
        const coerced = strict
          ? tryCastToStreamingPartial(parsed.items[i], elementType, itemCtx)
          : coerceToStreamingPartial(parsed.items[i], elementType, itemCtx);
        
        if (coerced === undefined) {
          if (strict) return undefined;
          flags.push({ _tag: "arrayItemError", index: i, error: "Failed to coerce array item" });
        } else {
          items.push(coerced.value);
          flags.push(...coerced.flags);
        }
      }

      return {
        value: items,
        flags,
        score: totalScore(flags),
      };
    }
    
    case "null":
      if (strict) return undefined;
      return {
        value: [],
        flags: [{ _tag: "singleToArray" }],
        score: 1,
      };
    
    default:
      // Single value to array
      if (strict) return undefined;
      
      const coerced = coerceToStreamingPartial(parsed, elementType, ctx);
      if (coerced === undefined) {
        return undefined;
      }

      return {
        value: [coerced.value],
        flags: [...coerced.flags, { _tag: "singleToArray" }],
        score: coerced.score + 1,
      };
  }
}

// ============================================================================
// Struct/Object Coercion
// ============================================================================

function coerceStruct(
  parsed: ParsedValue,
  schemaAst: AST.AST,
  ctx: CoerceContext,
  strict: boolean
): CoercedResult | undefined {
  if (parsed._tag !== "object") {
    // Single-field implied key heuristic
    if (!strict) {
      const fields = getStructFields(schemaAst);
      if (fields.length === 1 && fields[0]) {
        const field = fields[0];
        const fieldCtx = enterScope(ctx, String(field.name));
        const coerced = coerceToStreamingPartial(parsed, field.type, fieldCtx);
        
        if (coerced !== undefined) {
          const result: Record<string, unknown> = {};
          result[String(field.name)] = coerced.value;
          
          return {
            value: result,
            flags: [
              ...coerced.flags,
              { _tag: "impliedKey", key: String(field.name) },
            ],
            score: coerced.score + 2,
          };
        }
      }
    }
    
    return undefined;
  }

  const fields = getStructFields(schemaAst);
  const result: Record<string, unknown> = {};
  const flags: CoercionFlag[] = [];
  
  if (parsed.state === "incomplete") {
    flags.push({ _tag: "incomplete" });
  }

  // Track which parsed keys we've matched to schema fields
  const matchedParsedKeys = new Set<string>();
  const parsedEntries = new Map(parsed.entries);

  // Match each schema field to a parsed value
  for (const field of fields) {
    const fieldName = String(field.name);
    const fieldCtx = enterScope(ctx, fieldName);
    
    // Try exact match first
    let parsedValue = parsedEntries.get(fieldName);
    let keyUsed: string | undefined = fieldName;
    
    // Try case-insensitive match
    if (parsedValue === undefined) {
      for (const [key, value] of parsedEntries) {
        if (key.toLowerCase() === fieldName.toLowerCase()) {
          parsedValue = value;
          keyUsed = key;
          break;
        }
      }
    }
    
    // Try alphanumeric-stripped match
    if (parsedValue === undefined) {
      const normalizedField = fieldName.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
      for (const [key, value] of parsedEntries) {
        const normalizedKey = key.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
        if (normalizedKey === normalizedField) {
          parsedValue = value;
          keyUsed = key;
          break;
        }
      }
    }

    if (parsedValue !== undefined && keyUsed !== undefined) {
      matchedParsedKeys.add(keyUsed);
    }

    if (parsedValue === undefined) {
      // Field is missing
      if (!field.isOptional) {
        if (strict) return undefined;
        flags.push({ _tag: "missingRequired", key: fieldName });
      } else {
        // Optional field missing — omit from result
      }
      continue;
    }

    const coerced = strict
      ? tryCastToStreamingPartial(parsedValue, field.type, fieldCtx)
      : coerceToStreamingPartial(parsedValue, field.type, fieldCtx);
    
    if (coerced === undefined) {
      if (field.isOptional) {
        // Optional field that failed coercion — omit
        continue;
      }
      if (strict) return undefined;
      flags.push({ _tag: "missingRequired", key: fieldName });
      continue;
    }

    result[fieldName] = coerced.value;
    flags.push(...coerced.flags);
  }

  // Handle extra keys
  for (const [key, value] of parsedEntries) {
    if (!matchedParsedKeys.has(key)) {
      if (strict) return undefined;
      flags.push({ _tag: "extraKey", key });
    }
  }

  return {
    value: result,
    flags,
    score: totalScore(flags),
  };
}

// ============================================================================
// Enum Coercion
// ============================================================================

function coerceEnum(
  parsed: ParsedValue,
  schemaAst: AST.AST,
  ctx: CoerceContext,
  strict: boolean
): CoercedResult | undefined {
  if (parsed._tag !== "string") {
    return undefined;
  }

  const enumValues = getEnumValues(schemaAst);
  const value = parsed.value;

  // Exact match
  for (const enumVal of enumValues) {
    if (enumVal === value) {
      return {
        value: enumVal,
        flags: parsed.state === "incomplete" ? [{ _tag: "incomplete" }] : [],
        score: parsed.state === "incomplete" ? 0 : 0,
      };
    }
  }

  if (strict) return undefined;

  // Case-insensitive match
  const lowerValue = value.toLowerCase();
  for (const enumVal of enumValues) {
    if (enumVal.toLowerCase() === lowerValue) {
      return {
        value: enumVal,
        flags: [
          { _tag: "stringToEnum", raw: value, expected: enumVal },
          ...(parsed.state === "incomplete" ? [{ _tag: "incomplete" } as CoercionFlag] : []),
        ],
        score: 1,
      };
    }
  }

  // Fuzzy match — find closest by Levenshtein distance
  let bestMatch: string | undefined;
  let bestDistance = Infinity;
  
  for (const enumVal of enumValues) {
    const distance = levenshteinDistance(value, enumVal);
    if (distance < bestDistance && distance <= 2) {
      bestDistance = distance;
      bestMatch = enumVal;
    }
  }

  if (bestMatch !== undefined) {
    return {
      value: bestMatch,
      flags: [
        { _tag: "stringToEnum", raw: value, expected: bestMatch },
        ...(parsed.state === "incomplete" ? [{ _tag: "incomplete" } as CoercionFlag] : []),
      ],
      score: 1,
    };
  }

  return undefined;
}

// ============================================================================
// Literal Coercion
// ============================================================================

function coerceLiteral(
  parsed: ParsedValue,
  schemaAst: AST.AST,
  ctx: CoerceContext,
  strict: boolean
): CoercedResult | undefined {
  const expectedValue = getLiteralValue(schemaAst);
  if (expectedValue === undefined) return undefined;

  switch (parsed._tag) {
    case "string":
      if (parsed.value === String(expectedValue)) {
        return {
          value: expectedValue,
          flags: parsed.state === "incomplete" ? [{ _tag: "incomplete" }] : [],
          score: 0,
        };
      }
      if (!strict && parsed.value.toLowerCase() === String(expectedValue).toLowerCase()) {
        return {
          value: expectedValue,
          flags: [{ _tag: "stringToEnum", raw: parsed.value, expected: String(expectedValue) }],
          score: 1,
        };
      }
      return undefined;
    
    case "number":
      if (expectedValue === parseFloat(parsed.value)) {
        return {
          value: expectedValue,
          flags: [],
          score: 0,
        };
      }
      return undefined;
    
    case "boolean":
      if (expectedValue === parsed.value) {
        return {
          value: expectedValue,
          flags: [],
          score: 0,
        };
      }
      return undefined;
    
    case "null":
      if (expectedValue === null) {
        return {
          value: null,
          flags: [],
          score: 0,
        };
      }
      return undefined;
    
    default:
      return undefined;
  }
}

// ============================================================================
// Union Coercion
// ============================================================================

function coerceUnion(
  parsed: ParsedValue,
  schemaAst: AST.AST,
  ctx: CoerceContext,
  strict: boolean
): CoercedResult | undefined {
  const members = getUnionMembers(schemaAst);
  
  // Try each variant
  const results: Array<{ result: CoercedResult; index: number }> = [];
  
  for (let i = 0; i < members.length; i++) {
    const member = members[i];
    
    const coerced = strict
      ? tryCastToStreamingPartial(parsed, member, ctx)
      : coerceToStreamingPartial(parsed, member, ctx);
    
    if (coerced !== undefined) {
      // Add union match flag
      const withFlag: CoercedResult = {
        ...coerced,
        flags: [...coerced.flags, { _tag: "unionMatch", variantIndex: i }],
      };
      
      // In strict mode, short-circuit on perfect match (score 0)
      if (strict && withFlag.score === 0) {
        return withFlag;
      }
      
      results.push({ result: withFlag, index: i });
    }
  }

  if (results.length === 0) {
    return undefined;
  }

  // Pick best match (lowest score)
  results.sort((a, b) => a.result.score - b.result.score);
  return results[0].result;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Convert a ParsedValue back to a raw JavaScript value for JSON.stringify.
 */
function parsedToRaw(parsed: ParsedValue): unknown {
  switch (parsed._tag) {
    case "string":
      return parsed.value;
    case "number":
      return parseFloat(parsed.value);
    case "boolean":
      return parsed.value;
    case "null":
      return null;
    case "object": {
      const obj: Record<string, unknown> = {};
      for (const [key, value] of parsed.entries) {
        obj[key] = parsedToRaw(value);
      }
      return obj;
    }
    case "array":
      return parsed.items.map(parsedToRaw);
  }
}

/**
 * Calculate Levenshtein distance between two strings.
 */
function levenshteinDistance(a: string, b: string): number {
  const matrix: number[][] = [];

  for (let i = 0; i <= b.length; i++) {
    matrix[i] = [i];
  }

  for (let j = 0; j <= a.length; j++) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      if (b.charAt(i - 1) === a.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1, // substitution
          matrix[i][j - 1] + 1,     // insertion
          matrix[i - 1][j] + 1      // deletion
        );
      }
    }
  }

  return matrix[b.length][a.length];
}
