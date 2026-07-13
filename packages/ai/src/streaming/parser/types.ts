/**
 * Types for the new JSON streaming parser.
 * Modeled after xml-act's type system: frame-level dispatch is compile-time safe
 * via discriminant narrowing in resolveHandler. Token dispatch within handlers
 * is runtime-checked (handlers receive the full JsonToken union).
 */

import type { Op } from './machine'

// ---------------------------------------------------------------------------
// Token types — emitted by the tokenizer
// ---------------------------------------------------------------------------

export type JsonToken =
  | { readonly _tag: "objectOpen" }
  | { readonly _tag: "objectClose" }
  | { readonly _tag: "arrayOpen" }
  | { readonly _tag: "arrayClose" }
  | { readonly _tag: "colon" }
  | { readonly _tag: "comma" }
  | { readonly _tag: "string"; readonly value: string; readonly complete: boolean }
  | { readonly _tag: "number"; readonly value: string; readonly complete: boolean }
  | { readonly _tag: "true" }
  | { readonly _tag: "false" }
  | { readonly _tag: "null" }
  | { readonly _tag: "unquotedString"; readonly value: string; readonly complete: boolean }

// ---------------------------------------------------------------------------
// Pending token state — exposed by tokenizer for partial tree reconstruction
// ---------------------------------------------------------------------------

export type PendingToken =
  | { readonly _tag: "string"; readonly content: string }
  | { readonly _tag: "number"; readonly content: string }
  | { readonly _tag: "keyword"; readonly content: string }
  | { readonly _tag: "unquoted"; readonly content: string }

// ---------------------------------------------------------------------------
// Tokenizer interface
// ---------------------------------------------------------------------------

export interface JsonTokenizer {
  push(chunk: string): void
  end(): void
  readonly pending: PendingToken | null
}

// ---------------------------------------------------------------------------
// Frame types — parser stack frames
// ---------------------------------------------------------------------------

export interface RootFrame {
  readonly type: "root"
  readonly value: import('../types').ParsedValue | undefined
}

export interface ObjectFrame {
  readonly type: "object"
  readonly keys: string[]
  readonly values: import('../types').ParsedValue[]
  readonly phase: "expectKey" | "expectColon" | "expectValue" | "afterValue"
}

export interface ArrayFrame {
  readonly type: "array"
  readonly items: import('../types').ParsedValue[]
  readonly phase: "expectValue" | "afterValue"
}

export type JsonFrame = RootFrame | ObjectFrame | ArrayFrame

// ---------------------------------------------------------------------------
// Event type — emitted by parser
// ---------------------------------------------------------------------------

export type JsonEvent = {
  readonly _tag: "value"
  readonly value: import('../types').ParsedValue
}

// ---------------------------------------------------------------------------
// Op alias
// ---------------------------------------------------------------------------

export type JsonOp = Op<JsonFrame, JsonEvent>

// ---------------------------------------------------------------------------
// Typed handler interfaces
// ---------------------------------------------------------------------------

export interface JsonParserContext {
  readonly tokenizer: JsonTokenizer
  readonly peekParent: () => JsonFrame | undefined
}

export interface TokenHandler<TFrame extends JsonFrame> {
  handle(token: JsonToken, frame: TFrame, ctx: JsonParserContext): JsonOp[]
}

export interface BoundTokenHandler {
  handle(token: JsonToken, ctx: JsonParserContext): JsonOp[]
}

export function bindHandler<TFrame extends JsonFrame>(
  handler: TokenHandler<TFrame>,
  frame: TFrame,
): BoundTokenHandler {
  return {
    handle: (token, ctx) => handler.handle(token, frame, ctx),
  }
}

// ---------------------------------------------------------------------------
// Parser interface
// ---------------------------------------------------------------------------

export interface JsonParser {
  feed(token: JsonToken): void
  end(): void
  readonly partial: import('../types').ParsedValue | undefined
  readonly currentPath: readonly string[]
}
