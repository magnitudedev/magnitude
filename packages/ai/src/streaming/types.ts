/**
 * Types for the streaming field parser module.
 *
 * Internal types (ParsedValue, etc.) are not exported from the package.
 * Public types (FieldEvent, StreamingPartial, StreamingLeaf) are exported via index.ts.
 */

import type { JsonValue } from "../prompt/parts"

// ---------------------------------------------------------------------------
// Internal types — used by json-parser and field-parser, not exported publicly
// ---------------------------------------------------------------------------

export type CompletionState = "complete" | "incomplete"

export type ParsedValue =
  | ParsedString
  | ParsedNumber
  | ParsedBoolean
  | ParsedNull
  | ParsedObject
  | ParsedArray

export interface ParsedString {
  readonly _tag: "string"
  readonly value: string
  readonly state: CompletionState
}

export interface ParsedNumber {
  readonly _tag: "number"
  readonly value: string
  readonly state: CompletionState
}

export interface ParsedBoolean {
  readonly _tag: "boolean"
  readonly value: boolean
  readonly state: "complete"
}

export interface ParsedNull {
  readonly _tag: "null"
  readonly state: "complete"
}

export interface ParsedObject {
  readonly _tag: "object"
  readonly entries: Array<[string, ParsedValue]>
  readonly state: CompletionState
}

export interface ParsedArray {
  readonly _tag: "array"
  readonly items: ParsedValue[]
  readonly state: CompletionState
}

export type JsonCollection =
  | ObjectCollection
  | ArrayCollection
  | QuotedStringCollection
  | UnquotedStringCollection

export interface ObjectCollection {
  readonly _tag: "object"
  keys: string[]
  values: ParsedValue[]
  state: CompletionState
}

export interface ArrayCollection {
  readonly _tag: "array"
  items: ParsedValue[]
  state: CompletionState
}

export interface QuotedStringCollection {
  readonly _tag: "quotedString"
  content: string
  state: CompletionState
  trailingBackslashes: number
  unescapedQuoteCount: number
  pendingEscape: boolean
  pendingUnicodeHex: string | null
}

export interface UnquotedStringCollection {
  readonly _tag: "unquotedString"
  content: string
  state: CompletionState
}

export type CloseStringResult =
  | { readonly _tag: "close"; readonly charsConsumed: number; readonly completion: CompletionState }
  | { readonly _tag: "continue"; readonly charsConsumed: number }

export type Pos =
  | { readonly _tag: "inNothing" }
  | { readonly _tag: "unknown" }
  | { readonly _tag: "inObjectKey" }
  | { readonly _tag: "inObjectValue" }
  | { readonly _tag: "inArray" }

// ---------------------------------------------------------------------------
// Internal parser interface — used by field-parser, not exported publicly
// ---------------------------------------------------------------------------

export interface IncrementalJsonParser {
  push(chunk: string): void
  end(): void
  readonly partial: ParsedValue | undefined
  readonly done: boolean
  readonly currentPath: readonly string[]
}

// ---------------------------------------------------------------------------
// Public types — exported from the package
// ---------------------------------------------------------------------------

/** Field-level events produced by the streaming field parser on each push/end. */
export type FieldEvent =
  | { readonly _tag: "field_start"; readonly path: readonly string[] }
  | { readonly _tag: "field_delta"; readonly path: readonly string[]; readonly delta: string }
  | { readonly _tag: "field_end"; readonly path: readonly string[]; readonly value: JsonValue }

/** A streaming leaf value — discriminated on finality. */
export type StreamingLeaf<T> =
  | { readonly isFinal: true; readonly value: T }
  | { readonly isFinal: false; readonly value: string }

/**
 * Transforms a type into its streaming partial shape.
 * Fields arrive incrementally — scalars are StreamingLeaf, objects are partial, arrays accumulate.
 */
export type StreamingPartial<T> = {
  [K in keyof T]?: T[K] extends ReadonlyArray<infer E>
    ? Array<E extends Record<string, unknown> ? StreamingPartial<E> : StreamingLeaf<E>>
    : T[K] extends Array<infer E>
      ? Array<E extends Record<string, unknown> ? StreamingPartial<E> : StreamingLeaf<E>>
      : T[K] extends Record<string, unknown>
        ? StreamingPartial<T[K]>
        : StreamingLeaf<T[K]>
}
