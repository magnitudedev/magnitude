import { Data, ParseResult, Schema } from 'effect'
import type { JsonValue } from '../schema'

// ---------------------------------------------------------------------------
// Decoded value — recursive type for values in the decoded tree
// ---------------------------------------------------------------------------

/**
 * A value in the decoded tree. By JsonSafeSchema construction, every field
 * is either a `JsonValue` scalar/container or an `Option` wrapping one.
 *
 * `JsonValue` alone is insufficient because decoded objects contain Option
 * fields (e.g. `{ id: string, label: Option<string> }`) — `Option<JsonValue>`
 * does not satisfy `JsonValue`'s `{ readonly [key: string]: JsonValue }`
 * index signature. This recursive type allows Options at any depth.
 *
 * Interfaces are used for indirect recursion (TypeScript restriction).
 * `DecodedSome` and `DecodedNone` are structurally identical to Effect's
 * `Option.Some` and `Option.None` — `Option.isOption` works at runtime.
 */
interface DecodedArray extends ReadonlyArray<DecodedValue> {}

interface DecodedRecord {
  readonly [key: string]: DecodedValue
}

export interface DecodedSome {
  readonly _tag: 'Some'
  readonly value: DecodedValue
}

export interface DecodedNone {
  readonly _tag: 'None'
}

export type DecodedValue =
  | string
  | number
  | boolean
  | null
  | DecodedArray
  | DecodedRecord
  | DecodedSome
  | DecodedNone

// ---------------------------------------------------------------------------
// Path
// ---------------------------------------------------------------------------

export type Path = readonly (string | number)[]

// ---------------------------------------------------------------------------
// Patch operations (decoded-level, path arrays not JSON Pointer strings)
// ---------------------------------------------------------------------------

export type DecodedPatchOp =
  | { readonly op: 'replace'; readonly path: Path; readonly value: JsonValue }
  | { readonly op: 'remove'; readonly path: Path }
  | { readonly op: 'add'; readonly path: Path; readonly value: JsonValue }
  | { readonly op: 'move'; readonly from: Path; readonly to: Path }

// ---------------------------------------------------------------------------
// Sub-schema type
// ---------------------------------------------------------------------------

export type JsonSubSchema = Schema.Schema<DecodedValue, JsonValue, never>
export type InnerSubSchema = JsonSubSchema

// ---------------------------------------------------------------------------
// Compiled field metadata
// ---------------------------------------------------------------------------

export interface CompiledField {
  readonly isOption: boolean
  readonly hasDefault: boolean
  readonly defaultValue: JsonValue
  readonly subSchema: JsonSubSchema
  readonly innerSubSchema: InnerSubSchema | null
}

// ---------------------------------------------------------------------------
// Compiled map — generic over the decoded type A
// ---------------------------------------------------------------------------

export interface CompiledMap<A = DecodedValue> {
  fieldAt(path: Path, decodedRoot?: A): CompiledField | null
  subSchemaAt(path: Path, decodedRoot?: A): JsonSubSchema | null
}

// ---------------------------------------------------------------------------
// Errors — distinct tagged errors per failure category
// ---------------------------------------------------------------------------

/** Schema encoding failed at a path during diff. */
export class PatchEncodeError extends Data.TaggedError('PatchEncodeError')<{
  readonly path: Path
  readonly cause: ParseResult.ParseError
}> {}

/** Schema decoding failed at a path during apply. */
export class PatchDecodeError extends Data.TaggedError('PatchDecodeError')<{
  readonly path: Path
  readonly cause: ParseResult.ParseError
}> {}

/** Structural navigation failure — the path is valid but the value can't be traversed. */
export class PatchNavigationError extends Data.TaggedError('PatchNavigationError')<{
  readonly path: Path
  readonly reason:
    | 'option_none'
    | 'null'
    | 'non_container'
    | 'cannot_remove_root'
    | 'move_source_missing'
    | 'move_source_none'
}> {}

/** No sub-schema resolved at a path — the compiled map has no field info. */
export class PatchSchemaError extends Data.TaggedError('PatchSchemaError')<{
  readonly path: Path
}> {}

/** Union of all patch errors. */
export type PatchApplyError =
  | PatchEncodeError
  | PatchDecodeError
  | PatchNavigationError
  | PatchSchemaError
