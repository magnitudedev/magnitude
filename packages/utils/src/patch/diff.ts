import { Effect, Schema } from 'effect'
import type { CompiledMap, DecodedPatchOp, DecodedValue, DecodedSome, DecodedNone, Path, PatchApplyError } from './types'
import type { JsonValue } from '../schema'
import { PatchEncodeError, PatchSchemaError } from './types'

// ---------------------------------------------------------------------------
// Type guards — proper narrowing, no `as`, no `unknown`
// ---------------------------------------------------------------------------

function isScalar(v: DecodedValue): v is string | number | boolean | null {
  return v === null || typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean'
}

function isDecodedOption(v: DecodedValue): v is DecodedSome | DecodedNone {
  return typeof v === 'object' && v !== null && !Array.isArray(v) && '_tag' in v && (v._tag === 'Some' || v._tag === 'None')
}

function isRecord(v: DecodedValue): v is { readonly [key: string]: DecodedValue } {
  return typeof v === 'object' && v !== null && !Array.isArray(v) && !isDecodedOption(v)
}

function isArray(v: DecodedValue): v is readonly DecodedValue[] {
  return Array.isArray(v)
}

// ---------------------------------------------------------------------------
// Encoding values for transport — Effect-native
// ---------------------------------------------------------------------------

function encodeValue(
  compiled: CompiledMap<DecodedValue>,
  path: Path,
  value: DecodedValue,
  decodedRoot: DecodedValue,
): Effect.Effect<JsonValue, PatchApplyError> {
  if (isScalar(value)) return Effect.succeed(value)
  const subSchema = compiled.subSchemaAt(path, decodedRoot)
  if (subSchema === null) {
    return Effect.fail(new PatchSchemaError({ path }))
  }
  return Schema.encode(subSchema)(value).pipe(
    Effect.mapError((cause) => new PatchEncodeError({ path, cause })),
  )
}

function encodeInnerValue(
  compiled: CompiledMap<DecodedValue>,
  path: Path,
  value: DecodedValue,
  decodedRoot: DecodedValue,
): Effect.Effect<JsonValue, PatchApplyError> {
  if (isScalar(value)) return Effect.succeed(value)
  const field = compiled.fieldAt(path, decodedRoot)
  if (field !== null && field.innerSubSchema !== null) {
    return Schema.encode(field.innerSubSchema)(value).pipe(
      Effect.mapError((cause) => new PatchEncodeError({ path, cause })),
    )
  }
  return encodeValue(compiled, path, value, decodedRoot)
}

// ---------------------------------------------------------------------------
// Core diff — Effect.gen, threads decodedRoot for union resolution
// ---------------------------------------------------------------------------

function diffRec(
  prev: DecodedValue,
  next: DecodedValue,
  path: Path,
  ops: DecodedPatchOp[],
  compiled: CompiledMap<DecodedValue>,
  decodedRoot: DecodedValue,
): Effect.Effect<void, PatchApplyError> {
  return Effect.gen(function* () {
    // Both Option
    if (isDecodedOption(prev) && isDecodedOption(next)) {
      if (prev._tag === 'Some' && next._tag === 'Some') {
        yield* diffRec(prev.value, next.value, path, ops, compiled, decodedRoot)
      } else if (prev._tag === 'Some' && next._tag === 'None') {
        ops.push({ op: 'remove', path })
      } else if (prev._tag === 'None' && next._tag === 'Some') {
        const encoded = yield* encodeInnerValue(compiled, path, next.value, decodedRoot)
        ops.push({ op: 'replace', path, value: encoded })
      }
      return
    }

    // One is Option, other is not → replace
    if (isDecodedOption(prev) !== isDecodedOption(next)) {
      const encoded = yield* encodeValue(compiled, path, next, decodedRoot)
      ops.push({ op: 'replace', path, value: encoded })
      return
    }

    // Both scalars
    if (isScalar(prev) && isScalar(next)) {
      if (prev !== next) {
        ops.push({ op: 'replace', path, value: next })
      }
      return
    }

    // Both arrays
    if (isArray(prev) && isArray(next)) {
      const minLen = Math.min(prev.length, next.length)
      for (let i = 0; i < minLen; i++) {
        if (prev[i] !== next[i]) {
          yield* diffRec(prev[i], next[i], [...path, i], ops, compiled, decodedRoot)
        }
      }
      for (let i = minLen; i < next.length; i++) {
        const encoded = yield* encodeValue(compiled, [...path, i], next[i], decodedRoot)
        ops.push({ op: 'add', path: [...path, i], value: encoded })
      }
      for (let i = prev.length - 1; i >= next.length; i--) {
        ops.push({ op: 'remove', path: [...path, i] })
      }
      return
    }

    // Both objects
    if (isRecord(prev) && isRecord(next)) {
      const prevKeys = Object.keys(prev)
      const nextKeys = Object.keys(next)

      for (const key of nextKeys) {
        const nextVal = next[key]
        if (!(key in prev)) {
          const encoded = yield* encodeValue(compiled, [...path, key], nextVal, decodedRoot)
          ops.push({ op: 'replace', path: [...path, key], value: encoded })
        } else if (prev[key] !== nextVal) {
          yield* diffRec(prev[key], nextVal, [...path, key], ops, compiled, decodedRoot)
        }
      }
      for (const key of prevKeys) {
        if (!(key in next)) {
          ops.push({ op: 'remove', path: [...path, key] })
        }
      }
      return
    }

    // Different types → replace with encoded value
    const encoded = yield* encodeValue(compiled, path, next, decodedRoot)
    ops.push({ op: 'replace', path, value: encoded })
  })
}

// ---------------------------------------------------------------------------
// Public API — generic over A (the decoded type), constrained to DecodedValue
// ---------------------------------------------------------------------------

export function diffDecoded<A extends DecodedValue>(
  prev: A,
  next: A,
  compiled: CompiledMap<DecodedValue>,
): Effect.Effect<readonly DecodedPatchOp[], PatchApplyError> {
  return Effect.gen(function* () {
    const ops: DecodedPatchOp[] = []
    yield* diffRec(prev, next, [], ops, compiled, prev)
    return ops
  })
}
