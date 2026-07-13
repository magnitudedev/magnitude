import { Effect, Schema, Option } from 'effect'
import type { CompiledField, CompiledMap, DecodedPatchOp, DecodedValue, DecodedSome, DecodedNone, Path, PatchApplyError } from './types'
import type { JsonValue } from '../schema'
import { PatchDecodeError, PatchNavigationError, PatchSchemaError, PatchEncodeError } from './types'

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
// Decode a leaf value — Effect-native, typed
// ---------------------------------------------------------------------------

function decodeLeaf(
  opValue: JsonValue,
  field: CompiledField | null,
): Effect.Effect<DecodedValue, PatchApplyError> {
  if (isScalar(opValue)) {
    if (field !== null && field.isOption) {
      return Effect.succeed(Option.some(opValue))
    }
    return Effect.succeed(opValue)
  }

  if (field !== null && field.isOption && field.innerSubSchema !== null) {
    return Schema.decode(field.innerSubSchema)(opValue).pipe(
      Effect.mapBoth({
        onFailure: (cause) => new PatchDecodeError({ path: [], cause }),
        onSuccess: (decoded) => Option.some(decoded),
      }),
    )
  }

  if (field !== null) {
    return Schema.decode(field.subSchema)(opValue).pipe(
      Effect.mapError((cause) => new PatchDecodeError({ path: [], cause })),
    )
  }

  return Effect.fail(new PatchSchemaError({ path: [] }))
}

// ---------------------------------------------------------------------------
// Unwrap Option for navigation
// ---------------------------------------------------------------------------

function unwrapOption(v: DecodedValue): DecodedValue | null {
  if (isDecodedOption(v)) {
    return v._tag === 'Some' ? v.value : null
  }
  return v
}

function toIndex(key: string | number): number {
  return typeof key === 'number' ? key : parseInt(String(key), 10)
}

// ---------------------------------------------------------------------------
// Recursive replace/add — Effect.gen
// ---------------------------------------------------------------------------

function replaceOrAdd(
  current: DecodedValue,
  path: Path,
  fullPath: Path,
  value: JsonValue,
  isAdd: boolean,
  compiled: CompiledMap<DecodedValue>,
  decodedRoot: DecodedValue,
): Effect.Effect<DecodedValue, PatchApplyError> {
  return Effect.gen(function* () {
    if (path.length === 0) {
      const field = compiled.fieldAt(fullPath, decodedRoot)
      return yield* decodeLeaf(value, field)
    }

    const key = path[0]
    const rest = path.slice(1)

    const unwrapped = unwrapOption(current)
    if (unwrapped === null) {
      // For replace ops, navigating into Option.none or null is an error
      if (!isAdd) {
        return yield* Effect.fail(new PatchNavigationError({ path: fullPath, reason: 'option_none' }))
      }
      // For add ops, auto-create container for new paths
      if (typeof key === 'number') {
        const newArr: DecodedValue[] = []
        const newChild = yield* replaceOrAdd(Option.none(), rest, fullPath, value, isAdd, compiled, decodedRoot)
        if (isAdd) {
          newArr.splice(toIndex(key), 0, newChild)
        } else {
          newArr[toIndex(key)] = newChild
        }
        const isOptionWrapped = isDecodedOption(current)
        return isOptionWrapped ? Option.some(newArr) : newArr
      }
      const newObj: Record<string, DecodedValue> = {}
      const newChild = yield* replaceOrAdd(Option.none(), rest, fullPath, value, isAdd, compiled, decodedRoot)
      newObj[String(key)] = newChild
      const isOptionWrapped = isDecodedOption(current)
      return isOptionWrapped ? Option.some(newObj) : newObj
    }

    if (!isArray(unwrapped) && !isRecord(unwrapped)) {
      return yield* Effect.fail(new PatchNavigationError({ path: fullPath, reason: 'non_container' }))
    }

    const isOptionWrapped = isDecodedOption(current)

    if (rest.length === 0) {
      // AT LEAF — apply the operation
      const field = compiled.fieldAt(fullPath, decodedRoot)
      const decodedValue = yield* decodeLeaf(value, field)

      if (isArray(unwrapped)) {
        const idx = toIndex(key)
        const newArr = isAdd
          ? [...unwrapped.slice(0, idx), decodedValue, ...unwrapped.slice(idx)]
          : [...unwrapped.slice(0, idx), decodedValue, ...unwrapped.slice(idx + 1)]
        return isOptionWrapped ? Option.some(newArr) : newArr
      }

      const newObj = { ...unwrapped, [String(key)]: decodedValue }
      return isOptionWrapped ? Option.some(newObj) : newObj
    }

    // DESCEND — recurse into the child, then rebuild
    const child: DecodedValue = isArray(unwrapped)
      ? (unwrapped[toIndex(key)] ?? Option.none())
      : (unwrapped[String(key)] ?? Option.none())

    const newChild = yield* replaceOrAdd(child, rest, fullPath, value, isAdd, compiled, decodedRoot)

    const rebuilt: DecodedValue = isArray(unwrapped)
      ? [...unwrapped.slice(0, toIndex(key)), newChild, ...unwrapped.slice(toIndex(key) + 1)]
      : { ...unwrapped, [String(key)]: newChild }

    return isOptionWrapped ? Option.some(rebuilt) : rebuilt
  })
}

// ---------------------------------------------------------------------------
// Recursive remove — Effect.gen
// ---------------------------------------------------------------------------

function removeAt(
  current: DecodedValue,
  path: Path,
  fullPath: Path,
  compiled: CompiledMap<DecodedValue>,
  decodedRoot: DecodedValue,
): Effect.Effect<DecodedValue, PatchApplyError> {
  return Effect.gen(function* () {
    if (path.length === 0) {
      return yield* Effect.fail(new PatchNavigationError({ path: fullPath, reason: 'cannot_remove_root' }))
    }

    const key = path[0]
    const rest = path.slice(1)

    const unwrapped = unwrapOption(current)
    if (unwrapped === null) {
      return yield* Effect.fail(new PatchNavigationError({ path: fullPath, reason: 'option_none' }))
    }

    if (!isArray(unwrapped) && !isRecord(unwrapped)) {
      return yield* Effect.fail(new PatchNavigationError({ path: fullPath, reason: 'non_container' }))
    }

    const isOptionWrapped = isDecodedOption(current)

    if (rest.length === 0) {
      // AT LEAF — apply removal
      if (isArray(unwrapped)) {
        const idx = toIndex(key)
        const newArr = [...unwrapped.slice(0, idx), ...unwrapped.slice(idx + 1)]
        return isOptionWrapped ? Option.some(newArr) : newArr
      }

      // Object removal — resolve field metadata at the leaf
      const field = compiled.fieldAt(fullPath, decodedRoot)
      if (field !== null) {
        if (field.isOption) {
          const newObj = { ...unwrapped, [String(key)]: Option.none() }
          return isOptionWrapped ? Option.some(newObj) : newObj
        } else if (field.hasDefault) {
          const newObj = { ...unwrapped, [String(key)]: field.defaultValue }
          return isOptionWrapped ? Option.some(newObj) : newObj
        }
      }

      // Plain field or record entry — delete key
      const newObj = { ...unwrapped }
      delete newObj[String(key)]
      return isOptionWrapped ? Option.some(newObj) : newObj
    }

    // DESCEND — recurse, then rebuild
    const child: DecodedValue = isArray(unwrapped)
      ? (unwrapped[toIndex(key)] ?? Option.none())
      : (unwrapped[String(key)] ?? Option.none())

    const newChild = yield* removeAt(child, rest, fullPath, compiled, decodedRoot)

    const rebuilt: DecodedValue = isArray(unwrapped)
      ? [...unwrapped.slice(0, toIndex(key)), newChild, ...unwrapped.slice(toIndex(key) + 1)]
      : { ...unwrapped, [String(key)]: newChild }

    return isOptionWrapped ? Option.some(rebuilt) : rebuilt
  })
}

// ---------------------------------------------------------------------------
// Move — Effect.gen
// ---------------------------------------------------------------------------

function encodeForMove(
  compiled: CompiledMap<DecodedValue>,
  targetPath: Path,
  value: DecodedValue,
  decodedRoot: DecodedValue,
): Effect.Effect<JsonValue, PatchApplyError> {
  const subSchema = compiled.subSchemaAt(targetPath, decodedRoot)
  if (subSchema !== null) {
    return Schema.encode(subSchema)(value).pipe(
      Effect.mapError((cause) => new PatchEncodeError({ path: targetPath, cause })),
    )
  }
  return Effect.fail(new PatchSchemaError({ path: targetPath }))
}

function moveOp(
  root: DecodedValue,
  from: Path,
  to: Path,
  compiled: CompiledMap<DecodedValue>,
  decodedRoot: DecodedValue,
): Effect.Effect<DecodedValue, PatchApplyError> {
  return Effect.gen(function* () {
    // Extract the value at 'from' by navigating
    let value: DecodedValue = root
    for (const key of from) {
      const unwrapped = unwrapOption(value)
      if (unwrapped === null || (!isArray(unwrapped) && !isRecord(unwrapped))) {
        return yield* Effect.fail(new PatchNavigationError({ path: from, reason: 'move_source_missing' }))
      }
      value = isArray(unwrapped)
        ? (unwrapped[toIndex(key)] ?? Option.none())
        : (unwrapped[String(key)] ?? Option.none())
    }
    const extracted = unwrapOption(value)
    if (extracted === null) {
      return yield* Effect.fail(new PatchNavigationError({ path: from, reason: 'move_source_none' }))
    }

    // Remove from source
    const afterRemove = yield* removeAt(root, from, from, compiled, decodedRoot)

    // Add at target
    const encoded = isScalar(extracted) ? extracted : yield* encodeForMove(compiled, to, extracted, afterRemove)
    return yield* replaceOrAdd(afterRemove, to, to, encoded, true, compiled, afterRemove)
  })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

function applyOp(
  root: DecodedValue,
  op: DecodedPatchOp,
  compiled: CompiledMap<DecodedValue>,
  decodedRoot: DecodedValue,
): Effect.Effect<DecodedValue, PatchApplyError> {
  switch (op.op) {
    case 'replace':
      return replaceOrAdd(root, op.path, op.path, op.value, false, compiled, root)
    case 'add':
      return replaceOrAdd(root, op.path, op.path, op.value, true, compiled, root)
    case 'remove':
      return removeAt(root, op.path, op.path, compiled, root)
    case 'move':
      return moveOp(root, op.from, op.to, compiled, root)
  }
}

export function applyDecodedPatch<A extends DecodedValue>(
  prev: A,
  ops: readonly DecodedPatchOp[],
  compiled: CompiledMap<DecodedValue>,
): Effect.Effect<DecodedValue, PatchApplyError> {
  return Effect.gen(function* () {
    let current: DecodedValue = prev
    for (const op of ops) {
      current = yield* applyOp(current, op, compiled, current)
    }
    return current
  })
}
