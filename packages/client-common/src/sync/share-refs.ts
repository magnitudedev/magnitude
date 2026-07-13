/**
 * shareRefs — Generic structural sharing.
 *
 * Walks `next` recursively, comparing to `old`. Returns a value with the same
 * shape as `next` but with stable references for unchanged parts.
 *
 * Rules:
 *   - Same reference → return old (already shared)
 *   - Scalar → === old ? old : next
 *   - Array with `_key` elements → match old↔new by _key, reuse old refs for unchanged items
 *   - Array without `_key` elements → compare by index
 *   - Plain object → recurse field by field, reuse old object if all fields equal
 *   - Other (class instances, etc.) → ref comparison
 *
 * If the entire value is equal to old, returns old (same top-level ref).
 *
 * `_key` is an explicit contract: the store layer's `transform` callback is
 * responsible for injecting `_key` onto objects that need identity-based
 * matching. `shareRefs` checks for it — it does not sniff for arbitrary fields.
 *
 * The optional `transform` callback runs on each plain object during the walk,
 * before comparison. This is where `_key` injection happens — in the same pass
 * as the comparison, not a separate walk.
 */

function isPlainObject(v: unknown): v is Record<string, unknown> {
  if (v === null || typeof v !== 'object') return false
  const proto = Object.getPrototypeOf(v)
  return proto === Object.prototype || proto === null
}

export type Transform = (obj: Record<string, unknown>) => Record<string, unknown>

export function shareRefs<T>(old: T | undefined, next: T, transform?: Transform): T {
  if (old === undefined) return next
  if (old === next) return old

  // Scalars
  if (typeof next !== 'object' || next === null) {
    return old === next ? old : next
  }

  // Arrays
  if (Array.isArray(next)) {
    return shareArray(old as unknown[], next, transform) as T
  }

  // Plain objects
  if (isPlainObject(next)) {
    // Apply transform before comparison (e.g. _key injection)
    const transformed = transform ? transform(next as Record<string, unknown>) : next
    return shareObject(old as Record<string, unknown> | undefined, transformed as Record<string, unknown>, transform) as T
  }

  // Non-plain objects (Date, class instances, etc.) — compare by ref
  return old === next ? old : next
}

function hasKey(item: unknown): item is { _key: string } {
  return (
    item !== null &&
    typeof item === 'object' &&
    '_key' in item &&
    typeof (item as { _key: unknown })._key === 'string'
  )
}

function shareArray(old: unknown[] | undefined | null, next: unknown[], transform?: Transform): unknown[] {
  if (old === undefined || old === null) return next
  if (old === next) return old
  if (old.length === 0 && next.length === 0) return old

  // Apply transform to array elements (in-place check, only spreads if needed)
  let transformed = next
  if (transform) {
    let changed = false
    const result = next.map(item => {
      if (isPlainObject(item)) {
        const t = transform(item)
        if (t !== item) changed = true
        return t
      }
      return item
    })
    if (changed) transformed = result
  }

  // If elements have `_key`, match by key — handles reordering, removal, append
  if (transformed.length > 0 && hasKey(transformed[0])) {
    return shareArrayByKey(old, transformed, transform)
  }

  // Index-based comparison
  let changed = false
  const result: unknown[] = []
  for (let i = 0; i < transformed.length; i++) {
    const shared = shareRefs(old[i], transformed[i], transform)
    if (shared !== old[i]) changed = true
    result.push(shared)
  }
  if (old.length !== transformed.length) changed = true
  return changed ? result : old
}

function shareArrayByKey(old: unknown[], next: unknown[], transform?: Transform): unknown[] {
  const oldByKey = new Map<string, unknown>()
  for (const item of old) {
    if (hasKey(item)) oldByKey.set(item._key, item)
  }

  let changed = false
  const result: unknown[] = []

  for (let i = 0; i < next.length; i++) {
    const newItem = next[i]
    const key = (newItem as { _key: string })._key
    const oldItem = oldByKey.get(key)
    // Pass transform through so nested objects get _key injection too
    const shared = shareRefs(oldItem, newItem, transform)
    if (shared !== oldItem) changed = true
    result.push(shared)
    // Check order: is this the same key as old[i]?
    const oldAtIdx = old[i]
    if (!hasKey(oldAtIdx) || oldAtIdx._key !== key) changed = true
  }

  if (old.length !== next.length) changed = true
  return changed ? result : old
}

function shareObject(
  old: Record<string, unknown> | undefined,
  next: Record<string, unknown>,
  transform?: Transform,
): Record<string, unknown> {
  if (old === undefined || old === null || !isPlainObject(old)) return next
  if (old === next) return old

  let changed = false
  const result: Record<string, unknown> = {}
  const oldKeys = new Set(Object.keys(old))

  for (const key of Object.keys(next)) {
    const shared = shareRefs(old[key], next[key], transform)
    if (shared !== old[key]) changed = true
    result[key] = shared
    oldKeys.delete(key)
  }

  // If old had keys that next doesn't, shape changed
  if (oldKeys.size > 0) changed = true

  return changed ? result : old
}
