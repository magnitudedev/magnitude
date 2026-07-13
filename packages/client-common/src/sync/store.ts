/**
 * ReferencePreservingStore — a store where `set(fullNewState)` preserves
 * references to every unchanged part via shareRefs.
 *
 * If nothing changed, set() returns early — no notification, no re-render.
 * Only subscribers whose slice's reference changed will re-render.
 *
 * Optional `transform` is passed to shareRefs and runs on each plain object
 * during the single walk — used by the display store to inject `_key` for
 * identity-based array matching. No separate preprocessing pass.
 */

import { shareRefs, type Transform } from './share-refs'

type Listener = () => void

/**
 * Recursively apply transform to every plain object in the value.
 * Used for initial state only — subsequent sets go through shareRefs which
 * applies transform during the comparison walk (single pass).
 */
function applyTransform<T>(value: T, transform: Transform): T {
  if (value === null || typeof value !== 'object') return value
  if (Array.isArray(value)) {
    let changed = false
    const result = value.map(item => {
      const t = applyTransform(item, transform)
      if (t !== item) changed = true
      return t
    })
    return changed ? result as T : value
  }
  const proto = Object.getPrototypeOf(value)
  if (proto !== Object.prototype && proto !== null) return value
  // It's a plain object — apply transform, then recurse into values
  let transformed = transform(value as Record<string, unknown>)
  let changed = transformed !== value
  const result: Record<string, unknown> = {}
  for (const key of Object.keys(transformed)) {
    const t = applyTransform((transformed as Record<string, unknown>)[key], transform)
    if (t !== (transformed as Record<string, unknown>)[key]) changed = true
    result[key] = t
  }
  return changed ? result as T : value
}

export class ReferencePreservingStore<T> {
  private state: T
  private listeners = new Set<Listener>()
  private transform: Transform | undefined

  constructor(initial: T, transform?: Transform) {
    // Transform initial state — shareRefs returns next as-is when old is undefined,
    // so we need to apply transform manually for the initial state
    if (transform) {
      this.state = applyTransform(initial, transform)
    } else {
      this.state = initial
    }
    this.transform = transform
  }

  get = (): T => this.state

  set = (next: T): void => {
    const shared = shareRefs(this.state, next, this.transform)
    if (shared === this.state) return // no-op — nothing changed
    this.state = shared
    for (const l of this.listeners) l()
  }

  /** Update state via a function. Useful for local mutations like appending an error. */
  update = (fn: (prev: T) => T): void => {
    this.set(fn(this.state))
  }

  subscribe = (listener: Listener): (() => void) => {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }
}
