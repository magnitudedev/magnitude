/**
 * Output normalization helpers for model responses.
 * Converts curly single/double quotes to straight ASCII quotes.
 */

const CURLY_SINGLE_REGEX = /[‘’]/g
const CURLY_DOUBLE_REGEX = /[“”]/g

export function normalizeQuotesInString(value: string): string {
  if (value.indexOf('‘') === -1 && value.indexOf('’') === -1 && value.indexOf('“') === -1 && value.indexOf('”') === -1) {
    return value
  }
  return value.replace(CURLY_SINGLE_REGEX, "'").replace(CURLY_DOUBLE_REGEX, '"')
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object') return false
  const proto = Object.getPrototypeOf(value)
  return proto === Object.prototype || proto === null
}

export function normalizeModelOutput<T>(value: T): T {
  const seen = new WeakMap<object, unknown>()

  function walk(v: unknown): unknown {
    if (typeof v === 'string') return normalizeQuotesInString(v)
    if (Array.isArray(v)) {
      if (seen.has(v)) return seen.get(v)!
      const out: unknown[] = []
      seen.set(v, out)
      for (const item of v) out.push(walk(item))
      return out
    }
    if (!v || typeof v !== 'object') return v
    if (!isPlainObject(v)) return v

    if (seen.has(v)) return seen.get(v)!
    const out: Record<string, unknown> = {}
    seen.set(v, out)

    for (const [k, child] of Object.entries(v)) {
      out[k] = walk(child)
    }
    return out
  }

  return walk(value) as T
}
