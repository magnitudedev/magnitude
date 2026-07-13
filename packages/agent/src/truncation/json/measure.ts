import { CHARS_PER_TOKEN_UPPER } from '../../constants'
import { charsToTokensUpper } from '../budget'
import type { Measurement } from '../budget'
import type { JsonValue } from '@magnitudedev/ai'

/**
 * Count the JSON-escaped length of a single character by char code.
 * Conservative: overcounts surrogate pairs but NEVER undercounts.
 */
export function jsonEscapedCharLen(code: number): number {
  if (code === 0x22 || code === 0x5C) return 2
  if (code < 0x20) {
    if (code === 0x08 || code === 0x09 || code === 0x0A || code === 0x0C || code === 0x0D) return 2
    return 6
  }
  if (code >= 0xD800 && code <= 0xDFFF) return 6
  return 1
}

/**
 * Measure the serialized size of a value in tokens, stopping early if it exceeds the cap.
 *
 * Why bounded? If budget is 500 tokens and value is 50k tokens, we only need to know
 * "too big" - not exactly how big. This makes measurement O(min(size, cap)).
 */
export function measureBounded(value: JsonValue, capTokens: number): Measurement {
  const capChars = capTokens * CHARS_PER_TOKEN_UPPER
  let count = 0

  function measure(v: unknown): boolean {
    if (count > capChars) return false

    if (v === null) {
      count += 4
      return true
    }
    if (v === undefined) {
      count += 9
      return true
    }
    if (typeof v === 'boolean') {
      count += v ? 4 : 5
      return true
    }
    if (typeof v === 'number') {
      count += String(v).length
      return true
    }
    if (typeof v === 'string') {
      count += 2
      for (let j = 0; j < v.length; j++) {
        count += jsonEscapedCharLen(v.charCodeAt(j))
        if (count > capChars) return false
      }
      return count <= capChars
    }

    if (Array.isArray(v)) {
      count += 2
      for (let i = 0; i < v.length; i++) {
        if (i > 0) count += 2
        if (!measure(v[i])) return false
      }
      return count <= capChars
    }

    if (typeof v === 'object') {
      count += 2
      const entries = Object.entries(v)
      for (let i = 0; i < entries.length; i++) {
        if (i > 0) count += 2
        count += entries[i][0].length + 2
        if (!measure(entries[i][1])) return false
      }
      return count <= capChars
    }

    return true
  }

  const completed = measure(value)
  return {
    size: charsToTokensUpper(Math.min(count, capChars)),
    exceeded: !completed || count > capChars
  }
}
