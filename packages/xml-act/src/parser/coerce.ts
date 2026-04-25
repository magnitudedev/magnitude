/**
 * Scalar value coercion — converts raw string parameter values to typed values.
 */

import type { FieldType } from './types'

export function coerceScalarValue(rawValue: string, fieldType: FieldType): { value: unknown; ok: boolean } {
  const trimmed = rawValue.trim()
  switch (fieldType) {
    case 'string':
      return { value: rawValue, ok: true }
    case 'number': {
      const num = parseFloat(trimmed)
      if (isNaN(num)) return { value: rawValue, ok: false }
      return { value: num, ok: true }
    }
    case 'boolean': {
      const lower = trimmed.toLowerCase()
      if (lower === 'true' || lower === '1') return { value: true, ok: true }
      if (lower === 'false' || lower === '0') return { value: false, ok: true }
      return { value: rawValue, ok: false }
    }
    default:
      return { value: rawValue, ok: true }
  }
}
