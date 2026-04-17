/**
 * Attribute Value Coercion
 *
 * Converts raw string attribute values from XML to typed values
 * based on the declared schema type.
 *
 * Maximally tolerant of model output variations while rejecting
 * genuinely wrong values.
 */

import type { AttributeValue } from './types'

export type CoercionResult =
  | { ok: true; value: AttributeValue }
  | { ok: false }

const TRUTHY = new Set(['true', 'True', 'TRUE', '1', 'yes', 'Yes', 'YES'])
const FALSY = new Set(['false', 'False', 'FALSE', '0', 'no', 'No', 'NO'])

export type ScalarType = 'string' | 'number' | 'boolean' | { readonly _tag: 'enum'; readonly values: readonly string[] }

/**
 * Coerce a raw string value to the expected scalar type.
 *
 * String: always succeeds (identity).
 * Number: accepts anything Number() can parse, rejects NaN/empty/Infinity.
 * Boolean: accepts common true/false representations, rejects everything else.
 * Enum: accepts any value in the enum set.
 */
export function coerceAttributeValue(raw: string, type: ScalarType): CoercionResult {
  if (typeof type === 'object' && type._tag === 'enum') {
    return type.values.includes(raw) ? { ok: true, value: raw } : { ok: false }
  }
  switch (type) {
    case 'string':
      return { ok: true, value: raw }
    case 'number': {
      if (raw === '' || raw === 'NaN' || raw === 'Infinity' || raw === '-Infinity') {
        return { ok: false }
      }
      const n = Number(raw)
      if (isNaN(n)) return { ok: false }
      return { ok: true, value: n }
    }
    case 'boolean': {
      if (TRUTHY.has(raw)) return { ok: true, value: true }
      if (FALSY.has(raw)) return { ok: true, value: false }
      return { ok: false }
    }
    default: return { ok: false }
  }
}
