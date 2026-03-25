/**
 * Attribute Validation
 *
 * Validates and coerces attribute values during parsing.
 * Called by format handlers when an attribute is finalized.
 */

import type { AttributeValue } from './types'
import type { TagSchema, ChildTagSchema } from '../execution/binding-validator'
import { coerceAttributeValue } from './coerce'

export type AttrValidationError =
  | { readonly _tag: 'UnknownAttribute' }
  | { readonly _tag: 'InvalidAttributeValue'; readonly expected: string; readonly received: string }

export type AttrValidationResult =
  | { ok: true; value: AttributeValue }
  | { ok: false; error: AttrValidationError }

export function validateToolAttr(
  tagName: string,
  schema: TagSchema,
  key: string,
  raw: string,
): AttrValidationResult {
  if (key === 'observe') return { ok: true, value: raw }
  if (tagName === 'think') return { ok: true, value: raw }

  const attrSchema = schema.attributes.get(key)
  if (!attrSchema) {
    const validAttrs = [...schema.attributes.keys()]
    return {
      ok: false,
      error: { _tag: 'UnknownAttribute' },
    }
  }

  const result = coerceAttributeValue(raw, attrSchema.type)
  if (!result.ok) {
    return {
      ok: false,
      error: { _tag: 'InvalidAttributeValue', expected: attrSchema.type, received: raw },
    }
  }

  return result
}

export function validateChildAttr(
  parentTagName: string,
  childTagName: string,
  childSchema: ChildTagSchema,
  key: string,
  raw: string,
): AttrValidationResult {
  const attrSchema = childSchema.attributes.get(key)
  if (!attrSchema) return { ok: true, value: raw }

  const result = coerceAttributeValue(raw, attrSchema.type)
  if (!result.ok) {
    return {
      ok: false,
      error: { _tag: 'InvalidAttributeValue', expected: attrSchema.type, received: raw },
    }
  }

  return result
}
