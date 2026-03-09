/**
 * Attribute Validation
 *
 * Validates and coerces attribute values during parsing.
 * Called by the streaming parser when an attribute is finalized.
 *
 * Returns either a coerced value or a ParseErrorDetail to emit.
 */

import type { AttributeValue, BaseToolParseErrorDetail } from './types'
import type { TagSchema, ChildTagSchema } from '../execution/binding-validator'
import { coerceAttributeValue } from './coerce'

export type AttrValidationResult =
  | { ok: true; value: AttributeValue }
  | { ok: false; error: BaseToolParseErrorDetail }

/**
 * Validate and coerce a top-level tool attribute.
 *
 * - 'id' is always valid (RefStore convention)
 * - Unknown attributes → UnknownAttribute error
 * - Invalid values → InvalidAttributeValue error
 * - Valid values → coerced to declared type
 */
export function validateToolAttr(
  tagName: string,
  schema: TagSchema,
  key: string,
  raw: string,
): AttrValidationResult {
  // 'id' is always valid (used by RefStore)
  if (key === 'id') return { ok: true, value: raw }

  // about attr on think
  if (tagName === 'think') return { ok: true, value: raw }

  const attrSchema = schema.attributes.get(key)
  if (!attrSchema) {
    const validAttrs = [...schema.attributes.keys()]
    return {
      ok: false,
      error: {
        _tag: 'UnknownAttribute',
        attribute: key,
        detail: `Attribute '${key}' is not recognized on <${tagName}>. Valid attributes: ${validAttrs.join(', ') || 'none'}`,
      },
    }
  }

  const result = coerceAttributeValue(raw, attrSchema.type)
  if (!result.ok) {
    return {
      ok: false,
      error: {
        _tag: 'InvalidAttributeValue',
        attribute: key,
        expected: attrSchema.type,
        received: raw,
        detail: `Attribute '${key}' on <${tagName}> expects a ${attrSchema.type}, got "${raw}"`,
      },
    }
  }

  return result
}

/**
 * Validate and coerce a child element attribute.
 *
 * Unknown child attributes are silently kept as strings (existing behavior —
 * buildInput drops attributes not in the binding anyway).
 */
export function validateChildAttr(
  parentTagName: string,
  childTagName: string,
  childSchema: ChildTagSchema,
  key: string,
  raw: string,
): AttrValidationResult {
  const attrSchema = childSchema.attributes.get(key)
  if (!attrSchema) {
    // Unknown child attributes silently pass through as strings
    return { ok: true, value: raw }
  }

  const result = coerceAttributeValue(raw, attrSchema.type)
  if (!result.ok) {
    return {
      ok: false,
      error: {
        _tag: 'InvalidAttributeValue',
        attribute: key,
        expected: attrSchema.type,
        received: raw,
        detail: `Attribute '${key}' on child <${childTagName}> inside <${parentTagName}> expects a ${attrSchema.type}, got "${raw}"`,
      },
    }
  }

  return result
}
