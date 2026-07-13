/**
 * Tool Error Helpers
 */

import { Schema } from 'effect'

/**
 * Helper to create a tool error schema with a tagged discriminant.
 */
export function ToolErrorSchema<
  Tag extends string,
  Fields extends Schema.Struct.Fields
>(
  tag: Tag,
  fields: Fields
) {
  return Schema.Struct({
    _tag: Schema.Literal(tag),
    message: Schema.String,
    ...fields
  })
}
