/**
 * @magnitudedev/tools — Error Helpers
 */

import { Schema } from "@effect/schema"

/**
 * Base interface for tool errors.
 * All tool errors must have a `_tag` (discriminant) and `message`.
 */
export interface ToolErrorBase {
  readonly _tag: string
  readonly message: string
}

/**
 * Helper to create a tool error schema with a tagged discriminant.
 *
 * @example
 * ```ts
 * const QueryError = ToolErrorSchema("QueryError", { code: Schema.String })
 * const NotFoundError = ToolErrorSchema("NotFoundError", {})
 * const DbError = Schema.Union(QueryError, NotFoundError)
 * ```
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
