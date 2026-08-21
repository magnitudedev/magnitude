import { Schema } from "effect"

const optional = <A, I, R>(schema: Schema.Schema<A, I, R>) =>
  Schema.optionalWith(schema, { as: "Option", exact: true })

const NullableStringSchema = Schema.NullOr(Schema.String)

export const CrwSearchRequestSchema = Schema.Struct({
  query: Schema.String.pipe(Schema.minLength(1)),
  limit: optional(
    Schema.Int.pipe(
      Schema.greaterThanOrEqualTo(1),
      Schema.lessThanOrEqualTo(50),
    ),
  ),
})

export const CrwSearchResultSchema = Schema.Struct({
  url: Schema.String,
  title: optional(NullableStringSchema),
  description: optional(NullableStringSchema),
  snippet: optional(NullableStringSchema),
  position: optional(Schema.Number),
  score: optional(Schema.Number),
  category: optional(NullableStringSchema),
})

/**
 * The engine returns results either as a bare array (managed endpoint) or
 * wrapped in `{ results: [...] }` (self-hosted). Accept both.
 *
 * The wrapped form is an array only while the request asks for neither sources
 * nor categories, which is all this client sends. Widen this alongside the
 * request body if that ever changes.
 */
export const CrwSearchPayloadSchema = Schema.Union(
  Schema.Array(CrwSearchResultSchema),
  Schema.Struct({ results: Schema.Array(CrwSearchResultSchema) }),
)

export const CrwSearchResponseSchema = Schema.Struct({
  success: optional(Schema.Boolean),
  data: optional(CrwSearchPayloadSchema),
  error: optional(Schema.String),
})

export const CrwSearchErrorResponseSchema = Schema.Struct({
  error: Schema.String,
})

export type CrwSearchRequest = typeof CrwSearchRequestSchema.Type
export type CrwSearchResponse = typeof CrwSearchResponseSchema.Type
export type CrwSearchResult = typeof CrwSearchResultSchema.Type
