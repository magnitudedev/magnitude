import { Schema } from "effect"
import { JsonValueSchema } from "@magnitudedev/ai"

const optional = <A, I, R>(schema: Schema.Schema<A, I, R>) =>
  Schema.optionalWith(schema, { as: "Option", exact: true })

const JsonRecordSchema = Schema.Record({
  key: Schema.String,
  value: JsonValueSchema,
})

const NullableStringSchema = Schema.NullOr(Schema.String)

export const ExaSearchTypeSchema = Schema.Literal(
  "keyword",
  "neural",
  "hybrid",
  "instant",
  "fast",
  "auto",
  "deep-lite",
  "deep",
  "deep-reasoning",
)

export const ExaTextOutputSchema = Schema.Struct({
  type: Schema.Literal("text"),
  description: optional(Schema.String),
})

export const ExaObjectOutputSchema = Schema.Struct({
  type: Schema.Literal("object"),
  description: optional(Schema.String),
  properties: optional(JsonRecordSchema),
  required: optional(Schema.Array(Schema.String)),
  additionalProperties: optional(Schema.Boolean),
})

export const ExaOutputSchema = Schema.Union(
  ExaTextOutputSchema,
  ExaObjectOutputSchema,
)

export const ExaTextContentsOptionsSchema = Schema.Struct({
  maxCharacters: optional(Schema.Int),
  includeHtmlTags: optional(Schema.Boolean),
  verbosity: optional(Schema.Literal("compact", "standard", "full")),
  includeSections: optional(Schema.Array(Schema.Literal(
    "unspecified",
    "header",
    "navigation",
    "banner",
    "body",
    "sidebar",
    "footer",
    "metadata",
  ))),
  excludeSections: optional(Schema.Array(Schema.Literal(
    "unspecified",
    "header",
    "navigation",
    "banner",
    "body",
    "sidebar",
    "footer",
    "metadata",
  ))),
})

export const ExaHighlightsContentsOptionsSchema = Schema.Struct({
  query: optional(Schema.String),
  maxCharacters: optional(Schema.Int),
  numSentences: optional(Schema.Int),
  highlightsPerUrl: optional(Schema.Int),
})

export const ExaSummaryContentsOptionsSchema = Schema.Struct({
  query: optional(Schema.String),
  schema: optional(JsonRecordSchema),
})

export const ExaContextOptionsSchema = Schema.Struct({
  maxCharacters: optional(Schema.Int),
})

export const ExaExtrasOptionsSchema = Schema.Struct({
  links: optional(Schema.Int),
  imageLinks: optional(Schema.Int),
})

export const ExaContentsOptionsSchema = Schema.Struct({
  text: optional(Schema.Union(Schema.Boolean, ExaTextContentsOptionsSchema)),
  highlights: optional(Schema.Union(Schema.Boolean, ExaHighlightsContentsOptionsSchema)),
  summary: optional(Schema.Union(Schema.Boolean, ExaSummaryContentsOptionsSchema)),
  livecrawl: optional(Schema.Literal("never", "fallback", "always", "auto", "preferred")),
  context: optional(Schema.Union(Schema.Boolean, ExaContextOptionsSchema)),
  livecrawlTimeout: optional(Schema.Int),
  maxAgeHours: optional(Schema.Int),
  filterEmptyResults: optional(Schema.Boolean),
  subpages: optional(Schema.Int),
  subpageTarget: optional(Schema.Union(Schema.String, Schema.Array(Schema.String))),
  extras: optional(ExaExtrasOptionsSchema),
})

export const ExaSearchRequestSchema = Schema.Struct({
  query: Schema.String.pipe(Schema.minLength(1)),
  type: optional(Schema.NullOr(ExaSearchTypeSchema)),
  stream: optional(Schema.NullOr(Schema.Boolean)),
  numResults: optional(Schema.NullOr(
    Schema.Int.pipe(
      Schema.greaterThanOrEqualTo(1),
      Schema.lessThanOrEqualTo(100),
    ),
  )),
  category: optional(NullableStringSchema),
  userLocation: optional(Schema.NullOr(Schema.String.pipe(Schema.length(2)))),
  includeDomains: optional(Schema.NullOr(
    Schema.Array(Schema.String).pipe(Schema.maxItems(1_200)),
  )),
  excludeDomains: optional(Schema.NullOr(
    Schema.Array(Schema.String).pipe(Schema.maxItems(1_200)),
  )),
  startCrawlDate: optional(NullableStringSchema),
  endCrawlDate: optional(NullableStringSchema),
  startPublishedDate: optional(NullableStringSchema),
  endPublishedDate: optional(NullableStringSchema),
  includeText: optional(Schema.NullOr(
    Schema.Array(Schema.String).pipe(Schema.maxItems(1)),
  )),
  excludeText: optional(Schema.NullOr(
    Schema.Array(Schema.String).pipe(Schema.maxItems(1)),
  )),
  flags: optional(Schema.NullOr(Schema.Array(Schema.String))),
  moderation: optional(Schema.NullOr(Schema.Boolean)),
  useAutoprompt: optional(Schema.NullOr(Schema.Boolean)),
  additionalQueries: optional(Schema.NullOr(
    Schema.Array(Schema.String).pipe(
      Schema.minItems(1),
      Schema.maxItems(10),
    ),
  )),
  contents: optional(Schema.NullOr(ExaContentsOptionsSchema)),
  outputSchema: optional(ExaOutputSchema),
  systemPrompt: optional(NullableStringSchema),
  compliance: optional(Schema.NullOr(Schema.Literal("hipaa"))),
})

export const ExaSearchErrorTagSchema = Schema.Literal(
  "INVALID_API_KEY",
  "NO_MORE_CREDITS",
  "API_KEY_BUDGET_EXCEEDED",
  "TEAM_BUDGET_EXCEEDED",
  "ACCESS_DENIED",
  "FEATURE_DISABLED",
  "CONTENT_FILTER_ERROR",
  "INVALID_REQUEST_BODY",
  "INVALID_REQUEST",
  "INVALID_NUM_RESULTS",
  "INVALID_FLAGS",
  "NUM_RESULTS_EXCEEDED",
  "DEFAULT_ERROR",
  "INTERNAL_ERROR",
)

export const ExaSearchErrorResponseSchema = Schema.Struct({
  requestId: optional(Schema.String),
  error: Schema.String,
  tag: optional(ExaSearchErrorTagSchema),
})

export const ExaEntitySchema = Schema.Struct({
  id: Schema.String,
  type: Schema.String,
  version: Schema.Number,
  properties: JsonRecordSchema,
})

const ExaExtrasResponseSchema = Schema.Struct({
  links: optional(Schema.Array(Schema.String)),
  imageLinks: optional(Schema.Array(Schema.String)),
})

const ExaSubpageResultSchema = Schema.Struct({
  title: Schema.NullOr(Schema.String),
  url: Schema.String,
  id: Schema.String,
  publishedDate: optional(NullableStringSchema),
  author: optional(NullableStringSchema),
  score: optional(Schema.Number),
  image: optional(NullableStringSchema),
  favicon: optional(NullableStringSchema),
  text: optional(Schema.String),
  highlights: optional(Schema.Array(Schema.String)),
  highlightScores: optional(Schema.Array(Schema.Number)),
  summary: optional(Schema.String),
  extras: optional(ExaExtrasResponseSchema),
  entities: optional(Schema.Array(ExaEntitySchema)),
})

export const ExaSearchResultSchema = Schema.Struct({
  title: Schema.NullOr(Schema.String),
  url: Schema.String,
  id: Schema.String,
  publishedDate: optional(NullableStringSchema),
  author: optional(NullableStringSchema),
  score: optional(Schema.Number),
  image: optional(NullableStringSchema),
  favicon: optional(NullableStringSchema),
  text: optional(Schema.String),
  highlights: optional(Schema.Array(Schema.String)),
  highlightScores: optional(Schema.Array(Schema.Number)),
  summary: optional(Schema.String),
  subpages: optional(Schema.Array(ExaSubpageResultSchema)),
  extras: optional(ExaExtrasResponseSchema),
  entities: optional(Schema.Array(ExaEntitySchema)),
})

export const ExaGroundingCitationSchema = Schema.Struct({
  url: Schema.String,
  title: Schema.String,
})

export const ExaGroundingSchema = Schema.Struct({
  field: Schema.String,
  citations: Schema.Array(ExaGroundingCitationSchema),
  confidence: Schema.Literal("low", "medium", "high"),
})

export const ExaSynthesizedOutputSchema = Schema.Struct({
  content: Schema.Union(Schema.String, JsonRecordSchema),
  grounding: Schema.Array(ExaGroundingSchema),
})

export const ExaCostDollarsSchema = Schema.Struct({
  total: Schema.Number,
  search: optional(Schema.Struct({
    neural: optional(Schema.Number),
    keyword: optional(Schema.Number),
  })),
  contents: optional(Schema.Struct({
    text: optional(Schema.Number),
    highlights: optional(Schema.Number),
    summary: optional(Schema.Number),
  })),
})

export const ExaStatusSchema = Schema.Struct({
  id: Schema.String,
  status: Schema.String,
  source: optional(Schema.String),
})

export const ExaSearchResponseSchema = Schema.Struct({
  results: Schema.Array(ExaSearchResultSchema),
  requestId: Schema.String,
  output: optional(ExaSynthesizedOutputSchema),
  statuses: optional(Schema.Array(ExaStatusSchema)),
  costDollars: optional(ExaCostDollarsSchema),
  resolvedSearchType: optional(Schema.String),
  searchTime: optional(Schema.Number),
  context: optional(Schema.String),
  autoDate: optional(Schema.String),
})

export type ExaSearchRequest = typeof ExaSearchRequestSchema.Type
export type ExaSearchResponse = typeof ExaSearchResponseSchema.Type
