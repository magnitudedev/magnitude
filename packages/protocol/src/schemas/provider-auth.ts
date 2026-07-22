import { Schema } from "effect"

export const ApiKeyAuthSchema = Schema.Struct({
  type: Schema.Literal("api"),
  key: Schema.String,
})

export const EndpointAuthSchema = Schema.Struct({
  type: Schema.Literal("endpoint"),
  endpoint: Schema.String,
  apiKey: Schema.optional(Schema.String),
})

export const ProviderAuthSchema = Schema.Union(ApiKeyAuthSchema, EndpointAuthSchema)
export type ProviderAuth = Schema.Schema.Type<typeof ProviderAuthSchema>
