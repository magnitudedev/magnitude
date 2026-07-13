import { Schema } from 'effect'

export const ApiKeyAuthSchema = Schema.Struct({
  type: Schema.Literal('api'),
  key: Schema.String,
})
export type ApiKeyAuth = Schema.Schema.Type<typeof ApiKeyAuthSchema>

export const EndpointAuthSchema = Schema.Struct({
  type: Schema.Literal('endpoint'),
  endpoint: Schema.String,
  apiKey: Schema.optional(Schema.String),
})
export type EndpointAuth = Schema.Schema.Type<typeof EndpointAuthSchema>

export const AuthInfoSchema = Schema.Union(ApiKeyAuthSchema, EndpointAuthSchema)
export type AuthInfo = Schema.Schema.Type<typeof AuthInfoSchema>

export function isValidAuthInfo(value: unknown): value is AuthInfo {
  return Schema.is(AuthInfoSchema)(value)
}
