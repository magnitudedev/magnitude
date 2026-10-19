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

/**
 * The Hugging Face account Magnitude signed in to, stored in `auth.json` under its own key beside the
 * provider credentials. Tokens decode as `Redacted` so they never reach a log by accident.
 */
export const HuggingFaceCredentialSchema = Schema.Struct({
  type: Schema.Literal('huggingface-oauth'),
  accessToken: Schema.Redacted(Schema.String),
  refreshToken: Schema.optionalWith(Schema.Redacted(Schema.String), { as: 'Option', exact: true }),
  expiresAt: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  username: Schema.String,
  avatarUrl: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
})
export type HuggingFaceCredential = Schema.Schema.Type<typeof HuggingFaceCredentialSchema>
