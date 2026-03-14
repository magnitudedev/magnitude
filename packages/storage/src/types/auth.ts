import { Schema } from 'effect'

export const ApiKeyAuthSchema = Schema.Struct({
  type: Schema.Literal('api'),
  key: Schema.String,
})
export type ApiKeyAuth = Schema.Schema.Type<typeof ApiKeyAuthSchema>

export const OAuthAuthSchema = Schema.Struct({
  type: Schema.Literal('oauth'),
  accessToken: Schema.String,
  refreshToken: Schema.String,
  expiresAt: Schema.Number,
  accountId: Schema.optional(Schema.String),
  providerSpecific: Schema.optional(
    Schema.Record({ key: Schema.String, value: Schema.Unknown })
  ),
})
export type OAuthAuth = Schema.Schema.Type<typeof OAuthAuthSchema>

export const AwsAuthSchema = Schema.Struct({
  type: Schema.Literal('aws'),
  profile: Schema.optional(Schema.String),
  region: Schema.optional(Schema.String),
})
export type AwsAuth = Schema.Schema.Type<typeof AwsAuthSchema>

export const GcpAuthSchema = Schema.Struct({
  type: Schema.Literal('gcp'),
  credentialsPath: Schema.String,
  project: Schema.optional(Schema.String),
  location: Schema.optional(Schema.String),
})
export type GcpAuth = Schema.Schema.Type<typeof GcpAuthSchema>

export const AuthInfoSchema = Schema.Union(
  ApiKeyAuthSchema,
  OAuthAuthSchema,
  AwsAuthSchema,
  GcpAuthSchema
)
export type AuthInfo = Schema.Schema.Type<typeof AuthInfoSchema>

export function isValidAuthInfo(value: unknown): value is AuthInfo {
  return Schema.is(AuthInfoSchema)(value)
}