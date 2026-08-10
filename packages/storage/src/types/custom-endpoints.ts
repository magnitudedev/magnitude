import { ReasoningEffortSchema } from '@magnitudedev/ai'
import { Option, Schema } from 'effect'

const SerializableOptional = <A, I, R>(schema: Schema.Schema<A, I, R>) =>
  Schema.optionalWith(schema, { as: 'Option', exact: true } as const)

const DisplayNameSchema = Schema.NonEmptyString.pipe(Schema.maxLength(256))

export const CustomEndpointNameSchema = Schema.String.pipe(
  Schema.pattern(/^[a-z0-9][a-z0-9._-]*$/),
  Schema.maxLength(256),
  Schema.brand('CustomEndpointName'),
)
export type CustomEndpointName = Schema.Schema.Type<typeof CustomEndpointNameSchema>

export const ChatCompletionsModelNameSchema = Schema.String.pipe(
  Schema.minLength(1),
  Schema.maxLength(4_096),
  Schema.brand('ChatCompletionsModelName'),
)
export type ChatCompletionsModelName = Schema.Schema.Type<typeof ChatCompletionsModelNameSchema>

export const EnvironmentVariableNameSchema = Schema.String.pipe(
  Schema.pattern(/^[A-Za-z_][A-Za-z0-9_]*$/),
  Schema.maxLength(256),
  Schema.brand('EnvironmentVariableName'),
)

export const HttpHeaderNameSchema = Schema.String.pipe(
  Schema.pattern(/^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/),
  Schema.maxLength(256),
  Schema.brand('HttpHeaderName'),
)

export const HttpHeaderValueSchema = Schema.String.pipe(
  Schema.maxLength(8_192),
  Schema.filter((value) => !/[\r\n]/.test(value), {
    message: () => 'HTTP header values must not contain CR or LF',
  }),
  Schema.brand('HttpHeaderValue'),
)

export const HttpBaseUrlSchema = Schema.String.pipe(
  Schema.filter((value) => {
    try {
      const url = new URL(value)
      return (url.protocol === 'http:' || url.protocol === 'https:')
        && url.username === ''
        && url.password === ''
        && url.search === ''
        && url.hash === ''
        && !/\/chat\/completions\/?$/i.test(url.pathname)
    } catch {
      return false
    }
  }, {
    message: () => 'baseUrl must be an HTTP(S) API root without credentials, query, fragment, or /chat/completions',
  }),
  Schema.brand('HttpBaseUrl'),
)

export const EnvironmentCredentialSchema = Schema.Struct({
  type: Schema.Literal('environment'),
  variable: EnvironmentVariableNameSchema,
})
export type EnvironmentCredential = Schema.Schema.Type<typeof EnvironmentCredentialSchema>

export const CustomEndpointAuthenticationSchema = Schema.Union(
  Schema.Struct({ type: Schema.Literal('none') }),
  Schema.Struct({
    type: Schema.Literal('bearer'),
    credential: EnvironmentCredentialSchema,
  }),
  Schema.Struct({
    type: Schema.Literal('header'),
    name: HttpHeaderNameSchema,
    credential: EnvironmentCredentialSchema,
  }),
)
export type CustomEndpointAuthentication = Schema.Schema.Type<typeof CustomEndpointAuthenticationSchema>

export const CustomEndpointHeadersSchema = Schema.Record({
  key: HttpHeaderNameSchema,
  value: HttpHeaderValueSchema,
}).pipe(Schema.filter((headers) => {
  const names = Object.keys(headers).map((name) => name.toLowerCase())
  return new Set(names).size === names.length
}, { message: () => 'HTTP header names must be unique case-insensitively' }))

const authenticationHeader = (authentication: CustomEndpointAuthentication): string | undefined =>
  authentication.type === 'bearer'
    ? 'authorization'
    : authentication.type === 'header'
      ? authentication.name.toLowerCase()
      : undefined

export const CustomEndpointConnectionSchema = Schema.Struct({
  baseUrl: HttpBaseUrlSchema,
  authentication: CustomEndpointAuthenticationSchema,
  headers: SerializableOptional(CustomEndpointHeadersSchema),
}).pipe(Schema.filter(({ authentication, headers }) => {
  const owned = authenticationHeader(authentication)
  return owned === undefined || !Option.exists(headers, (values) =>
    Object.keys(values).some((name) => name.toLowerCase() === owned))
}, { message: () => 'literal headers must not replace the authentication header' }))

export const CustomEndpointReasoningSchema = Schema.Struct({
  efforts: Schema.NonEmptyArray(ReasoningEffortSchema).pipe(
    Schema.filter((efforts) => new Set(efforts).size === efforts.length, {
      message: () => 'reasoning efforts must be unique',
    }),
  ),
  defaultEffort: ReasoningEffortSchema,
}).pipe(Schema.filter(({ efforts, defaultEffort }) => efforts.includes(defaultEffort), {
  message: () => 'defaultEffort must be one of efforts',
}))

export const CustomEndpointCapabilitiesSchema = Schema.Struct({
  vision: SerializableOptional(Schema.Boolean),
  reasoning: SerializableOptional(CustomEndpointReasoningSchema),
})

const PositiveSafeIntegerSchema = Schema.Number.pipe(
  Schema.int(),
  Schema.positive(),
  Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER),
)

export const CustomEndpointModelSchema = Schema.Struct({
  displayName: DisplayNameSchema,
  contextWindow: PositiveSafeIntegerSchema,
  maxOutputTokens: PositiveSafeIntegerSchema,
  capabilities: SerializableOptional(CustomEndpointCapabilitiesSchema),
}).pipe(Schema.filter(({ contextWindow, maxOutputTokens }) => maxOutputTokens <= contextWindow, {
  message: () => 'maxOutputTokens must not exceed contextWindow',
}))
export type CustomEndpointModel = Schema.Schema.Type<typeof CustomEndpointModelSchema>

export const CustomEndpointModelsSchema = Schema.Record({
  key: ChatCompletionsModelNameSchema,
  value: CustomEndpointModelSchema,
}).pipe(Schema.filter((models) => Object.keys(models).length > 0, {
  message: () => 'a custom endpoint must declare at least one model',
}))

export const CustomEndpointDeclarationSchema = Schema.Struct({
  displayName: DisplayNameSchema,
  connection: CustomEndpointConnectionSchema,
  models: CustomEndpointModelsSchema,
})
export type CustomEndpointDeclaration = Schema.Schema.Type<typeof CustomEndpointDeclarationSchema>

export const CustomEndpointDeclarationsSchema = Schema.Record({
  key: CustomEndpointNameSchema,
  value: CustomEndpointDeclarationSchema,
})
export type CustomEndpointDeclarations = Schema.Schema.Type<typeof CustomEndpointDeclarationsSchema>
