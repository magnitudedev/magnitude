import { Effect, Option, Schema, Secret } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import type { LlamaCppConnection } from "../contracts"
import { LlamaCppEndpointClientError } from "../errors"

const ModelMetadata = Schema.Struct({
  n_ctx: Schema.optional(Schema.Number),
  n_ctx_train: Schema.optional(Schema.Number),
  n_vocab: Schema.optional(Schema.Number),
  n_params: Schema.optional(Schema.Number),
  size: Schema.optional(Schema.Number),
  ftype: Schema.optional(Schema.String),
  "general.architecture": Schema.optional(Schema.String),
  "general.name": Schema.optional(Schema.String),
  "general.basename": Schema.optional(Schema.String),
  "general.version": Schema.optional(Schema.String),
  "general.finetune": Schema.optional(Schema.String),
  "general.size_label": Schema.optional(Schema.String),
  "tokenizer.ggml.model": Schema.optional(Schema.String),
  "tokenizer.ggml.pre": Schema.optional(Schema.String),
  general_architecture: Schema.optional(Schema.String),
  general_name: Schema.optional(Schema.String),
  general_basename: Schema.optional(Schema.String),
  general_version: Schema.optional(Schema.String),
  general_finetune: Schema.optional(Schema.String),
  general_size_label: Schema.optional(Schema.String),
  tokenizer_ggml_model: Schema.optional(Schema.String),
  tokenizer_ggml_pre: Schema.optional(Schema.String),
})
const ServedModel = Schema.Struct({
  id: Schema.String,
  object: Schema.String,
  path: Schema.optional(Schema.String),
  aliases: Schema.optional(Schema.Array(Schema.String)),
  tags: Schema.optional(Schema.Array(Schema.String)),
  created: Schema.optional(Schema.Number),
  owned_by: Schema.optional(Schema.String),
  meta: Schema.optional(Schema.NullOr(ModelMetadata)),
  status: Schema.optional(Schema.Struct({
    value: Schema.optional(Schema.String),
    args: Schema.optional(Schema.Array(Schema.String)),
  })),
  architecture: Schema.optional(Schema.Struct({
    input_modalities: Schema.optional(Schema.Array(Schema.String)),
    output_modalities: Schema.optional(Schema.Array(Schema.String)),
  })),
})
const ModelsResponse = Schema.Struct({ data: Schema.Array(ServedModel) })
const HealthResponse = Schema.Struct({ status: Schema.Literal("ok") })
const PropsResponse = Schema.Struct({
  build_info: Schema.optional(Schema.String),
  role: Schema.optional(Schema.String),
  model_alias: Schema.optional(Schema.String),
  model_path: Schema.optional(Schema.String),
  model_ftype: Schema.optional(Schema.String),
  chat_template: Schema.optional(Schema.String),
  modalities: Schema.optional(Schema.Struct({
    vision: Schema.optional(Schema.Boolean),
    audio: Schema.optional(Schema.Boolean),
  })),
  default_generation_settings: Schema.optional(Schema.Struct({
    n_ctx: Schema.optional(Schema.Number),
  })),
})

export type LlamaCppServedModel = Schema.Schema.Type<typeof ServedModel>
export type LlamaCppEndpointProps = Schema.Schema.Type<typeof PropsResponse>

export type LlamaCppHealth =
  | { readonly _tag: "Ready" }
  | { readonly _tag: "Loading" }
  | { readonly _tag: "Unavailable"; readonly message: string }

export interface LlamaCppEndpointClient {
  readonly health: Effect.Effect<LlamaCppHealth, never, HttpClient.HttpClient>
  readonly props: Effect.Effect<LlamaCppEndpointProps, LlamaCppEndpointClientError, HttpClient.HttpClient>
  readonly models: Effect.Effect<readonly LlamaCppServedModel[], LlamaCppEndpointClientError, HttpClient.HttpClient>
}

const normalizeBaseUrl = (value: string): string => value.trim().replace(/\/+$/, "")

const requestHeaders = (connection: LlamaCppConnection): Record<string, string> =>
  Option.match(connection.apiKey, {
    onNone: () => ({}),
    onSome: (secret) => ({ Authorization: `Bearer ${Secret.value(secret)}` }),
  })

const error = (
  operation: LlamaCppEndpointClientError["operation"],
  endpoint: string,
  reason: string,
  cause?: unknown,
): LlamaCppEndpointClientError => new LlamaCppEndpointClientError({
  operation,
  endpoint,
  reason,
  ...(cause === undefined ? {} : { cause }),
})

export const makeLlamaCppEndpointClient = (
  connection: LlamaCppConnection,
): LlamaCppEndpointClient => {
  const endpoint = normalizeBaseUrl(connection.baseUrl)
  const headers = requestHeaders(connection)
  const execute = (
    operation: LlamaCppEndpointClientError["operation"],
    request: HttpClientRequest.HttpClientRequest,
  ) => Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    const response = yield* client.execute(HttpClientRequest.setHeaders(request, headers)).pipe(
      Effect.mapError((cause) => error(operation, endpoint, "Request failed", cause)),
    )
    if (response.status < 200 || response.status >= 300) {
      const body = yield* response.text.pipe(Effect.orElseSucceed(() => ""))
      return yield* error(operation, endpoint, body.trim() || `HTTP ${response.status}`)
    }
    return response
  })

  const decodeJson = <A, I>(
    operation: LlamaCppEndpointClientError["operation"],
    schema: Schema.Schema<A, I>,
    request: HttpClientRequest.HttpClientRequest,
  ): Effect.Effect<A, LlamaCppEndpointClientError, HttpClient.HttpClient> =>
    execute(operation, request).pipe(
      Effect.flatMap((response) => response.json),
      Effect.flatMap(Schema.decodeUnknown(schema)),
      Effect.mapError((cause) => cause instanceof LlamaCppEndpointClientError
        ? cause
        : error(operation, endpoint, "Response did not match the llama.cpp contract", cause)),
    )

  return {
    health: Effect.gen(function* () {
      const client = yield* HttpClient.HttpClient
      const response = yield* client.execute(
        HttpClientRequest.get(`${endpoint}/health`).pipe(HttpClientRequest.setHeaders(headers)),
      ).pipe(
        Effect.timeout("2 seconds"),
        Effect.catchAll(() => Effect.succeed(null)),
      )
      if (response === null) return { _tag: "Unavailable", message: "Connection failed" }
      if (response.status === 200) {
        const valid = yield* response.json.pipe(
          Effect.flatMap(Schema.decodeUnknown(HealthResponse)),
          Effect.as(true),
          Effect.orElseSucceed(() => false),
        )
        return valid
          ? { _tag: "Ready" }
          : { _tag: "Unavailable", message: "Invalid llama.cpp health response" }
      }
      if (response.status === 503) return { _tag: "Loading" }
      return { _tag: "Unavailable", message: `HTTP ${response.status}` }
    }),
    props: decodeJson("props", PropsResponse, HttpClientRequest.get(`${endpoint}/props`)).pipe(
      Effect.orElse(() => decodeJson("props", PropsResponse, HttpClientRequest.get(`${endpoint}/v1/props`))),
    ),
    models: decodeJson("models", ModelsResponse, HttpClientRequest.get(`${endpoint}/v1/models`)).pipe(
      Effect.map((response) => response.data),
    ),
  }
}

export type { LlamaCppConnection } from "../contracts"
export { LlamaCppEndpointClientError } from "../errors"
