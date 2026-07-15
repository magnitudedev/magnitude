import { Data, Effect, Option, Secret } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import {
  ModelCatalogError,
  StreamStartOperationalFailure,
  type BaseCallOptions,
  type BoundModel,
  type ModelCatalog,
  type Provider,
  type ProviderModelBindOptions,
} from "@magnitudedev/ai"
import {
  makeLlamaCppEndpointClient,
  type LlamaCppConnection,
} from "@magnitudedev/llamacpp/client"
import { createLlamaCppCatalog } from "./catalog"
import { createLlamaCppCompatibleSpec, wrapAsBaseModel } from "./models"
import type {
  LlamaCppCallOptions,
  LlamaCppDiscoveryResult,
  LlamaCppModelInfo,
} from "./contract"
import { classifyModelFamilyFromEvidence } from "../family-registry"

export const PROVIDER_ID = "llamacpp" as const
export const DEFAULT_LLAMACPP_ENDPOINT = "http://127.0.0.1:8080"

export class LlamaCppProviderBackendError extends Data.TaggedError("LlamaCppProviderBackendError")<{
  readonly operation: "resolve_connection"
  readonly modelId: string
  readonly reason: string
  readonly cause?: unknown
}> {}

export interface LlamaCppProviderBackend {
  readonly listModels: Effect.Effect<readonly LlamaCppModelInfo[], ModelCatalogError, HttpClient.HttpClient>
  readonly resolveConnection: (
    providerModelId: string,
  ) => Effect.Effect<LlamaCppConnection, LlamaCppProviderBackendError>
  readonly status: Effect.Effect<{
    readonly status: "ok" | "loading" | "not_found" | "error"
    readonly endpointLabel?: string
    readonly message?: string
    readonly hint?: string
  }, never, HttpClient.HttpClient>
}

export interface FixedLlamaCppEndpointConfig {
  readonly endpoint?: string
  readonly apiKey?: string
}

export type LlamaCppProvider = Provider<LlamaCppModelInfo>

export interface LlamaCppProviderComponents {
  readonly provider: LlamaCppProvider
  readonly catalog: ModelCatalog<LlamaCppModelInfo>
  readonly checkStatus: Effect.Effect<LlamaCppDiscoveryResult, never, HttpClient.HttpClient>
}

const normalizeEndpoint = (endpoint: string): string => endpoint.trim().replace(/\/+$/, "")

const authFor = (connection: LlamaCppConnection) => (headers: Headers): void => {
  if (Option.isSome(connection.apiKey)) {
    headers.set("Authorization", `Bearer ${Secret.value(connection.apiKey.value)}`)
  }
}

const inferenceEndpoint = (connection: LlamaCppConnection): string =>
  `${normalizeEndpoint(connection.baseUrl)}/v1`

export const makeFixedEndpointBackend = (
  config: FixedLlamaCppEndpointConfig = {},
): LlamaCppProviderBackend => {
  const connection: LlamaCppConnection = {
    baseUrl: normalizeEndpoint(config.endpoint ?? DEFAULT_LLAMACPP_ENDPOINT),
    apiKey: config.apiKey?.trim()
      ? Option.some(Secret.fromString(config.apiKey.trim()))
      : Option.none(),
  }
  const classify = makeClassifier()
  const catalog = createLlamaCppCatalog({
    endpoint: connection.baseUrl,
    auth: authFor(connection),
    classify,
  })
  const endpointClient = makeLlamaCppEndpointClient(connection)
  return {
    listModels: catalog.list,
    resolveConnection: () => Effect.succeed(connection),
    status: endpointClient.health.pipe(Effect.map((health) => {
      switch (health._tag) {
        case "Ready":
          return { status: "ok", endpointLabel: connection.baseUrl } as const
        case "Loading":
          return { status: "loading", endpointLabel: connection.baseUrl, message: "llama-server is loading a model" } as const
        case "Unavailable":
          return {
            status: "not_found",
            endpointLabel: connection.baseUrl,
            message: health.message,
            hint: "Start llama-server or configure a different endpoint",
          } as const
      }
    })),
  }
}

type ModelWithoutFamily = Omit<LlamaCppModelInfo, "modelFamilyId">

const makeClassifier = () => (model: ModelWithoutFamily): Option.Option<string> =>
  classifyModelFamilyFromEvidence({
    architecture: model.modelArchitecture,
    tokenizerModel: model.tokenizerModel,
    tokenizerPre: model.tokenizerPre,
  }, [
    model.metadataName,
    ...(model.baseModelNames ?? []),
    ...(model.baseModelRepositories ?? []),
    model.sourceModelPath,
    model.displayName,
    model.providerModelId,
    model.modelArchitecture,
  ])

const backendCatalog = (
  backend: LlamaCppProviderBackend,
  classify: (model: ModelWithoutFamily) => Option.Option<string>,
): ModelCatalog<LlamaCppModelInfo> => {
  const list = backend.listModels.pipe(
    Effect.map((models) => models.map((model) => ({
      ...model,
      modelFamilyId: Option.getOrElse(classify(model), () => model.modelFamilyId),
    }))),
  )
  return {
    list,
    refresh: list,
    get: (_providerId, providerModelId) => list.pipe(Effect.flatMap((models) => {
      const model = models.find((candidate) => candidate.providerModelId === providerModelId)
      return model
        ? Effect.succeed(model)
        : Effect.fail(new ModelCatalogError({ message: `Model not found: ${providerModelId}` }))
    })),
  }
}

const callOptions = (defaults: Partial<BaseCallOptions> | undefined): Partial<LlamaCppCallOptions> | undefined =>
  defaults
    ? {
        ...(defaults.maxTokens === undefined ? {} : { maxTokens: defaults.maxTokens }),
        ...(defaults.toolChoice === undefined ? {} : { toolChoice: defaults.toolChoice }),
        ...(defaults.reasoningEffort === undefined ? {} : { reasoningEffort: defaults.reasoningEffort }),
      }
    : undefined

const dynamicBoundModel = (
  backend: LlamaCppProviderBackend,
  providerModelId: string,
  options: ProviderModelBindOptions | undefined,
): BoundModel<BaseCallOptions> => ({
  stream: (prompt, tools, requestOptions) => backend.resolveConnection(providerModelId).pipe(
    Effect.mapError((cause) => new StreamStartOperationalFailure({
      call: {
        provider: PROVIDER_ID,
        model: providerModelId,
        method: "POST",
        url: `llamacpp://resolve/${encodeURIComponent(providerModelId)}`,
      },
      reason: {
        _tag: "RequestFailedBeforeResponse",
        cause: { _tag: "ErrorCause", name: cause._tag, message: cause.reason },
      },
    })),
    Effect.flatMap((connection) => wrapAsBaseModel(
      createLlamaCppCompatibleSpec({
        modelId: providerModelId,
        endpoint: inferenceEndpoint(connection),
      }).bind({
        auth: authFor(connection),
        defaults: callOptions(options?.defaults),
        ...(options?.imagePlaceholders ? { imagePlaceholders: options.imagePlaceholders } : {}),
      }),
    ).stream(prompt, tools, requestOptions)),
  ),
})

export function createLlamaCppProvider(backend: LlamaCppProviderBackend): LlamaCppProviderComponents {
  const classifyModelFamily = makeClassifier()
  const catalog = backendCatalog(backend, classifyModelFamily)
  const provider: LlamaCppProvider = {
    id: PROVIDER_ID,
    displayName: "Llama.cpp",
    catalog,
    bindModel: (providerModelId, options) => Effect.succeed(dynamicBoundModel(
      backend,
      providerModelId,
      options,
    )),
    classifyModelFamily,
  }
  const checkStatus = Effect.gen(function* () {
    const status = yield* backend.status
    const models = status.status === "ok"
      ? yield* backend.listModels.pipe(Effect.orElseSucceed(() => []))
      : []
    return {
      models,
      status: status.status,
      endpoint: status.endpointLabel ?? "managed",
      ...(status.message ? { message: status.message } : {}),
      ...(status.hint ? { hint: status.hint } : {}),
    } satisfies LlamaCppDiscoveryResult
  })
  return { provider, catalog, checkStatus }
}
