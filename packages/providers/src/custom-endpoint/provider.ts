import { Effect, Option, Schema } from "effect"
import {
  Auth,
  AVAILABLE_PROVIDER_MODEL,
  ModelCatalogError,
  ModelDiscoveryOperationIdSchema,
  NativeChatCompletions,
  Option as CallOption,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  ReasoningProperty,
  StreamStartProviderRejection,
  VisionProperty,
  type AuthApplicator,
  type BaseCallOptions,
  type BoundModel,
  type ModelCatalog,
  type Provider,
  type ProviderModel,
  type ProviderModelBindOptions,
  type RejectedHttpResponse,
  type ProviderCall,
  type StreamStartFailure,
} from "@magnitudedev/ai"
import type {
  CustomEndpointDeclaration,
  CustomEndpointName,
} from "@magnitudedev/storage"
import { classifyModelFamily } from "../family-registry"
import type { AuthStatus, DiscoverableProviderInstance } from "../registry"
import { customEndpointChunkDecoder } from "./chunk-decoder"

export const customEndpointProviderId = (name: CustomEndpointName) =>
  ProviderIdSchema.make(`custom:${name}`)

const callOptions = (reasoningDeclared: boolean) => ({
  maxTokens: NativeChatCompletions.options.maxTokens,
  toolChoice: NativeChatCompletions.options.toolChoice,
  reasoningEffort: reasoningDeclared
    ? CallOption.field("reasoning_effort", Schema.String)
    : CallOption.ignore<string>(),
})

const classifyRejectedResponse = (
  call: ProviderCall,
  response: RejectedHttpResponse,
): StreamStartProviderRejection => {
  const message = response.body.slice(0, 500) || `HTTP ${response.status}`
  const rejection = response.status === 401 || response.status === 403
    ? { _tag: "AuthRejected" as const, message }
    : response.status === 404
      ? { _tag: "ModelUnavailable" as const, message }
      : response.status === 429
        ? {
            _tag: "RateLimited" as const,
            message,
            retryPolicy: {
              retry: true,
              retryAfterMs: Option.fromNullable(response.retryAfterMs),
            },
          }
        : response.status >= 500
          ? {
              _tag: "UpstreamFailure" as const,
              message,
              retryPolicy: {
                retry: true,
                retryAfterMs: Option.fromNullable(response.retryAfterMs),
              },
            }
          : { _tag: "InvalidRequest" as const, message }
  return new StreamStartProviderRejection({ call, response, rejection })
}

const normalizedEndpoint = (baseUrl: string): string => baseUrl.replace(/\/+$/, "")

const resolveAuthentication = (
  declaration: CustomEndpointDeclaration,
  environment: Readonly<Record<string, string | undefined>>,
): { readonly auth: AuthApplicator; readonly status: AuthStatus } => {
  const { authentication, headers } = declaration.connection
  const literalHeaders = Option.getOrElse(headers, () => ({}))
  const credential = authentication.type === "none"
    ? undefined
    : environment[authentication.credential.variable]?.trim()
  const status: AuthStatus = authentication.type === "none"
    ? { _tag: "no_auth_required" }
    : credential
      ? { _tag: "authenticated" }
      : {
          _tag: "not_configured",
          reason: `Set ${authentication.credential.variable} and restart Magnitude`,
        }
  const auth = (headers: Headers): void => {
    for (const [name, value] of Object.entries(literalHeaders)) headers.set(name, value)
    if (authentication.type === "none") return
    if (!credential) throw new Error(`Missing environment credential ${authentication.credential.variable}`)
    if (authentication.type === "bearer") Auth.bearer(credential)(headers)
    else Auth.header(authentication.name, credential)(headers)
  }
  return { auth, status }
}

const toProviderModels = (
  name: CustomEndpointName,
  declaration: CustomEndpointDeclaration,
): readonly ProviderModel[] => {
  const providerId = customEndpointProviderId(name)
  return Object.entries(declaration.models).map(([modelName, model]) => {
    const reasoning = Option.flatMap(model.capabilities, (capabilities) => capabilities.reasoning)
    const efforts = Option.match(reasoning, {
      onNone: () => [ReasoningEffortSchema.make("none")],
      onSome: (value) => [...value.efforts],
    })
    const defaultReasoningEffort = Option.match(reasoning, {
      onNone: () => ReasoningEffortSchema.make("none"),
      onSome: (value) => value.defaultEffort,
    })
    const providerModelId = ProviderModelIdSchema.make(modelName)
    return {
      providerId,
      providerModelId,
      modelFamilyId: Option.getOrUndefined(classifyModelFamily(modelName)),
      displayName: model.displayName,
      contextWindow: model.contextWindow,
      maxOutputTokens: model.maxOutputTokens,
      defaultReasoningEffort,
      properties: {
        vision: new VisionProperty.states.Resolved({
          value: Option.flatMap(model.capabilities, (capabilities) => capabilities.vision).pipe(
            Option.getOrElse(() => false),
          ),
        }),
        reasoning: new ReasoningProperty.states.Resolved({ value: efforts }),
      },
      servingCapabilities: { tools: true, structuredOutput: false },
      availability: AVAILABLE_PROVIDER_MODEL,
      pricing: Option.none(),
    }
  })
}

export const createCustomEndpointProvider = <TPreparation = never>(
  name: CustomEndpointName,
  declaration: CustomEndpointDeclaration,
  environment: Readonly<Record<string, string | undefined>> = process.env,
): DiscoverableProviderInstance<TPreparation> => {
  const providerId = customEndpointProviderId(name)
  const models = toProviderModels(name, declaration)
  const auth = resolveAuthentication(declaration, environment)
  const catalog: ModelCatalog<ProviderModel> = {
    list: Effect.succeed(models),
    refresh: Effect.succeed(models),
    get: (requestedProviderId, providerModelId) => {
      const model = models.find((candidate) => requestedProviderId === providerId
        && candidate.providerModelId === providerModelId)
      return model === undefined
        ? Effect.fail(new ModelCatalogError({ message: `Unknown custom endpoint model: ${providerModelId}` }))
        : Effect.succeed(model)
    },
  }
  const provider: Provider<ProviderModel, TPreparation> = {
    id: providerId,
    displayName: declaration.displayName,
    catalog,
    classifyModelFamily: (model) => classifyModelFamily(model.providerModelId),
    discoverModelProperties: () => Effect.succeed(
      ModelDiscoveryOperationIdSchema.make(`custom:${name}:authoritative`),
    ),
    bindModel: (
      providerModelId,
      options?: ProviderModelBindOptions,
    ): Effect.Effect<BoundModel<BaseCallOptions, StreamStartFailure, TPreparation>> => Effect.sync(() => {
      const declaredModel = Object.entries(declaration.models).find(
        ([modelName]) => modelName === providerModelId,
      )?.[1]
      const reasoningDeclared = declaredModel !== undefined
        && Option.exists(declaredModel.capabilities, (capabilities) =>
          Option.isSome(capabilities.reasoning))
      const internal = NativeChatCompletions.model({
        modelId: providerModelId,
        endpoint: normalizedEndpoint(declaration.connection.baseUrl),
        options: callOptions(reasoningDeclared),
        chunkDecoder: customEndpointChunkDecoder,
        classifyRejectedResponse,
      }).bind({
        auth: auth.auth,
        ...(options?.defaults === undefined ? {} : { defaults: options.defaults }),
        ...(options?.imagePlaceholders ? { imagePlaceholders: options.imagePlaceholders } : {}),
      })
      return {
        stream: (prompt, tools, callOptions) => internal.stream(prompt, tools, callOptions),
      }
    }),
  }
  return {
    provider,
    kind: "Custom",
    authStatus: auth.status,
    checkStatus: Effect.succeed({ status: "ok" as const }),
  }
}
