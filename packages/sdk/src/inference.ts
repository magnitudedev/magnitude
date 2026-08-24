import type { GeneratedClientError } from "@magnitudedev/openapi-effect/client-runtime"
import type { IcnApiClient } from "@magnitudedev/icn-protocol/client"
import { makeIcnApiClient } from "@magnitudedev/icn-protocol/client"
import type { RpcClient } from "@effect/rpc"
import { RpcClientError } from "@effect/rpc/RpcClientError"
import * as HttpClient from "@effect/platform/HttpClient"
import * as S from "@magnitudedev/icn-protocol/schemas"
import {
  Group,
  Mutation,
  Operation,
  Query,
  QueryClient,
  Subscription,
} from "@magnitudedev/effect-query"
import { Data, Effect, Layer, Option, Schema, Stream } from "effect"
import { AcnBoundary, AcnRpc } from "@magnitudedev/acn-protocol"
import { MAGNITUDE_INFERENCE_BASE_URL } from "./inference-endpoint"

const Empty = Schema.Struct({})
const Never = Schema.Never

const GetInferenceHardware = Query.make("GetInferenceHardware", {
  payload: Empty,
  success: S.HardwareSnapshot,
  error: Never,
  staleTime: Infinity,
  gcTime: Infinity,
})
const GetInferenceModels = Query.make("GetInferenceModels", {
  payload: Empty,
  success: S.ModelsResponse,
  error: Never,
  staleTime: Infinity,
  gcTime: Infinity,
})
const GetInferenceModel = Query.make("GetInferenceModel", {
  payload: Schema.Struct({ modelId: Schema.String }),
  success: S.InferenceModel,
  error: Never,
})
const GetServableModels = Query.make("GetServableModels", {
  payload: Empty,
  success: S.OpenAiModelsResponse,
  error: Never,
  staleTime: Infinity,
  gcTime: Infinity,
})
const GetInferencePackages = Query.make("GetInferencePackages", {
  payload: Empty,
  success: S.InstalledModelPackagesResponse,
  error: Never,
  staleTime: Infinity,
  gcTime: Infinity,
})
const GetInferencePackage = Query.make("GetInferencePackage", {
  payload: Schema.Struct({ packageId: Schema.String }),
  success: S.InstalledModelPackage,
  error: Never,
})
const GetInferenceDownloads = Query.make("GetInferenceDownloads", {
  payload: Empty,
  success: S.ModelDownloadsResponse,
  error: Never,
  staleTime: Infinity,
  gcTime: Infinity,
})
const GetInferenceDownload = Query.make("GetInferenceDownload", {
  payload: Schema.Struct({ downloadId: Schema.String }),
  success: S.ModelDownload,
  error: Never,
})
const GetInferenceInstances = Query.make("GetInferenceInstances", {
  payload: Empty,
  success: S.ModelInstancesSnapshot,
  error: Never,
  staleTime: Infinity,
  gcTime: Infinity,
})
const GetInferenceInstance = Query.make("GetInferenceInstance", {
  payload: Schema.Struct({ instanceId: Schema.String }),
  success: S.ModelInstance,
  error: Never,
})
const GetInferenceResidencyPolicy = Query.make("GetInferenceResidencyPolicy", {
  payload: Empty,
  success: S.ModelResidencyPolicyResponse,
  error: Never,
  staleTime: Infinity,
  gcTime: Infinity,
})
const AssessInferenceModels = Query.make("AssessInferenceModels", {
  payload: S.AssessModelsRequest,
  success: S.AssessModelsResponse,
  error: Never,
})
const PreviewInferenceModelLoad = Query.make("PreviewInferenceModelLoad", {
  payload: Schema.Struct({ modelId: Schema.String }),
  success: S.ModelLoadPlan,
  error: Never,
})
const GetInferenceModelProperties = Query.make("GetInferenceModelProperties", {
  payload: Schema.Struct({ modelId: Schema.String }),
  success: S.PropsResponse,
  error: Never,
})
const ApplyInferenceChatTemplate = Query.make("ApplyInferenceChatTemplate", {
  payload: S.ApplyTemplateRequest,
  success: S.ApplyTemplateResponse,
  error: Never,
})
const SearchHuggingFaceModels = Query.make("SearchHuggingFaceModels", {
  payload: S.HuggingFaceModelSearchRequest,
  success: S.HuggingFaceModelSearchResults,
  error: Never,
})
const ResolveHuggingFaceRepository = Query.make("ResolveHuggingFaceRepository", {
  payload: S.HuggingFaceRepositoryRequest,
  success: S.HuggingFaceRepositorySnapshot,
  error: Never,
})

const invalidateInferenceQueries = (...queries: ReadonlyArray<Operation.Any>) =>
  Effect.forEach(
    queries,
    (query) => QueryClient.invalidate({ name: Operation.declaration(query).name }),
    { discard: true },
  )

const refreshInferenceQuery = <Input, Output, Error, Requirements>(
  query: Query.Query<Input, Output, Error, Requirements>,
  input: Input,
) => QueryClient.refetch(query.match(input)).pipe(
  // Synchronization must await the authoritative reread. Invalidation alone
  // intentionally schedules a refresh without waiting for it.
  Effect.zipRight(QueryClient.fetch(query, input)),
)

export class InferenceMutationSynchronizationFailed extends Data.TaggedError(
  "InferenceMutationSynchronizationFailed",
)<{
  readonly operation: string
  readonly message: string
}> {}

const synchronizationFailed = (operation: string, message: string) =>
  new InferenceMutationSynchronizationFailed({ operation, message })

const inferenceInvalidationDependencies = {
  hardware: [
    GetInferenceHardware,
    AssessInferenceModels,
    PreviewInferenceModelLoad,
  ],
  models: [
    GetInferenceModels,
    GetInferenceModel,
    GetServableModels,
    AssessInferenceModels,
    PreviewInferenceModelLoad,
    GetInferenceModelProperties,
    ApplyInferenceChatTemplate,
  ],
  packages: [
    GetInferencePackages,
    GetInferencePackage,
  ],
  downloads: [
    GetInferenceDownloads,
    GetInferenceDownload,
  ],
  instances: [
    GetInferenceInstances,
    GetInferenceInstance,
    GetInferenceHardware,
    AssessInferenceModels,
    PreviewInferenceModelLoad,
  ],
  "residency-policy": [GetInferenceResidencyPolicy],
} satisfies Record<S.InferenceResourceTopic, ReadonlyArray<Operation.Any>>

/** Invalidates every Query whose answer can change with the named native resource. */
export const invalidateInferenceTopic = (topic: S.InferenceResourceTopic) =>
  invalidateInferenceQueries(...inferenceInvalidationDependencies[topic])

export const invalidateAllInferenceQueries = () => invalidateInferenceQueries(
  ...new Set(Object.values(inferenceInvalidationDependencies).flat()),
)

const InstallInferenceModel = Mutation.make("InstallInferenceModel", {
  payload: S.ModelIdentityRequest,
  success: S.InstallModelResponse,
  error: Never,
  scope: ({ modelId }) => Mutation.MutationScope(`inference-model:${modelId}`),
  synchronize: (output, { modelId }) => Effect.gen(function* () {
    yield* invalidateInferenceQueries(
      GetInferenceModels,
      GetServableModels,
      GetInferencePackages,
      GetInferencePackage,
      GetInferenceDownloads,
    )
    if (output._tag === "DownloadAdmitted") {
      const download = yield* refreshInferenceQuery(GetInferenceDownload, {
        downloadId: output.downloadId,
      })
      if (download.id !== output.downloadId) {
        return yield* synchronizationFailed(
          "install",
          "The admitted download was absent from the authoritative download resource.",
        )
      }
      return
    }
    const model = yield* refreshInferenceQuery(GetInferenceModel, { modelId })
    if (model.id !== modelId
      || model.localState._tag !== "Installed"
      || model.localState.updateState._tag !== "Current") {
      return yield* synchronizationFailed(
        "install",
        "The model installation acknowledgement was not visible in the authoritative model resource.",
      )
    }
  }),
})
const UninstallInferenceModel = Mutation.make("UninstallInferenceModel", {
  payload: S.ModelIdentityRequest,
  success: S.UninstallModelResponse,
  error: Never,
  scope: ({ modelId }) => Mutation.MutationScope(`inference-model:${modelId}`),
  synchronize: (output, { modelId }) => Effect.gen(function* () {
    if (output.modelId !== modelId) {
      return yield* synchronizationFailed(
        "uninstall",
        "The uninstall response identified a different model.",
      )
    }
    const model = yield* refreshInferenceQuery(GetInferenceModel, { modelId })
    const packages = yield* refreshInferenceQuery(GetInferencePackages, {})
    yield* invalidateInferenceQueries(GetInferenceModels, GetInferencePackage, GetServableModels)
    const installedPackageIds = new Set(packages.packages.map((candidate) => candidate.package.id))
    if (model.localState._tag !== "NotInstalled"
      || output.removedPackageIds.some((packageId) => installedPackageIds.has(packageId))) {
      return yield* synchronizationFailed(
        "uninstall",
        "The model uninstall acknowledgement was not visible in the authoritative resources.",
      )
    }
  }),
})
const RemoveInferencePackage = Mutation.make("RemoveInferencePackage", {
  payload: Schema.Struct({ packageId: Schema.String }),
  success: S.RemoveInstalledModelPackageResponse,
  error: Never,
  scope: ({ packageId }) => Mutation.MutationScope(`inference-package:${packageId}`),
  synchronize: (output, { packageId }) => Effect.gen(function* () {
    if (output.packageId !== packageId) {
      return yield* synchronizationFailed(
        "remove-package",
        "The package-removal response identified a different package.",
      )
    }
    const packages = yield* refreshInferenceQuery(GetInferencePackages, {})
    yield* invalidateInferenceQueries(GetInferenceModels, GetInferenceModel, GetServableModels)
    if (output.removed && packages.packages.some((candidate) => candidate.package.id === packageId)) {
      return yield* synchronizationFailed(
        "remove-package",
        "The removed package remained present in the authoritative package collection.",
      )
    }
  }),
})
const CancelInferenceDownload = Mutation.make("CancelInferenceDownload", {
  payload: Schema.Struct({ downloadId: Schema.String }),
  success: S.ModelDownload,
  error: Never,
  scope: ({ downloadId }) => Mutation.MutationScope(`inference-download:${downloadId}`),
  synchronize: (output, { downloadId }) => Effect.gen(function* () {
    const download = yield* refreshInferenceQuery(GetInferenceDownload, { downloadId })
    yield* invalidateInferenceQueries(GetInferenceDownloads, GetInferenceModels, GetInferenceModel)
    if (output.id !== downloadId || download.id !== downloadId) {
      return yield* synchronizationFailed(
        "cancel-download",
        "The cancelled download was absent from the authoritative download resource.",
      )
    }
  }),
})
const AcknowledgeInferenceDownloadFailure = Mutation.make("AcknowledgeInferenceDownloadFailure", {
  payload: Schema.Struct({ downloadId: Schema.String }),
  success: S.ModelDownload,
  error: Never,
  scope: ({ downloadId }) => Mutation.MutationScope(`inference-download:${downloadId}`),
  synchronize: (output, { downloadId }) => Effect.gen(function* () {
    const download = yield* refreshInferenceQuery(GetInferenceDownload, { downloadId })
    yield* invalidateInferenceQueries(GetInferenceDownloads)
    if (output.id !== downloadId
      || download.id !== downloadId
      || output.state._tag !== "Failed"
      || !output.state.acknowledged
      || download.state._tag !== "Failed"
      || !download.state.acknowledged) {
      return yield* synchronizationFailed(
        "acknowledge-download-failure",
        "The acknowledged failure was absent from the authoritative download resource.",
      )
    }
  }),
})
const StopInferenceInstance = Mutation.make("StopInferenceInstance", {
  payload: Schema.Struct({ instanceId: Schema.String }),
  success: Schema.Void,
  error: Never,
  scope: ({ instanceId }) => Mutation.MutationScope(`inference-instance:${instanceId}`),
  synchronize: (_, { instanceId }) => Effect.gen(function* () {
    const instance = yield* refreshInferenceQuery(GetInferenceInstance, { instanceId })
    yield* invalidateInferenceQueries(GetInferenceInstances, GetInferenceHardware)
    if (instance.id !== instanceId
      || instance.lifecycle._tag === "Loading"
      || instance.lifecycle._tag === "Ready") {
      return yield* synchronizationFailed(
        "stop-instance",
        "The stop acknowledgement was not visible on the authoritative instance resource.",
      )
    }
  }),
})
const EnsureInferenceInstance = Mutation.make("EnsureInferenceInstance", {
  payload: S.EnsureModelInstanceRequest,
  success: S.ModelInstance,
  error: Never,
  scope: ({ modelId }) => Mutation.MutationScope(`inference-model-residency:${modelId}`),
  synchronize: (output, { modelId }) => Effect.gen(function* () {
    const instance = yield* refreshInferenceQuery(GetInferenceInstance, {
      instanceId: output.id,
    })
    yield* invalidateInferenceQueries(GetInferenceInstances, GetInferenceHardware)
    if (output.lifecycle._tag !== "Ready"
      || instance.id !== output.id
      || instance.modelId !== modelId
      || instance.lifecycle._tag !== "Ready") {
      return yield* synchronizationFailed(
        "ensure-instance",
        "The admitted instance was absent from the authoritative instance resource.",
      )
    }
  }),
})
const SetInferenceResidencyPolicy = Mutation.make("SetInferenceResidencyPolicy", {
  payload: S.SetModelResidencyPolicyRequest,
  success: Schema.Void,
  error: Never,
  scope: () => Mutation.MutationScope("inference-residency-policy"),
  synchronize: (_, input) => refreshInferenceQuery(GetInferenceResidencyPolicy, {}).pipe(
    Effect.filterOrFail(
      (policy) => policy.generation === input.generation
        && policy.idleTimeoutSeconds === input.idleTimeoutSeconds,
      () => synchronizationFailed(
        "set-residency-policy",
        "The acknowledged residency policy was absent from the authoritative resource.",
      ),
    ),
    Effect.asVoid,
  ),
})

const WatchInferenceEvents = Subscription.make("WatchInferenceEvents", {
  payload: Empty,
  success: S.InferenceResourceInvalidation,
  error: Never,
})
const StreamInferenceChatCompletion = Subscription.make("StreamInferenceChatCompletion", {
  payload: S.ChatCompletionRequest,
  success: S.ChatCompletionChunk,
  error: Never,
})
const StreamInferenceResponse = Subscription.make("StreamInferenceResponse", {
  payload: S.ResponseCreateRequest,
  success: S.ResponseStreamEvent,
  error: Never,
})

export const Inference = Group.make({
  GetInferenceHardware,
  GetInferenceModels,
  GetInferenceModel,
  GetServableModels,
  GetInferencePackages,
  GetInferencePackage,
  GetInferenceDownloads,
  GetInferenceDownload,
  GetInferenceInstances,
  GetInferenceInstance,
  GetInferenceResidencyPolicy,
  AssessInferenceModels,
  PreviewInferenceModelLoad,
  GetInferenceModelProperties,
  ApplyInferenceChatTemplate,
  SearchHuggingFaceModels,
  ResolveHuggingFaceRepository,
  InstallInferenceModel,
  UninstallInferenceModel,
  RemoveInferencePackage,
  CancelInferenceDownload,
  AcknowledgeInferenceDownloadFailure,
  StopInferenceInstance,
  EnsureInferenceInstance,
  SetInferenceResidencyPolicy,
  WatchInferenceEvents,
  StreamInferenceChatCompletion,
  StreamInferenceResponse,
})

/** One first-party Effect Query boundary over the ACN RPC and inference HTTP transports. */
export const MagnitudeBoundary = Group.extend(AcnBoundary, { Inference })

export type InferenceClientError = GeneratedClientError

export const makeInferenceImplementations = (
  client: IcnApiClient,
): Operation.ImplementationService<InferenceClientError> => {
  const finite = new Map<Operation.Any, (payload: never) => Effect.Effect<unknown, unknown>>([
    [GetInferenceHardware, () => client.system.getHardware({})],
    [GetInferenceModels, () => client.models.listModels({})],
    [GetInferenceModel, ({ modelId }: { readonly modelId: string }) =>
      client.models.getModel({ path: { model_id: modelId } })],
    [GetServableModels, () => client.inference.listServableModels({})],
    [GetInferencePackages, () => client.models.listInstalledModels({})],
    [GetInferencePackage, ({ packageId }: { readonly packageId: string }) =>
      client.models.getInstalledModelPackage({ path: { package_id: packageId } })],
    [GetInferenceDownloads, () => client.models.listModelDownloads({})],
    [GetInferenceDownload, ({ downloadId }: { readonly downloadId: string }) =>
      client.models.getModelDownload({ path: { download_id: downloadId } })],
    [GetInferenceInstances, () => client.models.getModelInstances({})],
    [GetInferenceInstance, ({ instanceId }: { readonly instanceId: string }) =>
      client.models.getModelInstance({ path: { instance_id: instanceId } })],
    [GetInferenceResidencyPolicy, () => client.models.getModelResidencyPolicy({})],
    [AssessInferenceModels, (payload: typeof S.AssessModelsRequest.Type) => client.models.assessModels({ payload })],
    [PreviewInferenceModelLoad, ({ modelId }: { readonly modelId: string }) =>
      client.models.previewModelLoad({ path: { model_id: modelId } })],
    [GetInferenceModelProperties, ({ modelId }: { readonly modelId: string }) =>
      client.models.getModelProperties({ path: { model_id: modelId } })],
    [ApplyInferenceChatTemplate, (payload: typeof S.ApplyTemplateRequest.Type) => client.chat.applyChatTemplate({ payload })],
    [SearchHuggingFaceModels, (payload: typeof S.HuggingFaceModelSearchRequest.Type) =>
      client.huggingFace.searchHuggingFaceModels({ payload })],
    [ResolveHuggingFaceRepository, (payload: typeof S.HuggingFaceRepositoryRequest.Type) =>
      client.huggingFace.resolveHuggingFaceRepository({ payload })],
    [InstallInferenceModel, (payload: typeof S.ModelIdentityRequest.Type) => client.models.installModel({ payload })],
    [UninstallInferenceModel, (payload: typeof S.ModelIdentityRequest.Type) => client.models.uninstallModel({ payload })],
    [RemoveInferencePackage, ({ packageId }: { readonly packageId: string }) => client.models.removeInstalledModel({ path: { package_id: packageId } })],
    [CancelInferenceDownload, ({ downloadId }: { readonly downloadId: string }) => client.models.cancelModelDownload({ path: { download_id: downloadId } })],
    [AcknowledgeInferenceDownloadFailure, ({ downloadId }: { readonly downloadId: string }) =>
      client.models.acknowledgeModelDownloadFailure({ path: { download_id: downloadId } })],
    [StopInferenceInstance, ({ instanceId }: { readonly instanceId: string }) => client.models.stopModelInstance({ path: { instance_id: instanceId } })],
    [EnsureInferenceInstance, (payload: typeof S.EnsureModelInstanceRequest.Type) =>
      client.models.ensureModelInstance({ payload })],
    [SetInferenceResidencyPolicy, (payload: typeof S.SetModelResidencyPolicyRequest.Type) => client.models.setModelResidencyPolicy({ payload })],
  ] as never)
  const streaming = new Map<Operation.Any, (payload: never) => Stream.Stream<unknown, unknown>>([
    [WatchInferenceEvents, () => Stream.unwrap(
      Effect.map(client.system.watchInferenceEvents({
        urlParams: { topics: Option.none() },
      }), ({ events }) => events),
    )],
    [StreamInferenceChatCompletion, (payload: typeof S.ChatCompletionRequest.Type) => Stream.unwrap(
      Effect.map(client.chat.createChatCompletion({
        payload,
        headers: { "Magnitude-Include-Progress": Option.some(true) },
      }), ({ events }) => events),
    )],
    [StreamInferenceResponse, (payload: typeof S.ResponseCreateRequest.Type) => Stream.unwrap(
      Effect.map(client.inference.createResponse({
        payload,
        headers: { "Magnitude-Include-Progress": Option.some(true) },
      }), ({ events }) => events),
    )],
  ] as never)
  return {
    execute: (operation, payload) => {
      const implementation = finite.get(operation)
      return implementation === undefined
        ? Effect.dieMessage(`No HTTP implementation for ${Operation.declaration(operation).name}`)
        : implementation(payload as never)
    },
    stream: (operation, payload) => {
      const implementation = streaming.get(operation)
      return implementation === undefined
        ? Stream.dieMessage(`No HTTP stream implementation for ${Operation.declaration(operation).name}`)
        : implementation(payload as never)
    },
  }
}

export const inferenceImplementationsLayer = (
  client: IcnApiClient,
): Layer.Layer<Operation.Implementations<InferenceClientError>> =>
  Layer.succeed(Operation.implementationsTag<InferenceClientError>(), makeInferenceImplementations(client))

export type MagnitudeImplementationError = RpcClientError

export const inferenceClientErrorMessage = (cause: unknown): string => {
  if (typeof cause !== "object" || cause === null || !("_tag" in cause)) return String(cause)
  if (cause._tag === "GeneratedClientRemoteError"
    && "body" in cause
    && typeof cause.body === "object"
    && cause.body !== null
    && "error" in cause.body
    && typeof cause.body.error === "object"
    && cause.body.error !== null
    && "message" in cause.body.error
    && typeof cause.body.error.message === "string") {
    return cause.body.error.message
  }
  if (cause._tag === "GeneratedClientInvalidResponseError"
    && "message" in cause
    && typeof cause.message === "string") return cause.message
  if (cause._tag === "GeneratedClientTransportError"
    && "cause" in cause
    && cause.cause instanceof Error) {
    return cause.cause.message
  }
  if ("message" in cause && typeof cause.message === "string" && cause.message.length > 0) {
    return cause.message
  }
  return "operationId" in cause && typeof cause.operationId === "string"
    ? `Inference HTTP request failed: ${cause.operationId}`
    : String(cause)
}

const inferenceRpcError = (cause: InferenceClientError) => new RpcClientError({
  reason: "Protocol",
  message: inferenceClientErrorMessage(cause),
  cause,
})

export const magnitudeImplementationsLayer = (
  protocolLayer: Layer.Layer<RpcClient.Protocol>,
  inferenceBaseUrl: string | URL = MAGNITUDE_INFERENCE_BASE_URL,
): Layer.Layer<Operation.Implementations<MagnitudeImplementationError>, never, HttpClient.HttpClient> =>
  Layer.scoped(
    Operation.implementationsTag<MagnitudeImplementationError>(),
    Effect.gen(function* () {
      const rpc = yield* AcnRpc.makeImplementations(AcnBoundary).pipe(
        Effect.provide(protocolLayer),
      )
      const inference = makeInferenceImplementations(
        yield* makeIcnApiClient({ baseUrl: inferenceBaseUrl }),
      )
      const inferenceOperations = new Set<Operation.Any>(Group.operations(Inference))
      return {
        execute: (operation, payload) => inferenceOperations.has(operation)
          ? inference.execute(operation, payload).pipe(
            Effect.mapError((cause) => inferenceRpcError(cause as InferenceClientError)),
          )
          : rpc.execute(operation, payload),
        stream: (operation, payload) => inferenceOperations.has(operation)
          ? inference.stream(operation, payload).pipe(
            Stream.mapError((cause) => inferenceRpcError(cause as InferenceClientError)),
          )
          : rpc.stream(operation, payload),
      }
    }),
  )
