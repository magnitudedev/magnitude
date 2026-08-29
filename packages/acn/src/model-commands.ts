import { Cause, Context, Effect, Layer, Option, Stream } from "effect"
import {
  LocalModelMutationFailed,
  ModelDownloadIdSchema,
  ModelSlotMutationRejected,
  PRIMARY_SLOT_ID,
  type LocalInferenceError,
  type ModelLoadPlan,
  type SlotId,
} from "@magnitudedev/acn-protocol"
import type { ProviderModelId } from "@magnitudedev/ai"
import { IcnClient, IcnDownloads, IcnInstances, IcnModels, type IcnClientService } from "@magnitudedev/icn"
import { projectInferenceLoadPlan } from "@magnitudedev/sdk"
import { LocalModelAcquisitionCoordinator } from "./local-model-acquisition-coordinator"
import { LocalModelPackages } from "./local-model-packages"
import { ModelSlotController } from "./model-slot-controller"
import { ModelCatalog } from "./model-catalog"

export type LocalModelSyncOutcome = "Started" | "AlreadyCurrent"

export interface ModelCommandsApi {
  readonly sync: (modelId: ProviderModelId) => Effect.Effect<{
    readonly outcome: LocalModelSyncOutcome
  }, LocalInferenceError>
  readonly cancelSync: (modelId: ProviderModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly acknowledgeSyncFailure: (modelId: ProviderModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly remove: (modelId: ProviderModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly loadSlot: (slotId: SlotId) => Effect.Effect<{}, LocalInferenceError>
  readonly previewSlotLoad: (slotId: SlotId) => Effect.Effect<ModelLoadPlan, LocalInferenceError>
  readonly stopSlot: (slotId: SlotId) => Effect.Effect<{}, LocalInferenceError>
  readonly loadModel: (modelId: ProviderModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly stopActiveModel: Effect.Effect<{}, LocalInferenceError>
}

export class ModelCommands extends Context.Tag("ModelCommands")<ModelCommands, ModelCommandsApi>() {}

type IcnModelCommandError = Effect.Effect.Error<
  ReturnType<IcnClientService["models"]["ensureModelInstance"]>
>

const externalFailureMessage = (cause: unknown): string => {
  if (cause instanceof Error && cause.message.length > 0) return cause.message
  if (typeof cause === "object" && cause !== null && "message" in cause
    && typeof cause.message === "string" && cause.message.length > 0) return cause.message
  return "The local inference service did not provide failure details"
}

export const modelCommandFailure = (
  operation: string,
  cause: IcnModelCommandError,
): LocalModelMutationFailed => {
  switch (cause._tag) {
    case "GeneratedClientRemoteError": return new LocalModelMutationFailed({
      code: cause.body.error.code,
      message: cause.body.error.message,
      retryable: cause.status === 409 || cause.status >= 500,
    })
    case "GeneratedClientTransportError": return new LocalModelMutationFailed({
      code: `model_${operation}_transport_failed`,
      message: externalFailureMessage(cause.cause),
      retryable: true,
    })
    case "GeneratedClientInputError": return new LocalModelMutationFailed({
      code: `model_${operation}_request_invalid`,
      message: `Invalid local inference request input at ${cause.location}`,
      retryable: false,
    })
    case "GeneratedClientInvalidResponseError": return new LocalModelMutationFailed({
      code: `model_${operation}_response_invalid`,
      message: cause.message,
      retryable: true,
    })
  }
}

export const ModelCommandsLive: Layer.Layer<
  ModelCommands,
  never,
  IcnClient | IcnDownloads | IcnModels | ModelSlotController | IcnInstances
    | LocalModelPackages | LocalModelAcquisitionCoordinator | ModelCatalog
> = Layer.scoped(ModelCommands, Effect.gen(function* () {
  const client = yield* IcnClient
  const downloads = yield* IcnDownloads
  const models = yield* IcnModels
  const slots = yield* ModelSlotController
  const instances = yield* IcnInstances
  const packages = yield* LocalModelPackages
  const acquisition = yield* LocalModelAcquisitionCoordinator
  const catalog = yield* ModelCatalog

  const localProduct = (modelId: ProviderModelId) => catalog.state.pipe(
    Effect.flatMap((state) => {
      const entry = state._tag === "Initializing" ? undefined : state.models.find((candidate) =>
        candidate._tag === "Local" && candidate.product.modelId === modelId)
      return entry?._tag === "Local"
        ? Effect.succeed(entry.product)
        : Effect.fail(new LocalModelMutationFailed({
            code: "model_not_found",
            message: `Unknown local model: ${modelId}`,
            retryable: false,
          }))
    }),
  )
  const selectedLocalModel = (slotId: SlotId) => slots.state.pipe(
    Effect.map((state) => state.slots[slotId === PRIMARY_SLOT_ID ? "primary" : "secondary"]),
    Effect.filterOrFail(
      (slot) => slot._tag === "ConfiguredLocal",
      () => new ModelSlotMutationRejected({
        slotId,
        message: "The slot does not contain a configured local model",
      }),
    ),
    Effect.map((slot) => slot.selection.providerModelId),
  )
  const nativeDownload = (downloadId: string) => downloads.get.pipe(
    Effect.map(({ state }) => state.downloads.find(({ id }) => id === downloadId)),
  )
  const cancelDownload = (downloadId: string) => client.models.cancelModelDownload({
    path: { download_id: downloadId },
  }).pipe(Effect.mapError((cause) => modelCommandFailure("cancel_download", cause)))

  const failureMessage = (cause: Cause.Cause<unknown>) => Option.match(
    Cause.failureOption(cause),
    {
      onNone: () => Cause.pretty(cause),
      onSome: externalFailureMessage,
    },
  )
  const removalFailure = (cause: Cause.Cause<unknown>): LocalModelMutationFailed => Option.match(
    Cause.failureOption(cause),
    {
      onNone: () => new LocalModelMutationFailed({
        code: "model_uninstall_model_failed",
        message: Cause.pretty(cause),
        retryable: true,
      }),
      onSome: (failure) => failure instanceof LocalModelMutationFailed
        ? failure
        : new LocalModelMutationFailed({
            code: "model_uninstall_model_failed",
            message: externalFailureMessage(failure),
            retryable: true,
          }),
    },
  )

  const runSync = (modelId: ProviderModelId, generation: number) => Effect.gen(function* () {
    const admission = yield* client.models.installModel({ payload: { modelId } }).pipe(
      Effect.mapError((cause) => modelCommandFailure("install_model", cause)),
    )
    if (admission._tag === "Current") {
      yield* acquisition.finishSync(modelId, generation)
      return
    }
    const downloadId = ModelDownloadIdSchema.make(admission.downloadId)
    const correlation = yield* acquisition.correlateSync(modelId, generation, downloadId)
    if (Option.isSome(correlation) && !correlation.value) return
    yield* cancelDownload(downloadId).pipe(
      Effect.zipRight(packages.refresh),
      Effect.zipRight(acquisition.finishSync(modelId, generation)),
    )
  }).pipe(Effect.onError((cause) => acquisition.failSyncAdmission(modelId, generation, {
    _tag: "Internal",
    message: failureMessage(cause),
  })))

  const retireTerminalSynchronizations = Effect.gen(function* () {
    const coordination = yield* acquisition.state
    if (coordination.syncs.size === 0) return
    const nativeDownloads = (yield* downloads.get).state.downloads
    const nativeModels = (yield* models.get).state.models
    const downloadsById = new Map(nativeDownloads.map((download) => [download.id, download]))
    const modelsById = new Map(nativeModels.map((model) => [model.id, model]))
    yield* Effect.forEach(coordination.syncs, ([modelId, sync]) => {
      if (sync._tag !== "Correlated") return Effect.void
      const download = downloadsById.get(sync.downloadId)
      if (download?.state._tag === "Cancelled"
        || (download?.state._tag === "Failed" && download.state.acknowledged)) {
        return acquisition.finishSync(modelId, sync.generation)
      }
      const model = modelsById.get(modelId)
      const isCurrent = model?.localState._tag === "Installed"
        && model.localState.updateState._tag === "Current"
      return download?.state._tag === "Completed" && isCurrent
        ? acquisition.finishSync(modelId, sync.generation)
        : Effect.void
    }, { discard: true })
  })

  yield* Stream.merge(
    downloads.changes.pipe(Stream.map(() => undefined)),
    models.changes.pipe(Stream.map(() => undefined)),
  ).pipe(
    Stream.runForEach(() => retireTerminalSynchronizations),
    Effect.forkScoped,
  )

  const sync = (modelId: ProviderModelId) => Effect.uninterruptibleMask((restore) => Effect.gen(function* () {
    const product = yield* restore(localProduct(modelId))
    if (product.acquisitionState._tag === "Installed") {
      return { outcome: "AlreadyCurrent" as const }
    }
    if (product.acquisitionState._tag === "Removing"
      || product.acquisitionState._tag === "RemoveFailed") {
      return yield* new LocalModelMutationFailed({
        code: "model_not_pullable",
        message: `${modelId} cannot be pulled from state ${product.acquisitionState._tag}`,
        retryable: false,
      })
    }
    const replaceGeneration = yield* restore(Effect.gen(function* () {
      const existing = (yield* acquisition.state).syncs.get(modelId)
      const existingDownload = existing?._tag === "Correlated"
        ? yield* nativeDownload(existing.downloadId)
        : undefined
      if (existing === undefined) return Option.none<number>()
      const replace = existing._tag === "AdmissionFailed"
        || (existingDownload !== undefined
          && existingDownload.state._tag !== "Pending"
          && existingDownload.state._tag !== "Downloading")
      return replace ? Option.some(existing.generation) : Option.none<number>()
    }))
    const admitted = yield* acquisition.admitSync(modelId, replaceGeneration)
    if (Option.isNone(admitted)) {
      return yield* new LocalModelMutationFailed({
        code: "model_sync_busy",
        message: `${modelId} already has active acquisition work`,
        retryable: true,
      })
    }
    yield* runSync(modelId, admitted.value)
    return { outcome: "Started" as const }
  }))

  const cancelSync = (modelId: ProviderModelId) => Effect.gen(function* () {
    const existing = (yield* acquisition.state).syncs.get(modelId)
    if (existing === undefined || existing._tag === "AdmissionFailed"
      || (existing._tag === "Admitting" && existing.cancelRequested)) {
      return yield* new LocalModelMutationFailed({
        code: "model_sync_not_active",
        message: `${modelId} has no cancellable pull`,
        retryable: false,
      })
    }
    const cancellation = yield* acquisition.requestSyncCancellation(modelId)
    if (Option.isNone(cancellation)) {
      // Admission has not produced its native download identity yet. The
      // coordinator records intent and runSync cancels immediately after
      // correlation, without exposing that native identity to the caller.
      return {}
    }
    yield* cancelDownload(cancellation.value.downloadId)
    yield* packages.refresh
    yield* acquisition.finishSync(modelId, cancellation.value.generation)
    return {}
  })

  const acknowledgeSyncFailure = (modelId: ProviderModelId) => Effect.gen(function* () {
    const syncState = (yield* acquisition.state).syncs.get(modelId)
    if (syncState?._tag === "AdmissionFailed") {
      yield* acquisition.finishSync(modelId, syncState.generation)
      return {}
    }
    if (syncState?._tag !== "Correlated") return {}
    const download = yield* nativeDownload(syncState.downloadId)
    if (download?.state._tag === "Failed" && !download.state.acknowledged) {
      yield* client.models.acknowledgeModelDownloadFailure({
        path: { download_id: syncState.downloadId },
      }).pipe(Effect.mapError((cause) => modelCommandFailure("acknowledge_download_failure", cause)))
      yield* packages.refresh
    }
    yield* acquisition.finishSync(modelId, syncState.generation)
    return {}
  })

  return ModelCommands.of({
    sync,
    cancelSync,
    acknowledgeSyncFailure,
    remove: (modelId) => Effect.uninterruptible(Effect.gen(function* () {
      const product = yield* localProduct(modelId)
      const state = product.acquisitionState
      if (!("packages" in state) || state._tag === "Removing") {
        return yield* new LocalModelMutationFailed({
          code: "model_not_removable",
          message: `${modelId} cannot be removed from state ${state._tag}`,
          retryable: false,
        })
      }
      if (state.residencyState._tag !== "Unloaded" && state.residencyState._tag !== "Failed") {
        return yield* new LocalModelMutationFailed({
          code: "model_active",
          message: `${modelId} is active; run \`magnitude models stop\` before removing it`,
          retryable: false,
        })
      }
      if ((yield* acquisition.state).syncs.has(modelId)) {
        return yield* new LocalModelMutationFailed({
          code: "model_sync_active",
          message: `${modelId} has an active pull; cancel it first`,
          retryable: false,
        })
      }
      const admitted = yield* acquisition.admitRemoval(modelId)
      if (Option.isNone(admitted)) {
        return yield* new LocalModelMutationFailed({
          code: "model_removal_active",
          message: `${modelId} is already being removed`,
          retryable: false,
        })
      }
      const generation = admitted.value
      const remove = client.models.uninstallModel({ payload: { modelId } }).pipe(
        Effect.mapError((cause) => modelCommandFailure("uninstall_model", cause)),
        Effect.zipRight(packages.refresh),
        Effect.zipRight(acquisition.finishRemoval(modelId, generation)),
        Effect.onError((cause) => acquisition.failRemoval(
          modelId,
          generation,
          removalFailure(cause),
        )),
      )
      yield* remove
      return {}
    })),
    loadSlot: (slotId) => selectedLocalModel(slotId).pipe(
      Effect.flatMap((modelId) => client.models.ensureModelInstance({ payload: { modelId } })),
      Effect.tap(() => slots.refresh),
      Effect.as({}),
      Effect.mapError((cause) => cause instanceof ModelSlotMutationRejected
        ? cause
        : modelCommandFailure("load_model", cause)),
    ),
    previewSlotLoad: (slotId) => selectedLocalModel(slotId).pipe(
      Effect.flatMap((modelId) => client.models.previewModelLoad({ path: { model_id: modelId } })),
      Effect.map(projectInferenceLoadPlan),
      Effect.mapError((cause) => cause instanceof ModelSlotMutationRejected
        ? cause
        : modelCommandFailure("preview_model_load", cause)),
    ),
    stopSlot: (slotId) => Effect.gen(function* () {
      const modelId = yield* selectedLocalModel(slotId)
      const instance = (yield* instances.get).instances.findLast((candidate) =>
        candidate.modelId === modelId
        && (candidate.lifecycle._tag === "Loading" || candidate.lifecycle._tag === "Ready"))
      if (instance === undefined) {
        return yield* new ModelSlotMutationRejected({
          slotId,
          message: "The selected model has no active instance",
        })
      }
      yield* client.models.stopModelInstance({ path: { instance_id: instance.id } }).pipe(
        Effect.mapError((cause) => modelCommandFailure("stop_model", cause)),
      )
      yield* slots.refresh
      return {}
    }),
    loadModel: (modelId) => Effect.gen(function* () {
      const product = yield* localProduct(modelId)
      if (!("packages" in product.acquisitionState)
        || product.acquisitionState._tag === "Removing") {
        return yield* new LocalModelMutationFailed({
            code: "model_not_installed",
            message: `${modelId} is not installed`,
            retryable: false,
          })
      }
      yield* client.models.ensureModelInstance({ payload: { modelId } }).pipe(
        Effect.mapError((cause) => modelCommandFailure("load_model", cause)),
      )
      return {}
    }),
    stopActiveModel: Effect.gen(function* () {
      const instance = (yield* instances.get).instances.findLast((candidate) =>
        candidate.lifecycle._tag === "Loading" || candidate.lifecycle._tag === "Ready")
      if (instance === undefined) {
        return yield* new LocalModelMutationFailed({
          code: "no_active_model",
          message: "There is no active local model to stop",
          retryable: false,
        })
      }
      yield* client.models.stopModelInstance({ path: { instance_id: instance.id } }).pipe(
        Effect.mapError((cause) => modelCommandFailure("stop_model", cause)),
      )
      return {}
    }),
  })
}))
