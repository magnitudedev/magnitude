import { Context, Effect, Layer, Option } from "effect"
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
import { IcnClient, IcnInstances, type IcnClientService } from "@magnitudedev/icn"
import { projectInferenceLoadPlan } from "@magnitudedev/sdk"
import { LocalModels } from "./local-models"
import { LocalModelSyncs } from "./local-model-syncs"
import { ModelSlotController } from "./model-slot-controller"

export interface ModelCommandsApi {
  readonly sync: (modelId: ProviderModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly cancelSync: (modelId: ProviderModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly acknowledgeSyncFailure: (modelId: ProviderModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly remove: (modelId: ProviderModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly loadSlot: (slotId: SlotId) => Effect.Effect<{}, LocalInferenceError>
  readonly previewSlotLoad: (slotId: SlotId) => Effect.Effect<ModelLoadPlan, LocalInferenceError>
  readonly stopSlot: (slotId: SlotId) => Effect.Effect<{}, LocalInferenceError>
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
  IcnClient | ModelSlotController | IcnInstances | LocalModels | LocalModelSyncs
> = Layer.effect(ModelCommands, Effect.gen(function* () {
  const client = yield* IcnClient
  const slots = yield* ModelSlotController
  const instances = yield* IcnInstances
  const localModels = yield* LocalModels
  const syncs = yield* LocalModelSyncs
  const modelMutationLock = yield* Effect.makeSemaphore(1)

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
  const admitSync = (modelId: ProviderModelId) => client.models.installModel({
    payload: { modelId },
  }).pipe(
    Effect.mapError((cause) => modelCommandFailure("install_model", cause)),
    Effect.tap((result) => result._tag === "DownloadAdmitted"
      ? syncs.admitted(modelId, ModelDownloadIdSchema.make(result.downloadId))
      : syncs.current(modelId)),
  )

  const sync = (modelId: ProviderModelId) => modelMutationLock.withPermits(1)(
    syncs.download(modelId).pipe(
      Effect.flatMap(Option.match({
        onNone: () => admitSync(modelId),
        onSome: (download) => download.state._tag === "Pending"
            || download.state._tag === "Downloading"
          ? Effect.succeed({ _tag: "DownloadAdmitted" as const, downloadId: download.id })
          : admitSync(modelId),
      })),
      Effect.andThen(localModels.refresh),
      Effect.as({}),
    ),
  )
  const cancelSync = (modelId: ProviderModelId) => modelMutationLock.withPermits(1)(
    syncs.download(modelId).pipe(
      Effect.flatMap(Option.match({
        onNone: () => Effect.void,
        onSome: (download) => download.state._tag === "Pending" || download.state._tag === "Downloading"
          ? client.models.cancelModelDownload({ path: { download_id: download.id } }).pipe(
              Effect.mapError((cause) => modelCommandFailure("cancel_download", cause)),
            )
          : Effect.void,
      })),
      Effect.andThen(localModels.refresh),
      Effect.as({}),
    ),
  )
  const acknowledgeSyncFailure = (modelId: ProviderModelId) => modelMutationLock.withPermits(1)(
    syncs.download(modelId).pipe(
      Effect.flatMap(Option.match({
        onNone: () => Effect.void,
        onSome: (download) => download.state._tag === "Failed" && !download.state.acknowledged
          ? client.models.acknowledgeModelDownloadFailure({ path: { download_id: download.id } }).pipe(
              Effect.mapError((cause) => modelCommandFailure("acknowledge_download_failure", cause)),
            )
          : Effect.void,
      })),
      Effect.andThen(localModels.refresh),
      Effect.as({}),
    ),
  )

  return ModelCommands.of({
    sync,
    cancelSync,
    acknowledgeSyncFailure,
    remove: (modelId) => modelMutationLock.withPermits(1)(
      Effect.uninterruptible(localModels.removalStarted(modelId).pipe(
        Effect.andThen(client.models.uninstallModel({ payload: { modelId } }).pipe(
          Effect.mapError((cause) => modelCommandFailure("uninstall_model", cause)),
        )),
        Effect.andThen(localModels.removalFinished(modelId)),
        Effect.tapError((failure) => localModels.removalFailed(modelId, failure)),
        Effect.as({}),
      )),
    ),
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
  })
}))
