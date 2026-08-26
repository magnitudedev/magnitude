import { Context, Effect, Layer, Option } from "effect"
import {
  LocalModelMutationFailed,
  ModelSlotMutationRejected,
  PRIMARY_SLOT_ID,
  type CatalogModelReconciliationAdmission,
  type LocalInferenceError,
  type ModelLoadPlan,
  type SlotId,
} from "@magnitudedev/acn-protocol"
import type { ProviderModelId } from "@magnitudedev/ai"
import { IcnClient, IcnInstances, type IcnClientService } from "@magnitudedev/icn"
import { projectInferenceLoadPlan } from "@magnitudedev/sdk"
import { LocalModels } from "./local-models"
import { ModelSlotController } from "./model-slot-controller"

export interface ModelCommandsApi {
  readonly install: (modelId: ProviderModelId) => Effect.Effect<CatalogModelReconciliationAdmission, LocalInferenceError>
  readonly cancelDownload: (modelId: ProviderModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly acknowledgeDownloadFailure: (modelId: ProviderModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly uninstall: (modelId: ProviderModelId) => Effect.Effect<{}, LocalInferenceError>
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

const rejected = (modelId: ProviderModelId, message: string) => new LocalModelMutationFailed({
  code: "model_download_absent",
  message: `${message} for ${modelId}`,
  retryable: false,
})

export const ModelCommandsLive: Layer.Layer<
  ModelCommands,
  never,
  IcnClient | ModelSlotController | IcnInstances | LocalModels
> = Layer.effect(ModelCommands, Effect.gen(function* () {
  const client = yield* IcnClient
  const slots = yield* ModelSlotController
  const instances = yield* IcnInstances
  const localModels = yield* LocalModels

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

  return ModelCommands.of({
    install: (modelId) => client.models.installModel({ payload: { modelId } }).pipe(
      Effect.map((result): CatalogModelReconciliationAdmission => result._tag === "Current"
        ? { _tag: "Current", providerModelId: modelId }
        : { _tag: "DownloadAdmitted", providerModelId: modelId }),
      Effect.mapError((cause) => modelCommandFailure("install_model", cause)),
    ),
    cancelDownload: (modelId) => localModels.currentDownload(modelId).pipe(
      Effect.flatMap(Option.match({
        onNone: () => rejected(modelId, "No download is running"),
        onSome: (download) => download.state._tag === "Pending" || download.state._tag === "Downloading"
          ? client.models.cancelModelDownload({ path: { download_id: download.id } }).pipe(
              Effect.mapError((cause) => modelCommandFailure("cancel_download", cause)),
            )
          : rejected(modelId, "No download is running"),
      })),
      Effect.as({}),
    ),
    acknowledgeDownloadFailure: (modelId) => localModels.currentDownload(modelId).pipe(
      Effect.flatMap(Option.match({
        onNone: () => rejected(modelId, "No failed download is awaiting acknowledgement"),
        onSome: (download) => download.state._tag === "Failed" && !download.state.acknowledged
          ? client.models.acknowledgeModelDownloadFailure({ path: { download_id: download.id } }).pipe(
              Effect.mapError((cause) => modelCommandFailure("acknowledge_download_failure", cause)),
            )
          : rejected(modelId, "No failed download is awaiting acknowledgement"),
      })),
      Effect.as({}),
    ),
    uninstall: (modelId) => client.models.uninstallModel({ payload: { modelId } }).pipe(
      Effect.as({}),
      Effect.mapError((cause) => modelCommandFailure("uninstall_model", cause)),
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
