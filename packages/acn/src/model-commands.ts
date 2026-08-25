import { Context, Effect, Layer } from "effect"
import {
  LocalModelMutationFailed,
  ModelDownloadIdSchema,
  ModelInstanceIdSchema,
  ModelSlotMutationRejected,
  PRIMARY_SLOT_ID,
  type CatalogModelReconciliationAdmission,
  type LocalInferenceError,
  type ModelDownloadId,
  type ModelInstanceId,
  type ModelLoadPlan,
  type SlotId,
} from "@magnitudedev/acn-protocol"
import type { ProviderModelId } from "@magnitudedev/ai"
import { IcnClient } from "@magnitudedev/icn"
import { projectInferenceLoadPlan } from "@magnitudedev/sdk"
import { ModelInstances } from "./model-instances"
import { ModelSlotController } from "./model-slot-controller"

export interface ModelCommandsApi {
  readonly install: (modelId: ProviderModelId) => Effect.Effect<CatalogModelReconciliationAdmission, LocalInferenceError>
  readonly cancelDownload: (downloadId: ModelDownloadId) => Effect.Effect<{}, LocalInferenceError>
  readonly acknowledgeDownloadFailure: (downloadId: ModelDownloadId) => Effect.Effect<{}, LocalInferenceError>
  readonly uninstall: (modelId: ProviderModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly loadSlot: (slotId: SlotId) => Effect.Effect<{ readonly instanceId: ModelInstanceId }, LocalInferenceError>
  readonly previewSlotLoad: (slotId: SlotId) => Effect.Effect<ModelLoadPlan, LocalInferenceError>
  readonly stopSlot: (slotId: SlotId) => Effect.Effect<{ readonly instanceId: ModelInstanceId }, LocalInferenceError>
}

export class ModelCommands extends Context.Tag("ModelCommands")<ModelCommands, ModelCommandsApi>() {}

const failed = (operation: string, cause: unknown) => new LocalModelMutationFailed({
  code: `model_${operation}_failed`,
  message: `Unable to ${operation.replaceAll("_", " ")}: ${String(cause)}`,
  retryable: true,
})

export const ModelCommandsLive: Layer.Layer<
  ModelCommands,
  never,
  IcnClient | ModelSlotController | ModelInstances
> = Layer.effect(ModelCommands, Effect.gen(function* () {
  const client = yield* IcnClient
  const slots = yield* ModelSlotController
  const instances = yield* ModelInstances

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
        : {
            _tag: "DownloadAdmitted",
            providerModelId: modelId,
            downloadId: ModelDownloadIdSchema.make(result.downloadId),
          }),
      Effect.mapError((cause) => failed("install_model", cause)),
    ),
    cancelDownload: (downloadId) => client.models.cancelModelDownload({
      path: { download_id: downloadId },
    }).pipe(Effect.as({}), Effect.mapError((cause) => failed("cancel_download", cause))),
    acknowledgeDownloadFailure: (downloadId) => client.models.acknowledgeModelDownloadFailure({
      path: { download_id: downloadId },
    }).pipe(Effect.as({}), Effect.mapError((cause) => failed("acknowledge_download_failure", cause))),
    uninstall: (modelId) => client.models.uninstallModel({ payload: { modelId } }).pipe(
      Effect.as({}),
      Effect.mapError((cause) => failed("uninstall_model", cause)),
    ),
    loadSlot: (slotId) => selectedLocalModel(slotId).pipe(
      Effect.flatMap((modelId) => client.models.ensureModelInstance({ payload: { modelId } })),
      Effect.tap(() => slots.refresh),
      Effect.map((instance) => ({ instanceId: ModelInstanceIdSchema.make(instance.id) })),
      Effect.mapError((cause) => cause instanceof ModelSlotMutationRejected
        ? cause
        : failed("load_model", cause)),
    ),
    previewSlotLoad: (slotId) => selectedLocalModel(slotId).pipe(
      Effect.flatMap((modelId) => client.models.previewModelLoad({ path: { model_id: modelId } })),
      Effect.map(projectInferenceLoadPlan),
      Effect.mapError((cause) => cause instanceof ModelSlotMutationRejected
        ? cause
        : failed("preview_model_load", cause)),
    ),
    stopSlot: (slotId) => Effect.gen(function* () {
      const modelId = yield* selectedLocalModel(slotId)
      const instance = (yield* instances.state).instances.findLast((candidate) =>
        candidate.modelId === modelId
        && (candidate.residency._tag === "Loading" || candidate.residency._tag === "Ready"))
      if (instance === undefined) {
        return yield* new ModelSlotMutationRejected({
          slotId,
          message: "The selected model has no active instance",
        })
      }
      yield* client.models.stopModelInstance({ path: { instance_id: instance.instanceId } }).pipe(
        Effect.mapError((cause) => failed("stop_model", cause)),
      )
      yield* slots.refresh
      return { instanceId: instance.instanceId }
    }),
  })
}))
