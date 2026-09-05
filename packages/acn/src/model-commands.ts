import { Context, Effect, Layer, Schema } from "effect"
import {
  ModelIdSchema,
  ModelSlotMutationRejected,
  PRIMARY_SLOT_ID,
  type LocalInferenceError,
  type ModelId,
  type CatalogFormModelId,
  type ModelLoadPlan,
  type SlotId,
} from "@magnitudedev/acn-protocol"
import { IcnCatalog, IcnCatalogInstallations, IcnClient, IcnInstances } from "@magnitudedev/icn"
import { projectInferenceLoadPlan } from "@magnitudedev/acn-protocol"
import { ModelSlotController } from "./model-slot-controller"
import { LocalModelRemovals } from "./local-model-removals"
import { icnCommandFailure, type IcnCommandError } from "./icn-command-failure"

export type LocalModelSyncOutcome = "Started" | "AlreadyCurrent"
export interface ModelCommandsApi {
  readonly sync: (modelId: CatalogFormModelId) => Effect.Effect<{ readonly outcome: LocalModelSyncOutcome }, LocalInferenceError>
  readonly cancelSync: (modelId: CatalogFormModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly acknowledgeSyncFailure: (modelId: CatalogFormModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly remove: (modelId: CatalogFormModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly loadSlot: (slotId: SlotId) => Effect.Effect<{}, LocalInferenceError>
  readonly previewSlotLoad: (slotId: SlotId) => Effect.Effect<ModelLoadPlan, LocalInferenceError>
  readonly stopSlot: (slotId: SlotId) => Effect.Effect<{}, LocalInferenceError>
  readonly loadModel: (modelId: ModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly stopActiveModel: Effect.Effect<{}, LocalInferenceError>
}
export class ModelCommands extends Context.Tag("ModelCommands")<ModelCommands, ModelCommandsApi>() {}

export const modelCommandFailure = (operation: string, cause: IcnCommandError) =>
  icnCommandFailure(operation, cause)

export const ModelCommandsLive: Layer.Layer<
  ModelCommands,
  never,
  IcnClient | IcnCatalog | IcnCatalogInstallations | IcnInstances | ModelSlotController | LocalModelRemovals
> = Layer.effect(ModelCommands, Effect.gen(function* () {
  const client = yield* IcnClient
  const catalog = yield* IcnCatalog
  const installations = yield* IcnCatalogInstallations
  const instances = yield* IcnInstances
  const slots = yield* ModelSlotController
  const removals = yield* LocalModelRemovals
  const selectedLocalModel = (slotId: SlotId) => slots.state.pipe(
    Effect.map((state) => state.slots[slotId === PRIMARY_SLOT_ID ? "primary" : "secondary"]),
    Effect.filterOrFail((slot) => slot._tag === "ConfiguredLocal", () => new ModelSlotMutationRejected({
      slotId, message: "The slot does not contain a configured local model",
    })),
    Effect.flatMap((slot) => Schema.decodeUnknown(ModelIdSchema)(slot.selection.providerModelId).pipe(
      Effect.mapError(() => new ModelSlotMutationRejected({
        slotId,
        message: "The slot does not contain a canonical local model identity",
      })),
    )),
  )
  const operationFor = (modelId: ModelId) => installations.get.pipe(
    Effect.map(({ state }) => state.operations.findLast((operation) => operation.modelId === modelId)),
  )
  const stopModel = (modelId: ModelId) => Effect.gen(function* () {
    const instance = (yield* instances.get).instances.findLast((candidate) => candidate.modelId === modelId
      && (candidate.lifecycle._tag === "Loading" || candidate.lifecycle._tag === "Ready"))
    if (instance !== undefined) yield* client.models.stopModelInstance({ path: { instance_id: instance.id } })
    return {}
  }).pipe(Effect.mapError((cause) => modelCommandFailure("stop", cause)))
  return ModelCommands.of({
    sync: (modelId) => client.catalog.installCatalogModel({ path: { model_id: modelId } }).pipe(
      Effect.mapError((cause) => modelCommandFailure("sync", cause)),
      Effect.tap(() => Effect.all([catalog.refresh, installations.refresh], { discard: true }).pipe(Effect.ignore)),
      Effect.map((admission) => ({ outcome: admission._tag === "Current" ? "AlreadyCurrent" as const : "Started" as const })),
    ),
    cancelSync: (modelId) => operationFor(modelId).pipe(
      Effect.mapError((cause) => modelCommandFailure("cancel_sync", cause)),
      Effect.flatMap((operation) => operation === undefined
        || (operation.state._tag !== "Pending" && operation.state._tag !== "Running")
        ? Effect.succeed({})
        : client.catalog.cancelCatalogInstallation({ path: { operation_id: operation.operationId } }).pipe(
            Effect.mapError((cause) => modelCommandFailure("cancel_sync", cause)),
            Effect.tap(() => installations.refresh.pipe(Effect.ignore)), Effect.as({}),
          )),
    ),
    acknowledgeSyncFailure: (modelId) => operationFor(modelId).pipe(
      Effect.mapError((cause) => modelCommandFailure("acknowledge_sync_failure", cause)),
      Effect.flatMap((operation) => operation?.state._tag !== "Failed" || operation.state.acknowledged
        ? Effect.succeed({})
        : client.catalog.acknowledgeCatalogInstallationFailure({ path: { operation_id: operation.operationId } }).pipe(
            Effect.mapError((cause) => modelCommandFailure("acknowledge_sync_failure", cause)),
            Effect.tap(() => installations.refresh.pipe(Effect.ignore)), Effect.as({}),
          )),
    ),
    remove: removals.remove,
    loadSlot: (slotId) => selectedLocalModel(slotId).pipe(
      Effect.flatMap((modelId) => client.models.ensureModelInstance({ payload: { modelId } })),
      Effect.tap(() => slots.refresh), Effect.as({}),
      Effect.mapError((cause) => cause instanceof ModelSlotMutationRejected ? cause : modelCommandFailure("load", cause))),
    previewSlotLoad: (slotId) => selectedLocalModel(slotId).pipe(
      Effect.flatMap((modelId) => client.models.previewModelLoad({ path: { model_id: modelId } })),
      Effect.map(projectInferenceLoadPlan),
      Effect.mapError((cause) => cause instanceof ModelSlotMutationRejected ? cause : modelCommandFailure("preview", cause))),
    stopSlot: (slotId) => selectedLocalModel(slotId).pipe(Effect.flatMap(stopModel)),
    loadModel: (modelId) => client.models.ensureModelInstance({ payload: { modelId } }).pipe(
      Effect.as({}), Effect.mapError((cause) => modelCommandFailure("load", cause))),
    stopActiveModel: instances.get.pipe(Effect.flatMap(({ instances: current }) => {
      const active = current.findLast((instance) =>
        instance.lifecycle._tag === "Loading" || instance.lifecycle._tag === "Ready")
      return active === undefined ? Effect.succeed({}) : client.models.stopModelInstance({ path: { instance_id: active.id } }).pipe(Effect.as({}))
    }), Effect.mapError((cause) => modelCommandFailure("stop", cause))),
  })
}))
