import { Context, Effect, Fiber, Layer, Option, Schema, Stream } from "effect"
import {
  CatalogFormModelIdSchema,
  CustomFormModelIdSchema,
  ModelIdSchema,
  ModelSlotMutationRejected,
  PRIMARY_SLOT_ID,
  LocalModelMutationFailed,
  type CatalogComponent,
  type LocalInferenceError,
  type LocalModelSyncOutcome,
  type ModelId,
  type CatalogFormModelId,
  type ModelLoadPlan,
  type SlotId,
} from "@magnitudedev/acn-protocol"
import { IcnCatalogInstallations, IcnClient, IcnInstances } from "@magnitudedev/icn"
import { canonicalLocalModelId, projectInferenceLoadPlan } from "@magnitudedev/acn-protocol"
import { LocalModels } from "./local-models"
import { ModelSlotController } from "./model-slot-controller"
import { LocalModelRemovals } from "./local-model-removals"
import { icnCommandFailure, type IcnCommandError } from "./icn-command-failure"
import { CustomModelCommands, externalCatalogRemovalRefusal } from "./custom-model-commands"
import { CatalogModelSync } from "./catalog-model-sync"

/** Commands by canonical model ID: a catalog ID addresses its installation, a custom ID its downloads. */
export interface ModelCommandsApi {
  readonly sync: (modelId: ModelId) => Effect.Effect<{ readonly outcome: LocalModelSyncOutcome }, LocalInferenceError>
  readonly downloadCatalogComponent: (modelId: ModelId, component: CatalogComponent) =>
    Effect.Effect<{ readonly started: boolean }, LocalInferenceError>
  readonly cancelSync: (modelId: ModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly acknowledgeSyncFailure: (modelId: ModelId) => Effect.Effect<{}, LocalInferenceError>
  readonly remove: (modelId: ModelId) => Effect.Effect<{}, LocalInferenceError>
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
  | IcnClient | IcnCatalogInstallations | IcnInstances | ModelSlotController | LocalModelRemovals
  | CustomModelCommands | LocalModels | CatalogModelSync
> = Layer.scoped(ModelCommands, Effect.gen(function* () {
  const custom = yield* CustomModelCommands
  const catalogSync = yield* CatalogModelSync
  const localModels = yield* LocalModels
  /** A catalog model's alias (its files' ID as a found model, before an upgrade) names that catalog model. */
  const canonical = (modelId: ModelId) => localModels.state.pipe(
    Effect.map(({ models, hidden }) => canonicalLocalModelId([...models, ...hidden], modelId)),
  )
  const scope = yield* Effect.scope
  const client = yield* IcnClient
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
  const admitted = (modelId: ModelId) => instances.changes.pipe(
    Stream.filter(({ instances: current }) => current.some((instance) => instance.modelId === modelId
      && (instance.lifecycle._tag === "Loading" || instance.lifecycle._tag === "Ready"))),
    Stream.runHead,
  )
  /**
   * ICN's warm demand answers only once the Instance is Ready, which can take many minutes. ACN
   * issues that demand in its own lifetime rather than the caller's and acknowledges the load once
   * ICN admits it, or with the demand's own earlier outcome, so no caller holds a request open for
   * the duration of a load.
   */
  const requestLoad = (modelId: ModelId) => Effect.gen(function* () {
    const demand = yield* client.models.ensureModelInstance({ payload: { modelId } }).pipe(Effect.forkIn(scope))
    yield* Effect.raceFirst(Fiber.join(demand), admitted(modelId))
    return {}
  })
  const stopModel = (modelId: ModelId) => Effect.gen(function* () {
    const instance = (yield* instances.get).instances.findLast((candidate) => candidate.modelId === modelId
      && (candidate.lifecycle._tag === "Loading" || candidate.lifecycle._tag === "Ready" || candidate.lifecycle._tag === "Stopping"))
    if (instance !== undefined) yield* client.models.stopModelInstance({ path: { instance_id: instance.id } })
    return {}
  }).pipe(Effect.mapError((cause) => modelCommandFailure("stop", cause)))
  const byForm = <A>(
    requested: ModelId,
    catalogCommand: (id: CatalogFormModelId) => Effect.Effect<A, LocalInferenceError>,
    customCommand: (id: typeof CustomFormModelIdSchema.Type) => Effect.Effect<A, LocalInferenceError>,
  ) => canonical(requested).pipe(Effect.flatMap((modelId) => Schema.is(CatalogFormModelIdSchema)(modelId)
    ? catalogCommand(modelId)
    : customCommand(Schema.decodeSync(CustomFormModelIdSchema)(modelId))))
  const catalogCancel = (modelId: CatalogFormModelId) => operationFor(modelId).pipe(
      Effect.mapError((cause) => modelCommandFailure("cancel_sync", cause)),
      Effect.flatMap((operation) => operation === undefined
        || (operation.state._tag !== "Pending" && operation.state._tag !== "Running"
          && operation.state._tag !== "Optimizing")
        ? Effect.succeed({})
        : client.catalog.cancelCatalogInstallation({ path: { operation_id: operation.operationId } }).pipe(
            Effect.mapError((cause) => modelCommandFailure("cancel_sync", cause)),
            Effect.tap(() => installations.refresh.pipe(Effect.ignore)), Effect.as({}),
          )),
    )
  const catalogAcknowledge = (modelId: CatalogFormModelId) => operationFor(modelId).pipe(
      Effect.mapError((cause) => modelCommandFailure("acknowledge_sync_failure", cause)),
      Effect.flatMap((operation) => operation?.state._tag !== "Failed" || operation.state.acknowledged
        ? Effect.succeed({})
        : client.catalog.acknowledgeCatalogInstallationFailure({ path: { operation_id: operation.operationId } }).pipe(
            Effect.mapError((cause) => modelCommandFailure("acknowledge_sync_failure", cause)),
            Effect.tap(() => installations.refresh.pipe(Effect.ignore)), Effect.as({}),
          )),
    )
  // Remove never deletes files Magnitude doesn't own; a catalog model found only elsewhere is hidden instead.
  const catalogRemove = (modelId: CatalogFormModelId) => custom.externalCatalogFiles(modelId).pipe(
    Effect.flatMap(Option.match({
      onSome: (files) => Effect.fail(externalCatalogRemovalRefusal(modelId, files)),
      onNone: () => stopModel(modelId).pipe(Effect.zipRight(removals.remove(modelId))),
    })),
  )
  return ModelCommands.of({
    sync: (modelId) => byForm(modelId, catalogSync.sync, custom.sync),
    downloadCatalogComponent: (modelId, component) => byForm(
      modelId,
      (id) => catalogSync.downloadComponent(id, component),
      (id) => Effect.fail(new LocalModelMutationFailed({
        code: "catalog_component_requires_catalog_model",
        message: `${id} is not a catalog model; only catalog models have components to download`,
        retryable: false,
      })),
    ),
    cancelSync: (modelId) => byForm(modelId, catalogCancel, custom.cancelSync),
    acknowledgeSyncFailure: (modelId) => byForm(modelId, catalogAcknowledge, custom.acknowledgeSyncFailure),
    remove: (modelId) => byForm(modelId, catalogRemove, (id) => stopModel(id).pipe(Effect.zipRight(custom.remove(id)))),
    loadSlot: (slotId) => selectedLocalModel(slotId).pipe(
      Effect.flatMap(requestLoad),
      Effect.tap(() => slots.refresh),
      Effect.mapError((cause) => cause instanceof ModelSlotMutationRejected ? cause : modelCommandFailure("load", cause))),
    previewSlotLoad: (slotId) => selectedLocalModel(slotId).pipe(
      Effect.flatMap((modelId) => client.models.previewModelLoad({ path: { model_id: modelId } })),
      Effect.map(projectInferenceLoadPlan),
      Effect.mapError((cause) => cause instanceof ModelSlotMutationRejected ? cause : modelCommandFailure("preview", cause))),
    stopSlot: (slotId) => selectedLocalModel(slotId).pipe(Effect.flatMap(stopModel)),
    loadModel: (modelId) => canonical(modelId).pipe(
      Effect.flatMap(requestLoad),
      Effect.mapError((cause) => modelCommandFailure("load", cause)),
    ),
    stopActiveModel: instances.get.pipe(Effect.flatMap(({ instances: current }) => {
      const active = current.findLast((instance) =>
        instance.lifecycle._tag === "Loading" || instance.lifecycle._tag === "Ready" || instance.lifecycle._tag === "Stopping")
      return active === undefined ? Effect.succeed({}) : client.models.stopModelInstance({ path: { instance_id: active.id } }).pipe(Effect.as({}))
    }), Effect.mapError((cause) => modelCommandFailure("stop", cause))),
  })
}))
