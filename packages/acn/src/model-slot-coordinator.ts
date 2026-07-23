import { Context, Effect, Layer, Match, Option, Ref, Schema, Scope, Stream } from "effect"
import {
  buildConfigStateFromSlots,
  type ConfigState,
} from "@magnitudedev/agent"
import {
  ModelSlotBlocked,
  ModelSlotLifecycle,
  ModelSlotLoadingLocalModel,
  ModelSlotReady,
  ModelSlotSchema,
  ModelSlotUnassigned,
  ModelSlotUnloadedLocalModel,
  ModelSlotUnloadingLocalModel,
  ModelSlotsMirror,
  LocalModelMutationFailed,
  ModelSlotMutationRejected,
  ModelSlotMutationFailed,
  PRIMARY_SLOT_ID,
  ProviderModelCatalogLifecycle,
  SECONDARY_SLOT_ID,
  type LocalInferenceError,
  type LocalModelId,
  type MirroredSnapshot,
  type ModelSlot,
  type ModelSlotBlockedReason,
  type ModelSlotsState,
  type ModelSlotUpdateError,
  type ProviderCatalogEntry,
  type ProviderCatalogFailure,
  type ProviderModelCatalogEntry,
  type ProviderModelCatalogState,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/protocol"
import { ReasoningEffortSchema, type ProviderId, type ProviderModelId } from "@magnitudedev/sdk"
import { IcnClient, IcnInventory } from "@magnitudedev/icn"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import { ModelConfiguration, type ModelConfigurationState } from "./model-configuration"
import { makeMirroredState, MirroredStateChanges } from "./mirrored-state"
import { LocalModelInventory } from "./local-model-inventory"
import { ProviderModelCatalog } from "./provider-model-catalog"

export interface ModelSlotCoordinatorApi {
  readonly snapshot: Effect.Effect<MirroredSnapshot<ModelSlotsState>>
  readonly changes: Stream.Stream<MirroredSnapshot<ModelSlotsState>>
  readonly agentModelConfigurations: Stream.Stream<ConfigState>
  readonly acquireLocalModel: (
    slotId: SlotId,
    providerModelId: ProviderModelId,
  ) => Effect.Effect<void, LocalInferenceError, Scope.Scope>
  readonly updateModelSlot: (
    slotId: SlotId,
    selection: Option.Option<SlotSelection>,
  ) => Effect.Effect<void, ModelSlotUpdateError>
  readonly loadModelSlot: (slotId: SlotId, selection: SlotSelection) => Effect.Effect<void, LocalInferenceError>
  readonly reloadModelSlot: (slotId: SlotId) => Effect.Effect<void, LocalInferenceError>
  readonly unloadModelSlot: (slotId: SlotId) => Effect.Effect<void, LocalInferenceError>
  readonly deleteLocalModel: (localModelId: LocalModelId) => Effect.Effect<void, LocalInferenceError>
}

export class ModelSlotCoordinator extends Context.Tag("ModelSlotCoordinator")<
  ModelSlotCoordinator,
  ModelSlotCoordinatorApi
>() {}

const sameSelection = (left: SlotSelection, right: SlotSelection): boolean =>
  left.providerId === right.providerId
  && left.providerModelId === right.providerModelId
  && left.reasoningEffort === right.reasoningEffort

const sameOptionalSelection = (
  left: Option.Option<SlotSelection>,
  right: Option.Option<SlotSelection>,
): boolean => Option.match(left, {
  onNone: () => Option.isNone(right),
  onSome: (selected) => Option.exists(right, (candidate) => sameSelection(selected, candidate)),
})

export const recoverRecentLocalSelection = (
  slotId: SlotId,
  selection: Option.Option<SlotSelection>,
  recency: readonly ProviderModelId[],
  models: readonly ProviderModelCatalogEntry[],
): Option.Option<SlotSelection> => Option.flatMap(selection, (selected) => {
  if (selected.providerId !== LOCAL_PROVIDER_ID) return Option.some(selected)
  const selectable = (model: ProviderModelCatalogEntry) => model.providerId === LOCAL_PROVIDER_ID
    && model.availability._tag === "Available"
    && model.supportedSlots.includes(slotId)
  const current = models.find((model) => model.providerModelId === selected.providerModelId && selectable(model))
  if (current) return Option.some(selected)
  const recent = recency
    .map((providerModelId) => models.find((model) => model.providerModelId === providerModelId && selectable(model)))
    .find((model): model is ProviderModelCatalogEntry => model !== undefined)
  if (!recent) return Option.some(selected)
  const reasoningEffort = recent.capabilities.reasoning.efforts.includes(selected.reasoningEffort)
    ? selected.reasoningEffort
    : Option.getOrElse(
        recent.capabilities.reasoning.defaultEffort,
        () => ReasoningEffortSchema.make("none"),
      )
  return Option.some({
    providerId: recent.providerId,
    providerModelId: recent.providerModelId,
    reasoningEffort,
  })
})

export const isModelSlotLoadSatisfied = (slot: ModelSlot): boolean =>
  slot._tag === "Ready" || slot._tag === "LoadingLocalModel"

export const isModelSlotUnloadSatisfied = (slot: ModelSlot): boolean =>
  slot._tag === "UnloadedLocalModel" || slot._tag === "UnloadingLocalModel"

const catalogContents = (state: ProviderModelCatalogState) => ProviderModelCatalogLifecycle.match(state, {
  Loading: () => ({ providers: [] as readonly ProviderCatalogEntry[], models: [] as readonly ProviderModelCatalogEntry[], failures: [] as readonly ProviderCatalogFailure[] }),
  Ready: ({ providers, models }) => ({ providers, models, failures: [] as readonly ProviderCatalogFailure[] }),
  Refreshing: ({ providers, models, failures }) => ({ providers, models, failures }),
  Degraded: ({ providers, models, failures }) => ({ providers, models, failures }),
  Unavailable: ({ providers, failures }) => ({ providers, models: [] as readonly ProviderModelCatalogEntry[], failures }),
})

const providerIssue = (
  state: ProviderModelCatalogState,
  providerId: ProviderId,
): Option.Option<string> => {
  const { providers, failures } = catalogContents(state)
  const failure = failures.find((candidate) => candidate._tag === "ProviderFailure"
    && candidate.providerId === providerId)
  if (failure) return Option.some(failure.message)
  const provider = providers.find((candidate) => candidate.providerId === providerId)
  if (!provider) return Option.some("The selected provider is unavailable")
  if (provider.authentication === "NotConfigured") return Option.some("The selected provider is not configured")
  return Match.value(provider.availability).pipe(
    Match.tag("Available", () => Option.none<string>()),
    Match.tag("Loading", ({ message }) => Option.some(Option.getOrElse(message, () => "The selected provider is loading"))),
    Match.tag("NotFound", ({ message }) => Option.some(Option.getOrElse(message, () => "The selected provider was not found"))),
    Match.tag("Failed", ({ message }) => Option.some(message)),
    Match.exhaustive,
  )
}

const selectedModelIssue = (
  state: ProviderModelCatalogState,
  slotId: SlotId,
  selection: SlotSelection,
): Option.Option<Exclude<ModelSlotBlockedReason, { readonly _tag: "LocalModelLoadFailed" }>> => {
  const unavailable = providerIssue(state, selection.providerId)
  if (Option.isSome(unavailable)) {
    return Option.some({ _tag: "ProviderUnavailable", message: unavailable.value })
  }
  const model = catalogContents(state).models.find((candidate) =>
    candidate.providerId === selection.providerId
    && candidate.providerModelId === selection.providerModelId)
  if (!model || model.availability._tag !== "Available") {
    return Option.some({ _tag: "ModelUnavailable", message: "The selected model is unavailable" })
  }
  if (!model.supportedSlots.includes(slotId)
    || model.capabilities.reasoning.supported
      && !model.capabilities.reasoning.efforts.includes(selection.reasoningEffort)) {
    return Option.some({ _tag: "InvalidConfiguration", message: "The selected slot configuration is invalid" })
  }
  return Option.none()
}

const transitionToTarget = (previous: ModelSlot, target: ModelSlot): ModelSlot => {
  switch (target._tag) {
    case "Unassigned": return ModelSlotLifecycle.transition(previous, "Unassigned", {})
    case "UnloadedLocalModel": return ModelSlotLifecycle.transition(previous, "UnloadedLocalModel", {
      selection: target.selection,
    })
    case "LoadingLocalModel": return ModelSlotLifecycle.transition(previous, "LoadingLocalModel", {
      selection: target.selection,
      percentage: target.percentage,
    })
    case "Ready": return ModelSlotLifecycle.transition(previous, "Ready", { selection: target.selection })
    case "UnloadingLocalModel": return ModelSlotLifecycle.transition(previous, "UnloadingLocalModel", {
      selection: target.selection,
    })
    case "Blocked": return ModelSlotLifecycle.transition(previous, "Blocked", {
      selection: target.selection,
      reason: target.reason,
    })
  }
}

const applyTarget = (previous: ModelSlot, target: ModelSlot): ModelSlot => {
  const selectionChanged = previous._tag !== "Unassigned"
    && target._tag !== "Unassigned"
    && !sameSelection(previous.selection, target.selection)
  if (selectionChanged) {
    const reset = ModelSlotLifecycle.transition(previous, "Unassigned", {})
    return transitionToTarget(reset, target)
  }
  if (previous._tag === target._tag) {
    switch (previous._tag) {
      case "Unassigned": return ModelSlotLifecycle.hold(previous, { slotId: target.slotId })
      case "UnloadedLocalModel": return target._tag === "UnloadedLocalModel"
        ? ModelSlotLifecycle.hold(previous, target)
        : previous
      case "LoadingLocalModel": return target._tag === "LoadingLocalModel"
        ? ModelSlotLifecycle.hold(previous, {
            ...target,
            percentage: sameSelection(previous.selection, target.selection)
              ? Math.max(previous.percentage, target.percentage)
              : target.percentage,
          })
        : previous
      case "Ready": return target._tag === "Ready" ? ModelSlotLifecycle.hold(previous, target) : previous
      case "UnloadingLocalModel": return target._tag === "UnloadingLocalModel"
        ? ModelSlotLifecycle.hold(previous, target)
        : previous
      case "Blocked": return target._tag === "Blocked" ? ModelSlotLifecycle.hold(previous, target) : previous
    }
  }
  return transitionToTarget(previous, target)
}

export const applyLocalModelLoadProgress = (
  slot: Exclude<ModelSlot, ModelSlotUnassigned>,
  fraction: number,
): ModelSlot => applyTarget(slot, new ModelSlotLoadingLocalModel({
  slotId: slot.slotId,
  selection: slot.selection,
  percentage: Math.max(0, Math.min(100, Math.round(fraction * 100))),
}))

export const applyLocalModelLoadingStage = (
  slot: ModelSlot,
  targetProviderModelId: ProviderModelId,
  fraction: number | null,
): ModelSlot => {
  if (slot._tag === "Unassigned" || slot.selection.providerId !== LOCAL_PROVIDER_ID) return slot
  if (slot.selection.providerModelId !== targetProviderModelId) {
    return applyReplacedLocalModelStage(slot, targetProviderModelId, "unloaded")
  }
  return applyLocalModelLoadProgress(slot, fraction ?? 0)
}

export const applyReplacedLocalModelStage = (
  slot: ModelSlot,
  targetProviderModelId: ProviderModelId,
  stage: "unloading" | "unloaded",
): ModelSlot => {
  if (slot._tag === "Unassigned"
    || slot.selection.providerId !== LOCAL_PROVIDER_ID
    || slot.selection.providerModelId === targetProviderModelId
    || slot._tag === "UnloadedLocalModel"
    || slot._tag === "Blocked") return slot
  return applyTarget(slot, stage === "unloading"
    ? new ModelSlotUnloadingLocalModel({ slotId: slot.slotId, selection: slot.selection })
    : new ModelSlotUnloadedLocalModel({ slotId: slot.slotId, selection: slot.selection }))
}

export const reconcileAvailableLocalSlot = (
  slotId: SlotId,
  selection: SlotSelection,
  previous: Option.Option<ModelSlot>,
): ModelSlot => {
  if (Option.isSome(previous)
    && previous.value._tag !== "Unassigned"
    && sameSelection(previous.value.selection, selection)
    && (previous.value._tag !== "Blocked"
      || previous.value.reason._tag === "LocalModelLoadFailed")) return previous.value
  return new ModelSlotUnloadedLocalModel({ slotId, selection })
}

export const ModelSlotCoordinatorLive: Layer.Layer<
  ModelSlotCoordinator,
  never,
  ModelConfiguration | LocalModelInventory | ProviderModelCatalog | IcnClient | IcnInventory | MirroredStateChanges
> = Layer.scoped(ModelSlotCoordinator, Effect.gen(function* () {
  const configuration = yield* ModelConfiguration
  const localModels = yield* LocalModelInventory
  const catalog = yield* ProviderModelCatalog
  const icn = yield* IcnClient
  const nativeInventory = yield* IcnInventory
  const scope = yield* Scope.Scope
  const reconciliationLock = yield* Effect.makeSemaphore(1)
  const modelAdmission = yield* Effect.makeSemaphore(1)

  const localSlotTarget = (
    slotId: SlotId,
    selection: SlotSelection,
    previous: Option.Option<ModelSlot>,
  ): Effect.Effect<ModelSlot> => Effect.gen(function* () {
    const nativeModelId = yield* localModels.nativeModelId(selection.providerModelId)
    const downloaded = (yield* nativeInventory.get).state.data.some((model) => model.id === nativeModelId)
    if (!downloaded) return new ModelSlotBlocked({
      slotId,
      selection,
      reason: { _tag: "ModelUnavailable", message: "The selected local model is not downloaded" },
    })
    return reconcileAvailableLocalSlot(slotId, selection, previous)
  }).pipe(Effect.catchTag("LocalModelMutationFailed", () => Effect.succeed<ModelSlot>(new ModelSlotBlocked({
    slotId,
    selection,
    reason: { _tag: "ModelUnavailable", message: "The selected local model is not downloaded" },
  }))))

  const targetFor = (
    slotId: SlotId,
    selection: Option.Option<SlotSelection>,
    catalogState: ProviderModelCatalogState,
    previous: Option.Option<ModelSlot>,
  ): Effect.Effect<ModelSlot> => Option.match(selection, {
    onNone: () => Effect.succeed<ModelSlot>(new ModelSlotUnassigned({ slotId })),
    onSome: (selected) => {
      const issue = selectedModelIssue(catalogState, slotId, selected)
      if (Option.isSome(issue)) return Effect.succeed<ModelSlot>(new ModelSlotBlocked({
        slotId,
        selection: selected,
        reason: issue.value,
      }))
      return selected.providerId === LOCAL_PROVIDER_ID
        ? localSlotTarget(slotId, selected, previous)
        : Effect.succeed<ModelSlot>(new ModelSlotReady({ slotId, selection: selected }))
    },
  })

  const recoverSelections = (
    configured: ModelConfigurationState,
    catalogState: ProviderModelCatalogState,
  ): Effect.Effect<ModelConfigurationState> => Effect.gen(function* () {
    const models = catalogContents(catalogState).models
    const primary = recoverRecentLocalSelection(
      PRIMARY_SLOT_ID,
      configured.slots.primary,
      configured.localModelRecency.primary,
      models,
    )
    const secondary = recoverRecentLocalSelection(
      SECONDARY_SLOT_ID,
      configured.slots.secondary,
      configured.localModelRecency.secondary,
      models,
    )
    if (!sameOptionalSelection(primary, configured.slots.primary)) {
      yield* configuration.updateSlot(PRIMARY_SLOT_ID, primary)
    }
    if (!sameOptionalSelection(secondary, configured.slots.secondary)) {
      yield* configuration.updateSlot(SECONDARY_SLOT_ID, secondary)
    }
    return yield* configuration.get
  }).pipe(Effect.catchAll((error) => Effect.logError("Failed to recover recent local model selection").pipe(
    Effect.annotateLogs({ error: String(error) }),
    Effect.zipRight(configuration.get),
  )))

  const initialCatalogSnapshot = yield* catalog.snapshot
  const initialNativeInventorySnapshot = yield* nativeInventory.get
  const initialCatalog = initialCatalogSnapshot.state
  const storedConfiguration = yield* configuration.get
  const persisted = yield* recoverSelections(storedConfiguration, initialCatalog)
  const initialPrimary = yield* targetFor(PRIMARY_SLOT_ID, persisted.slots.primary, initialCatalog, Option.none())
  const initialSecondary = yield* targetFor(SECONDARY_SLOT_ID, persisted.slots.secondary, initialCatalog, Option.none())
  const mirror = yield* makeMirroredState(ModelSlotsMirror, {
    slots: { primary: initialPrimary, secondary: initialSecondary },
  })

  const updateLocalSlots = (
    update: (slot: ModelSlot) => ModelSlot,
  ) => mirror.modify((state) => {
    let changed = false
    const apply = (slot: ModelSlot): ModelSlot => {
      const next = update(slot)
      changed ||= !Schema.equivalence(ModelSlotSchema)(slot, next)
      return next
    }
    return {
      state: { slots: { primary: apply(state.slots.primary), secondary: apply(state.slots.secondary) } },
      result: undefined,
      changed,
    }
  })

  const updateMatchingLocalSlots = (
    providerModelId: ProviderModelId,
    update: (slot: Exclude<ModelSlot, ModelSlotUnassigned>) => ModelSlot,
  ) => updateLocalSlots((slot) => {
    if (slot._tag === "Unassigned"
      || slot.selection.providerId !== LOCAL_PROVIDER_ID
      || slot.selection.providerModelId !== providerModelId) return slot
    return update(slot)
  })

  const updateLoadStage = (
    providerModelId: ProviderModelId,
    fraction: Option.Option<number | null>,
  ) => updateLocalSlots((slot) => applyLocalModelLoadingStage(
    slot,
    providerModelId,
    Option.getOrNull(fraction),
  ))

  const completeLoad = (providerModelId: ProviderModelId) => updateLocalSlots((slot) => {
    if (slot._tag === "Unassigned" || slot.selection.providerId !== LOCAL_PROVIDER_ID) return slot
    if (slot.selection.providerModelId !== providerModelId) {
      return applyReplacedLocalModelStage(slot, providerModelId, "unloaded")
    }
    return applyTarget(
      slot,
      new ModelSlotReady({ slotId: slot.slotId, selection: slot.selection }),
    )
  })

  const reconcileUnlocked = Effect.gen(function* () {
    const catalogState = (yield* catalog.snapshot).state
    const configured = yield* configuration.get
    const selections = (yield* recoverSelections(configured, catalogState)).slots
    const previous = (yield* mirror.get).state
    const primaryTarget = yield* targetFor(
      PRIMARY_SLOT_ID,
      selections.primary,
      catalogState,
      Option.some(previous.slots.primary),
    )
    const secondaryTarget = yield* targetFor(
      SECONDARY_SLOT_ID,
      selections.secondary,
      catalogState,
      Option.some(previous.slots.secondary),
    )
    yield* mirror.setIfChanged({
      slots: {
        primary: applyTarget(previous.slots.primary, primaryTarget),
        secondary: applyTarget(previous.slots.secondary, secondaryTarget),
      },
    }, Schema.equivalence(ModelSlotsMirror.stateSchema))
  })

  const reconcile = reconciliationLock.withPermits(1)(reconcileUnlocked)
  yield* Effect.forkIn(configuration.changes.pipe(Stream.runForEach(() => reconcile)), scope)
  yield* Effect.forkIn(nativeInventory.changes.pipe(
    Stream.dropWhile((snapshot) => snapshot.revision <= initialNativeInventorySnapshot.revision),
    Stream.runForEach(() => reconcile),
  ), scope)
  yield* Effect.forkIn(catalog.changes.pipe(
    Stream.dropWhile((snapshot) => snapshot.revision <= initialCatalogSnapshot.revision),
    Stream.runForEach(() => reconcile),
  ), scope)

  const reject = (slotId: SlotId, message: string) => new ModelSlotMutationRejected({ slotId, message })
  const slotFailure = (slotId: SlotId, code: string, error: unknown) => new ModelSlotMutationFailed({
    slotId,
    code,
    message: typeof error === "object" && error !== null && "message" in error
      ? String(error.message)
      : String(error),
    retryable: typeof error !== "object" || error === null || !("retryable" in error)
      || error.retryable !== false,
  })
  const localFailure = (code: string, error: unknown) => new LocalModelMutationFailed({
    code,
    message: typeof error === "object" && error !== null && "message" in error
      ? String(error.message)
      : String(error),
    retryable: typeof error !== "object" || error === null || !("retryable" in error)
      || error.retryable !== false,
  })

  const selectedSlot = (slotId: SlotId) => mirror.get.pipe(
    Effect.map(({ state }) => slotId === PRIMARY_SLOT_ID
      ? state.slots.primary
      : state.slots.secondary),
  )

  const validateSelection = (
    slotId: SlotId,
    selection: SlotSelection,
  ): Effect.Effect<void, ModelSlotMutationRejected> => Effect.gen(function* () {
    const state = (yield* catalog.snapshot).state
    const issue = selectedModelIssue(state, slotId, selection)
    if (Option.isSome(issue)) return yield* reject(slotId, issue.value.message)
  })

  const blockLoad = (
    providerModelId: ProviderModelId,
    error: { readonly code: string; readonly message: string; readonly retryable: boolean },
  ) =>
    updateMatchingLocalSlots(providerModelId, (slot) => {
      const reason: ModelSlotBlockedReason = {
        _tag: "LocalModelLoadFailed",
        error,
      }
      return slot._tag === "Blocked"
        ? ModelSlotLifecycle.hold(slot, { reason })
        : ModelSlotLifecycle.transition(slot, "Blocked", { reason })
    }).pipe(Effect.asVoid)

  const unloadProviderModel = (
    providerModelId: ProviderModelId,
  ): Effect.Effect<void, LocalInferenceError> => Effect.gen(function* () {
    const nativeModelId = yield* localModels.nativeModelId(providerModelId)
    yield* updateMatchingLocalSlots(providerModelId, (slot) => applyTarget(
      slot,
      new ModelSlotUnloadingLocalModel({ slotId: slot.slotId, selection: slot.selection }),
    ))
    yield* icn.models.unloadModel({ path: { model_id: nativeModelId } }).pipe(
      Effect.mapError((error) => localFailure("local_model_unload_failed", error)),
      Effect.tapError(() => updateMatchingLocalSlots(providerModelId, (slot) => applyTarget(
        slot,
        new ModelSlotReady({ slotId: slot.slotId, selection: slot.selection }),
      ))),
    )
    yield* updateMatchingLocalSlots(providerModelId, (slot) => applyTarget(
      slot,
      new ModelSlotUnloadedLocalModel({ slotId: slot.slotId, selection: slot.selection }),
    ))
  })

  const updateModelSlot: ModelSlotCoordinatorApi["updateModelSlot"] = (slotId, selection) =>
    Effect.gen(function* () {
      const previous = yield* selectedSlot(slotId)
      if (Option.isNone(selection) && previous._tag === "Unassigned") return
      if (Option.isSome(selection) && previous._tag !== "Unassigned"
        && sameSelection(previous.selection, selection.value)) return
      if (Option.isSome(selection)) yield* validateSelection(slotId, selection.value)
      yield* configuration.updateSlot(slotId, selection).pipe(
        Effect.mapError((error) => slotFailure(slotId, "model_slot_persistence_failed", error)),
      )
      yield* reconcile
      if (previous._tag === "Unassigned"
        || previous.selection.providerId !== LOCAL_PROVIDER_ID
        || previous._tag === "UnloadedLocalModel"
        || previous._tag === "Blocked") return
      const configured = (yield* configuration.get).slots
      const stillSelected = [configured.primary, configured.secondary].some((configuredSelection) =>
        Option.exists(configuredSelection, (value) => value.providerId === LOCAL_PROVIDER_ID
          && value.providerModelId === previous.selection.providerModelId))
      if (stillSelected) return
      yield* modelAdmission.withPermits(1)(
        unloadProviderModel(previous.selection.providerModelId).pipe(
          Effect.mapError((error) => slotFailure(slotId, "model_slot_followup_unload_failed", error)),
        ),
      )
    })

  const loadSelectedSlotUnlocked = (slotId: SlotId): Effect.Effect<void, LocalInferenceError> =>
    Effect.gen(function* () {
      const slot = yield* selectedSlot(slotId)
      if (slot._tag === "Unassigned" || slot.selection.providerId !== LOCAL_PROVIDER_ID) {
        return yield* reject(slotId, "The slot does not contain a local model")
      }
      if (isModelSlotLoadSatisfied(slot)) return
      if (slot._tag === "Blocked" && slot.reason._tag !== "LocalModelLoadFailed") {
        return yield* reject(slotId, "The selected local model is not loadable")
      }
      if (slot._tag !== "UnloadedLocalModel" && slot._tag !== "Blocked") {
        return yield* reject(slotId, "The selected local model is not loadable")
      }
      const providerModelId = slot.selection.providerModelId
      const nativeModelId = yield* localModels.nativeModelId(providerModelId)
      const catalogModel = catalogContents((yield* catalog.snapshot).state).models.find((model) =>
        model.providerId === LOCAL_PROVIDER_ID && model.providerModelId === providerModelId)
      if (!catalogModel) return yield* reject(slotId, "The selected local model is unavailable")
      yield* icn.models.configureModelServing({
        path: { model_id: nativeModelId },
        payload: { context_length: catalogModel.contextWindow, parallel_sequences: 1 },
      }).pipe(Effect.mapError((error) => localFailure("local_model_configuration_failed", error)))
      const response = yield* icn.models.loadModel({ path: { model_id: nativeModelId } }).pipe(
        Effect.mapError((error) => localFailure("local_model_load_failed", error)),
        Effect.tapError((error) => blockLoad(providerModelId, error)),
      )
      const readyObserved = yield* Ref.make(false)
      const replacementStarted = yield* Ref.make(false)
      yield* response.events.pipe(
        Stream.runForEach((event) => {
          if (event.type === "progress" && event.stage === "unloading") {
            return updateLocalSlots((current) => applyReplacedLocalModelStage(
              current,
              providerModelId,
              "unloading",
            )).pipe(Effect.zipRight(Ref.set(replacementStarted, true)))
          }
          if (event.type === "progress" && event.stage === "loading") {
            return updateLoadStage(providerModelId, event.fraction).pipe(Effect.asVoid)
          }
          if (event.type === "ready") {
            return completeLoad(providerModelId).pipe(Effect.zipRight(Ref.set(readyObserved, true)))
          }
          if (event.type === "failed") {
            const error = new LocalModelMutationFailed({
              code: event.code,
              message: event.message,
              retryable: event.retryable,
            })
            return blockLoad(providerModelId, error).pipe(Effect.zipRight(Effect.fail(error)))
          }
          return Effect.void
        }),
        Effect.mapError((error) => error._tag === "LocalModelMutationFailed"
          ? error
          : localFailure("local_model_load_stream_failed", error)),
        Effect.tapError(() => Ref.get(replacementStarted).pipe(
          Effect.flatMap((started) => started
            ? updateLocalSlots((current) => applyReplacedLocalModelStage(
                current,
                providerModelId,
                "unloaded",
              ))
            : Effect.void),
        )),
        Effect.tapError((error) => blockLoad(providerModelId, error)),
      )
      if (!(yield* Ref.get(readyObserved))) {
        const error = localFailure("local_model_load_stream_incomplete", "The model load ended before ICN reported readiness")
        yield* blockLoad(providerModelId, error)
        return yield* error
      }
    })

  const loadModelSlot: ModelSlotCoordinatorApi["loadModelSlot"] = (slotId, selection) =>
    updateModelSlot(slotId, Option.some(selection)).pipe(
      Effect.flatMap(() => modelAdmission.withPermits(1)(loadSelectedSlotUnlocked(slotId))),
    )

  const unloadModelSlotUnlocked = (slotId: SlotId): Effect.Effect<void, LocalInferenceError> =>
    Effect.gen(function* () {
      const slot = yield* selectedSlot(slotId)
      if (slot._tag === "Unassigned" || slot.selection.providerId !== LOCAL_PROVIDER_ID) {
        return yield* reject(slotId, "The slot does not contain a local model")
      }
      if (isModelSlotUnloadSatisfied(slot)) return
      if (slot._tag !== "Ready") return yield* reject(slotId, "The selected local model is not loaded")
      yield* unloadProviderModel(slot.selection.providerModelId)
    })

  const unloadModelSlot: ModelSlotCoordinatorApi["unloadModelSlot"] = (slotId) =>
    modelAdmission.withPermits(1)(unloadModelSlotUnlocked(slotId))

  const reloadModelSlot: ModelSlotCoordinatorApi["reloadModelSlot"] = (slotId) =>
    modelAdmission.withPermits(1)(Effect.gen(function* () {
      const slot = yield* selectedSlot(slotId)
      if (slot._tag === "Unassigned" || slot.selection.providerId !== LOCAL_PROVIDER_ID) {
        return yield* reject(slotId, "The slot does not contain a local model")
      }
      yield* unloadModelSlotUnlocked(slotId)
      yield* loadSelectedSlotUnlocked(slotId)
    }))

  const acquireLocalModel: ModelSlotCoordinatorApi["acquireLocalModel"] = (
    slotId,
    providerModelId,
  ) => Effect.acquireRelease(
    modelAdmission.take(1).pipe(
      Effect.zipRight(Effect.gen(function* () {
        const slot = yield* selectedSlot(slotId)
        if (slot._tag === "Unassigned"
          || slot.selection.providerId !== LOCAL_PROVIDER_ID
          || slot.selection.providerModelId !== providerModelId) {
          return yield* reject(slotId, "The local model request no longer matches the selected slot")
        }
        yield* loadSelectedSlotUnlocked(slotId)
      })),
      Effect.onError(() => modelAdmission.release(1)),
    ),
    () => modelAdmission.release(1),
  ).pipe(Effect.asVoid)

  const deleteLocalModel: ModelSlotCoordinatorApi["deleteLocalModel"] = (localModelId) =>
    modelAdmission.withPermits(1)(Effect.gen(function* () {
      const providerModelId = localModels.providerModelId(localModelId)
      const slots = (yield* mirror.get).state.slots
      const affected = [slots.primary, slots.secondary].filter((slot) => slot._tag !== "Unassigned"
        && slot.selection.providerId === LOCAL_PROVIDER_ID
        && slot.selection.providerModelId === providerModelId)
      const busy = affected.find((slot) => slot._tag === "LoadingLocalModel" || slot._tag === "UnloadingLocalModel")
      if (busy) return yield* reject(busy.slotId, "The local model cannot be deleted while loading or unloading")
      const ready = affected.find((slot) => slot._tag === "Ready")
      if (ready?._tag === "Ready") yield* unloadProviderModel(providerModelId)
      yield* localModels.delete(localModelId)
    }))

  const sameAgentConfiguration = (left: ConfigState, right: ConfigState): boolean =>
    (["primary", "secondary"] as const).every((slotId) => {
      const a = left.bySlot[slotId]
      const b = right.bySlot[slotId]
      if (a._tag !== b._tag) return false
      if (a._tag === "Unavailable" && b._tag === "Unavailable") return a.reason === b.reason
      if (a._tag !== "Ready" || b._tag !== "Ready") return false
      return a.config.providerId === b.config.providerId
        && a.config.providerModelId === b.config.providerModelId
        && a.config.reasoningEffort === b.config.reasoningEffort
        && a.config.profile.contextWindow === b.config.profile.contextWindow
        && a.config.profile.maxOutputTokens === b.config.profile.maxOutputTokens
        && a.config.vision === b.config.vision
        && a.config.hardCap === b.config.hardCap
        && a.config.softCap === b.config.softCap
    })

  const agentModelConfigurations = Stream.zipLatestAll(
    mirror.changes,
    catalog.changes,
    configuration.changes,
  ).pipe(
    Stream.map(([slots, catalogSnapshot, configured]) => buildConfigStateFromSlots(
      catalogContents(catalogSnapshot.state).models,
      slots.state.slots,
      configured.contextLimits,
    )),
    Stream.changesWith(sameAgentConfiguration),
    Stream.mapAccum(0, (revision, state) => [
      revision + 1,
      { ...state, revision: revision + 1 },
    ] as const),
  )

  return ModelSlotCoordinator.of({
    snapshot: mirror.get,
    changes: mirror.changes,
    agentModelConfigurations,
    acquireLocalModel,
    updateModelSlot,
    loadModelSlot,
    reloadModelSlot,
    unloadModelSlot,
    deleteLocalModel,
  })
}))
