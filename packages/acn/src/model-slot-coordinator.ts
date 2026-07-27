import {
  Cause,
  Context,
  Effect,
  Exit,
  Layer,
  Match,
  Option,
  Ref,
  Schema,
  Scope,
  Stream,
  SubscriptionRef,
} from "effect"
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
  ModelPreferenceMutationFailed,
  ModelSlotMutationRejected,
  ModelSlotMutationFailed,
  PRIMARY_SLOT_ID,
  ProviderModelCatalogLifecycle,
  SECONDARY_SLOT_ID,
  type LocalInferenceError,
  type MirroredSnapshot,
  type ModelSlot,
  type ModelSlotBlockedReason,
  type ModelSlotsState,
  type ModelSlotUpdateError,
  type ProviderCatalogEntry,
  type ProviderCatalogFailure,
  type ProviderModelCatalogEntry,
  type ProviderModelCatalogState,
  type ProviderModelIdentity,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/protocol"
import { ReasoningEffortSchema, type ProviderId, type ProviderModelId } from "@magnitudedev/sdk"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import { ModelConfiguration, type ModelConfigurationState } from "./model-configuration"
import { makeMirroredState, MirroredStateChanges } from "./mirrored-state"
import { LocalModelPackages } from "./local-model-packages"
import { LocalModelRuntime } from "./local-model-runtime"
import { LocalProviderOfferings } from "./local-provider-offerings"
import { modelOfferingTargetPackageIds } from "@magnitudedev/protocol"
import { ProviderModelCatalog } from "./provider-model-catalog"
import { AcnActivityTracker } from "./activity-tracker"
import {
  makeServiceOperationCoordinator,
  type ServiceOperationAdmission,
  type ServiceOperationRequest,
} from "./service-operation-coordinator"

export interface ModelSlotCoordinatorApi {
  readonly snapshot: Effect.Effect<MirroredSnapshot<ModelSlotsState>>
  readonly changes: Stream.Stream<MirroredSnapshot<ModelSlotsState>>
  readonly agentModelConfiguration: Effect.Effect<ConfigState>
  readonly agentModelConfigurations: Stream.Stream<ConfigState>
  readonly acquireLocalModel: (
    slotId: SlotId,
    providerModelId: ProviderModelId,
  ) => Effect.Effect<void, LocalInferenceError, Scope.Scope>
  readonly updateModelSlot: (
    slotId: SlotId,
    selection: Option.Option<SlotSelection>,
  ) => Effect.Effect<void, ModelSlotUpdateError>
  readonly setModelFavorite: (
    model: ProviderModelIdentity,
    favorite: boolean,
  ) => Effect.Effect<void, ModelPreferenceMutationFailed>
  readonly loadModel: (slotId: SlotId) => Effect.Effect<void, LocalInferenceError>
  readonly unloadModel: (slotId: SlotId) => Effect.Effect<void, LocalInferenceError>
  readonly unloadModelAndWait: (slotId: SlotId) => Effect.Effect<void, LocalInferenceError>
}

export class ModelSlotCoordinator extends Context.Tag("ModelSlotCoordinator")<
  ModelSlotCoordinator,
  ModelSlotCoordinatorApi
>() {}

type RuntimeTransitionKey =
  | {
      readonly _tag: "Load"
      readonly providerModelId: ProviderModelId
    }
  | {
      readonly _tag: "Unload"
      readonly providerModelId: ProviderModelId
    }

const sameTransition = (left: RuntimeTransitionKey, right: RuntimeTransitionKey): boolean =>
  left._tag === right._tag && left.providerModelId === right.providerModelId

const sameSelection = (left: SlotSelection, right: SlotSelection): boolean =>
  left.providerId === right.providerId
  && left.providerModelId === right.providerModelId
  && left.reasoningEffort === right.reasoningEffort

const sameModel = (left: SlotSelection, right: SlotSelection): boolean =>
  left.providerId === right.providerId
  && left.providerModelId === right.providerModelId

const sameOptionalSelection = (
  left: Option.Option<SlotSelection>,
  right: Option.Option<SlotSelection>,
): boolean => Option.match(left, {
  onNone: () => Option.isNone(right),
  onSome: (selected) => Option.exists(right, (candidate) => sameSelection(selected, candidate)),
})

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

const normalizeSelectionReasoning = (
  selection: SlotSelection,
  model: Pick<ProviderModelCatalogEntry, "capabilities">,
): SlotSelection => ({
  ...selection,
  reasoningEffort: model.capabilities.reasoning.efforts.includes(selection.reasoningEffort)
    ? selection.reasoningEffort
    : Option.getOrElse(
        model.capabilities.reasoning.defaultEffort,
        () => ReasoningEffortSchema.make("none"),
      ),
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
  if (current) return Option.some(normalizeSelectionReasoning(selected, current))
  const replacement = recency
    .map((providerModelId) => models.find((model) => model.providerModelId === providerModelId && selectable(model)))
    .find((model): model is ProviderModelCatalogEntry => model !== undefined)
  if (!replacement) return Option.some(selected)
  return Option.some(normalizeSelectionReasoning({
    ...selected,
    providerId: replacement.providerId,
    providerModelId: replacement.providerModelId,
  }, replacement))
})

export const isModelSlotLoadSatisfied = (slot: ModelSlot): boolean =>
  slot._tag === "Ready"

export const isModelSlotUnloadSatisfied = (slot: ModelSlot): boolean =>
  slot._tag === "UnloadedLocalModel"

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
): Option.Option<Exclude<
  ModelSlotBlockedReason,
  {
    readonly _tag:
      | "LocalModelLoadFailed"
      | "LocalModelRuntimeLost"
      | "LocalModelStoppedLowMemory"
  }
>> => {
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
  const reasoningIsValid = model.capabilities.reasoning.supported
    ? model.capabilities.reasoning.efforts.includes(selection.reasoningEffort)
    : selection.reasoningEffort === "none"
  if (!model.supportedSlots.includes(slotId) || !reasoningIsValid) {
    return Option.some({ _tag: "InvalidConfiguration", message: "The selected slot configuration is invalid" })
  }
  return Option.none()
}

const initializeSlot = (previous: ModelSlotUnassigned, target: ModelSlot): ModelSlot => {
  switch (target._tag) {
    case "Unassigned": return ModelSlotLifecycle.hold(previous, { slotId: target.slotId })
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

export const reconcileSlotState = (previous: ModelSlot, target: ModelSlot): ModelSlot => {
  const modelChanged = previous._tag !== "Unassigned"
    && target._tag !== "Unassigned"
    && !sameModel(previous.selection, target.selection)
  if (modelChanged) {
    const reset = ModelSlotLifecycle.transition(previous, "Unassigned", {})
    return initializeSlot(reset, target)
  }
  if (previous._tag === "Unassigned") return initializeSlot(previous, target)
  switch (target._tag) {
    case "Unassigned":
      return ModelSlotLifecycle.transition(previous, "Unassigned", {})
    case "UnloadedLocalModel":
      return previous._tag === "UnloadedLocalModel"
        ? ModelSlotLifecycle.hold(previous, target)
        : ModelSlotLifecycle.transition(previous, "UnloadedLocalModel", {
            selection: target.selection,
          })
    case "LoadingLocalModel":
      switch (previous._tag) {
        case "LoadingLocalModel":
          return ModelSlotLifecycle.hold(previous, {
            ...target,
            percentage: Math.max(previous.percentage, target.percentage),
          })
        case "UnloadedLocalModel":
        case "UnloadingLocalModel":
        case "Blocked":
          return ModelSlotLifecycle.transition(previous, "LoadingLocalModel", {
            selection: target.selection,
            percentage: target.percentage,
          })
        case "Ready":
          return previous
      }
    case "Ready":
      return previous._tag === "Ready"
        ? ModelSlotLifecycle.hold(previous, target)
        : ModelSlotLifecycle.transition(previous, "Ready", { selection: target.selection })
    case "UnloadingLocalModel":
      switch (previous._tag) {
        case "UnloadingLocalModel":
          return ModelSlotLifecycle.hold(previous, target)
        case "LoadingLocalModel":
        case "Ready":
          return ModelSlotLifecycle.transition(previous, "UnloadingLocalModel", {
            selection: target.selection,
          })
        case "UnloadedLocalModel":
        case "Blocked":
          return previous
      }
    case "Blocked":
      return previous._tag === "Blocked"
        ? ModelSlotLifecycle.hold(previous, target)
        : ModelSlotLifecycle.transition(previous, "Blocked", {
            selection: target.selection,
            reason: target.reason,
          })
  }
}

export const applyLocalModelLoadProgress = (
  slot: Exclude<ModelSlot, ModelSlotUnassigned>,
  fraction: number,
): ModelSlot => {
  // Only the terminal Ready event owns 100%. Estimated loading progress is capped at 99.
  const percentage = Math.max(0, Math.min(99, Math.round(fraction * 100)))
  switch (slot._tag) {
    case "LoadingLocalModel":
      return ModelSlotLifecycle.hold(slot, {
        percentage: Math.max(slot.percentage, percentage),
      })
    case "UnloadedLocalModel":
    case "UnloadingLocalModel":
    case "Blocked":
      return ModelSlotLifecycle.transition(slot, "LoadingLocalModel", { percentage })
    case "Ready":
      return slot
  }
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
  if (stage === "unloading") {
    return slot._tag === "UnloadingLocalModel"
      ? ModelSlotLifecycle.hold(slot, {})
      : ModelSlotLifecycle.transition(slot, "UnloadingLocalModel", {})
  }
  return ModelSlotLifecycle.transition(slot, "UnloadedLocalModel", {})
}

export const reconcileAvailableLocalSlot = (
  slotId: SlotId,
  selection: SlotSelection,
  previous: Option.Option<ModelSlot>,
): ModelSlot => {
  if (Option.isSome(previous)
    && previous.value._tag !== "Unassigned"
    && sameModel(previous.value.selection, selection)
    && (previous.value._tag !== "Blocked"
      || previous.value.reason._tag === "LocalModelLoadFailed"
      || previous.value.reason._tag === "LocalModelRuntimeLost"
      || previous.value.reason._tag === "LocalModelStoppedLowMemory")) {
    return ModelSlotLifecycle.hold(previous.value, { selection })
  }
  return new ModelSlotUnloadedLocalModel({ slotId, selection })
}

export const ModelSlotCoordinatorLive: Layer.Layer<
  ModelSlotCoordinator,
  never,
  ModelConfiguration | LocalModelPackages | LocalModelRuntime
    | LocalProviderOfferings | ProviderModelCatalog | MirroredStateChanges | AcnActivityTracker
> = Layer.scoped(ModelSlotCoordinator, Effect.gen(function* () {
  const configuration = yield* ModelConfiguration
  const localPackages = yield* LocalModelPackages
  const localRuntime = yield* LocalModelRuntime
  const localOfferings = yield* LocalProviderOfferings
  const catalog = yield* ProviderModelCatalog
  const scope = yield* Scope.Scope
  const reconciliationLock = yield* Effect.makeSemaphore(1)
  const modelAdmission = yield* Effect.makeSemaphore(1)
  const transitions = yield* makeServiceOperationCoordinator<
    RuntimeTransitionKey,
    LocalInferenceError
  >(sameTransition)

  const localSlotTarget = (
    slotId: SlotId,
    selection: SlotSelection,
    previous: Option.Option<ModelSlot>,
  ): Effect.Effect<ModelSlot> => Effect.gen(function* () {
    const offering = yield* localOfferings.resolve(selection.providerModelId)
    const installed = yield* localPackages.installedPackageIds
    const downloaded = modelOfferingTargetPackageIds(offering.configuration.target)
      .every((packageId) => installed.has(packageId))
    if (!downloaded) return new ModelSlotBlocked({
      slotId,
      selection,
      reason: { _tag: "ModelUnavailable", message: "The selected local model is not downloaded" },
    })
    return reconcileAvailableLocalSlot(slotId, selection, previous)
  }).pipe(Effect.catchAll(() => Effect.succeed<ModelSlot>(new ModelSlotBlocked({
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
    const recoverSlot = (
      slotId: SlotId,
      selection: Option.Option<SlotSelection>,
      recency: readonly ProviderModelId[],
    ) => Effect.gen(function* () {
      const selected = Option.getOrNull(selection)
      if (selected === null || selected.providerId !== LOCAL_PROVIDER_ID) return selection
      const current = models.find((model) => model.providerModelId === selected.providerModelId)
      const durableOffering = yield* Effect.option(localOfferings.resolve(selected.providerModelId))
      // A durable offering is authoritative user intent even before its packages exist locally.
      // Catalog availability must not replace an active download or its surfaced failure with an
      // older installed model.
      if (Option.isSome(durableOffering)) {
        if (current && !current.supportedSlots.includes(slotId)) {
          return recoverRecentLocalSelection(slotId, selection, recency, models)
        }
        return Option.some(normalizeSelectionReasoning(
          selected,
          current ?? durableOffering.value,
        ))
      }
      return recoverRecentLocalSelection(slotId, selection, recency, models)
    })
    const primary = yield* recoverSlot(
      PRIMARY_SLOT_ID,
      configured.slots.primary,
      configured.localModelRecency.primary,
    )
    const secondary = yield* recoverSlot(
      SECONDARY_SLOT_ID,
      configured.slots.secondary,
      configured.localModelRecency.secondary,
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
  const initialCatalog = initialCatalogSnapshot.state
  const storedConfiguration = yield* configuration.get
  const persisted = yield* recoverSelections(storedConfiguration, initialCatalog)
  const initialPrimary = yield* targetFor(PRIMARY_SLOT_ID, persisted.slots.primary, initialCatalog, Option.none())
  const initialSecondary = yield* targetFor(SECONDARY_SLOT_ID, persisted.slots.secondary, initialCatalog, Option.none())
  const mirror = yield* makeMirroredState(ModelSlotsMirror, {
    slots: { primary: initialPrimary, secondary: initialSecondary },
    recentModelIds: persisted.localModelRecency,
    favoriteModels: persisted.favoriteModels,
  })
  const initialAgentConfiguration = buildConfigStateFromSlots(
    catalogContents(initialCatalog).models,
    (yield* mirror.get).state.slots,
    persisted.contextLimits,
    1,
  )
  const agentConfiguration = yield* SubscriptionRef.make(initialAgentConfiguration)
  const agentConfigurationLock = yield* Effect.makeSemaphore(1)

  const reconcileAgentConfiguration = agentConfigurationLock.withPermits(1)(
    Effect.gen(function* () {
      const current = yield* SubscriptionRef.get(agentConfiguration)
      const slots = (yield* mirror.get).state.slots
      const catalogState = (yield* catalog.snapshot).state
      const configured = yield* configuration.get
      const next = buildConfigStateFromSlots(
        catalogContents(catalogState).models,
        slots,
        configured.contextLimits,
        current.revision + 1,
      )
      if (!sameAgentConfiguration(current, next)) {
        yield* SubscriptionRef.set(agentConfiguration, next)
      }
    }),
  )

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
      state: {
        ...state,
        slots: { primary: apply(state.slots.primary), secondary: apply(state.slots.secondary) },
      },
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

  const completeLoad = (providerModelId: ProviderModelId) => updateLocalSlots((slot) => {
    if (slot._tag === "Unassigned" || slot.selection.providerId !== LOCAL_PROVIDER_ID) return slot
    if (slot.selection.providerModelId !== providerModelId) {
      return applyReplacedLocalModelStage(slot, providerModelId, "unloaded")
    }
    switch (slot._tag) {
      case "Ready":
        return ModelSlotLifecycle.hold(slot, {})
      case "UnloadedLocalModel":
      case "LoadingLocalModel":
      case "UnloadingLocalModel":
      case "Blocked":
        return ModelSlotLifecycle.transition(slot, "Ready", {})
    }
  })

  const updateLoadProgress = (
    providerModelId: ProviderModelId,
    progress: {
      readonly stage: "queued" | "resolving" | "unloading" | "loading" | "verifying"
      readonly fraction: Option.Option<number | null>
    },
  ) => updateLocalSlots((slot) => {
    if (slot._tag === "Unassigned" || slot.selection.providerId !== LOCAL_PROVIDER_ID) return slot
    if (slot.selection.providerModelId !== providerModelId) {
      if (progress.stage === "queued" || progress.stage === "resolving") return slot
      return applyReplacedLocalModelStage(
        slot,
        providerModelId,
        progress.stage === "unloading" ? "unloading" : "unloaded",
      )
    }
    if (progress.stage === "queued"
      || progress.stage === "resolving"
      || progress.stage === "unloading") return slot
    const fraction = Option.getOrNull(progress.fraction)
    return fraction === null ? slot : applyLocalModelLoadProgress(slot, fraction)
  })

  const reconcileUnlocked = Effect.gen(function* () {
    const catalogState = (yield* catalog.snapshot).state
    const configured = yield* configuration.get
    const recovered = yield* recoverSelections(configured, catalogState)
    const selections = recovered.slots
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
        primary: reconcileSlotState(previous.slots.primary, primaryTarget),
        secondary: reconcileSlotState(previous.slots.secondary, secondaryTarget),
      },
      recentModelIds: recovered.localModelRecency,
      favoriteModels: recovered.favoriteModels,
    }, Schema.equivalence(ModelSlotsMirror.stateSchema))
  })

  const reconcile = reconciliationLock.withPermits(1)(reconcileUnlocked)
  const reconcileAll = reconcile.pipe(Effect.zipRight(reconcileAgentConfiguration))
  yield* Effect.forkIn(configuration.changes.pipe(Stream.runForEach(() => reconcileAll)), scope)
  yield* Effect.forkIn(localPackages.changes.pipe(
    Stream.runForEach(() => reconcileAll),
  ), scope)
  yield* Effect.forkIn(catalog.changes.pipe(
    Stream.dropWhile((snapshot) => snapshot.revision <= initialCatalogSnapshot.revision),
    Stream.runForEach(() => reconcileAll),
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
  const selectedSlot = (slotId: SlotId) => mirror.get.pipe(
    Effect.map(({ state }) => slotId === PRIMARY_SLOT_ID
      ? state.slots.primary
      : state.slots.secondary),
  )

  const localModelFailureReason = (
    fallback: "LocalModelLoadFailed" | "LocalModelRuntimeLost",
    error: { readonly code: string; readonly message: string; readonly retryable: boolean },
  ): ModelSlotBlockedReason => error.code === "low_memory"
    ? { _tag: "LocalModelStoppedLowMemory", error }
    : { _tag: fallback, error }

  const normalizeAndValidateSelection = (
    slotId: SlotId,
    selection: SlotSelection,
  ): Effect.Effect<SlotSelection, ModelSlotMutationRejected> => Effect.gen(function* () {
    if (selection.providerId === LOCAL_PROVIDER_ID) {
      const offering = yield* Effect.option(localOfferings.resolve(selection.providerModelId))
      if (Option.isSome(offering)) {
        const installed = yield* localPackages.installedPackageIds
        const awaitingInstallation = modelOfferingTargetPackageIds(offering.value.configuration.target)
          .some((packageId) => !installed.has(packageId))
        if (awaitingInstallation) {
          return normalizeSelectionReasoning(selection, offering.value)
        }
      }
    }
    const state = (yield* catalog.snapshot).state
    const model = catalogContents(state).models.find((candidate) =>
      candidate.providerId === selection.providerId
      && candidate.providerModelId === selection.providerModelId)
    const normalized = model ? normalizeSelectionReasoning(selection, model) : selection
    const issue = selectedModelIssue(state, slotId, normalized)
    if (Option.isSome(issue)) {
      return yield* reject(slotId, issue.value.message)
    }
    return normalized
  })

  const blockLoad = (
    providerModelId: ProviderModelId,
    error: { readonly code: string; readonly message: string; readonly retryable: boolean },
  ) =>
    updateLocalSlots((slot) => {
      if (slot._tag === "Unassigned" || slot.selection.providerId !== LOCAL_PROVIDER_ID) return slot
      if (slot.selection.providerModelId !== providerModelId) {
        return applyReplacedLocalModelStage(slot, providerModelId, "unloaded")
      }
      const reason = localModelFailureReason("LocalModelLoadFailed", error)
      return slot._tag === "Blocked"
        ? ModelSlotLifecycle.hold(slot, { reason })
        : ModelSlotLifecycle.transition(slot, "Blocked", { reason })
    }).pipe(Effect.asVoid)

  const blockRuntimeLoss = (
    providerModelId: ProviderModelId,
    error: { readonly code: string; readonly message: string; readonly retryable: boolean },
  ) =>
    updateMatchingLocalSlots(providerModelId, (slot) => {
      const reason = localModelFailureReason("LocalModelRuntimeLost", error)
      return slot._tag === "Blocked"
        ? ModelSlotLifecycle.hold(slot, { reason })
        : ModelSlotLifecycle.transition(slot, "Blocked", { reason })
    }).pipe(Effect.asVoid)

  yield* Effect.forkIn(localRuntime.changes.pipe(
    Stream.runForEach(({ providerModelId, error }) => blockRuntimeLoss(providerModelId, {
      code: error.code,
      message: error.message,
      retryable: error.retryable,
    })),
  ), scope)

  const transitionFailure = (
    operation: RuntimeTransitionKey,
    cause: Cause.Cause<LocalInferenceError>,
  ): LocalInferenceError => Option.getOrElse(Cause.failureOption(cause), () =>
    new LocalModelMutationFailed({
      code: Cause.isInterruptedOnly(cause)
        ? "local_model_transition_interrupted"
        : "local_model_transition_defect",
      message: Cause.isInterruptedOnly(cause)
        ? `The ${operation._tag.toLowerCase()} operation was interrupted`
        : Cause.pretty(cause).slice(0, 1_000),
      retryable: true,
    }))

  const loadFailureDetails = (error: LocalInferenceError) => ({
    code: "code" in error ? error.code : error._tag,
    message: error.message,
    retryable: "retryable" in error ? error.retryable : false,
  })

  const beginLoadState = (providerModelId: ProviderModelId) =>
    updateMatchingLocalSlots(providerModelId, (current) => {
      switch (current._tag) {
        case "Ready": {
          const unloaded = ModelSlotLifecycle.transition(current, "UnloadedLocalModel", {})
          return ModelSlotLifecycle.transition(unloaded, "LoadingLocalModel", { percentage: 0 })
        }
        case "UnloadedLocalModel":
        case "UnloadingLocalModel":
        case "Blocked":
          return ModelSlotLifecycle.transition(current, "LoadingLocalModel", { percentage: 0 })
        case "LoadingLocalModel":
          return ModelSlotLifecycle.hold(current, { percentage: 0 })
      }
    })

  const beginUnloadState = (providerModelId: ProviderModelId) =>
    updateMatchingLocalSlots(providerModelId, (slot) => {
      switch (slot._tag) {
        case "Ready":
        case "LoadingLocalModel":
          return ModelSlotLifecycle.transition(slot, "UnloadingLocalModel", {})
        case "UnloadingLocalModel":
          return ModelSlotLifecycle.hold(slot, {})
        case "UnloadedLocalModel":
        case "Blocked":
          return slot
      }
    })

  const completeUnload = (providerModelId: ProviderModelId) =>
    updateMatchingLocalSlots(providerModelId, (slot) => {
      switch (slot._tag) {
        case "UnloadedLocalModel":
          return ModelSlotLifecycle.hold(slot, {})
        case "Ready":
        case "LoadingLocalModel":
        case "UnloadingLocalModel":
          return ModelSlotLifecycle.transition(slot, "UnloadedLocalModel", {})
        case "Blocked":
          return slot
      }
    })

  const restoreFailedUnload = (providerModelId: ProviderModelId) =>
    updateMatchingLocalSlots(providerModelId, (slot) =>
      slot._tag === "UnloadingLocalModel"
        ? ModelSlotLifecycle.transition(slot, "Ready", {})
        : slot)

  const terminalizeTransition = (
    operation: RuntimeTransitionKey,
    exit: Exit.Exit<void, LocalInferenceError>,
  ) => Effect.gen(function* () {
    if (Exit.isSuccess(exit)) {
      if (operation._tag === "Load") {
        yield* completeLoad(operation.providerModelId)
      } else {
        yield* completeUnload(operation.providerModelId)
      }
    } else {
      const error = transitionFailure(operation, exit.cause)
      if (operation._tag === "Load") {
        yield* blockLoad(operation.providerModelId, loadFailureDetails(error))
      } else {
        yield* restoreFailedUnload(operation.providerModelId)
      }
      yield* Effect.logWarning("Owned local-model transition failed").pipe(
        Effect.annotateLogs({
          transition: operation._tag,
          providerModelId: operation.providerModelId,
          cause: Cause.pretty(exit.cause).slice(0, 1_000),
        }),
      )
    }
  })

  const admitTransition = <AdmissionError>(
    request: Effect.Effect<
      ServiceOperationRequest<RuntimeTransitionKey, LocalInferenceError, AdmissionError>,
      AdmissionError
    >,
  ): Effect.Effect<ServiceOperationAdmission<LocalInferenceError>, AdmissionError | LocalInferenceError> =>
    transitions.admit(request).pipe(
      Effect.catchTag("ResourceRetired", () => Effect.fail(new LocalModelMutationFailed({
        code: "local_model_transition_not_admitted",
        message: "ACN is no longer accepting local model work",
        retryable: true,
      }))),
    )

  const unloadDefinition = (
    key: Extract<RuntimeTransitionKey, { readonly _tag: "Unload" }>,
  ) => ({
    activityLabel: `local-model:unload:${key.providerModelId}`,
    commit: beginUnloadState(key.providerModelId).pipe(Effect.asVoid),
    operation: modelAdmission.withPermits(1)(localRuntime.unload(key.providerModelId)),
    terminalize: (exit: Exit.Exit<void, LocalInferenceError>) =>
      terminalizeTransition(key, exit),
  })

  const admitLoad = (slotId: SlotId): Effect.Effect<
    ServiceOperationAdmission<LocalInferenceError>,
    LocalInferenceError
  > => admitTransition(Effect.gen(function* () {
      const slot = yield* selectedSlot(slotId)
      if (slot._tag === "Unassigned" || slot.selection.providerId !== LOCAL_PROVIDER_ID) {
        return yield* reject(slotId, "The slot does not contain a local model")
      }
      const providerModelId = slot.selection.providerModelId
      const key = { _tag: "Load" as const, providerModelId }
      return {
        key,
        whenIdle: Effect.gen(function* () {
          if (slot._tag === "Ready" && (yield* localRuntime.isResident(providerModelId))) {
            return Option.none()
          }
          if (slot._tag === "Blocked"
            && slot.reason._tag !== "LocalModelLoadFailed"
            && slot.reason._tag !== "LocalModelRuntimeLost"
            && slot.reason._tag !== "LocalModelStoppedLowMemory") {
            return yield* reject(slotId, "The selected local model is not loadable")
          }
          if (slot._tag !== "UnloadedLocalModel"
            && slot._tag !== "Blocked"
            && slot._tag !== "Ready"
            && slot._tag !== "LoadingLocalModel") {
            return yield* reject(slotId, "The selected local model is not loadable")
          }
          const catalogModel = catalogContents((yield* catalog.snapshot).state).models.find((model) =>
            model.providerId === LOCAL_PROVIDER_ID && model.providerModelId === providerModelId)
          if (!catalogModel) return yield* reject(slotId, "The selected local model is unavailable")
          return Option.some({
            activityLabel: `local-model:load:${providerModelId}`,
            commit: beginLoadState(providerModelId).pipe(Effect.asVoid),
            operation: modelAdmission.withPermits(1)(localRuntime.load(
              providerModelId,
              (progress) => updateLoadProgress(providerModelId, progress),
            )),
            terminalize: (exit: Exit.Exit<void, LocalInferenceError>) =>
              terminalizeTransition(key, exit),
          })
        }),
      }
    }))

  const admitUnloadProviderModel = (
    providerModelId: ProviderModelId,
  ): Effect.Effect<ServiceOperationAdmission<LocalInferenceError>, LocalInferenceError> => {
    const key = { _tag: "Unload" as const, providerModelId }
    return admitTransition(Effect.succeed({
      key,
      whenIdle: Effect.gen(function* () {
        if (!(yield* localRuntime.isResident(providerModelId))) return Option.none()
        return Option.some(unloadDefinition(key))
      }),
    }))
  }

  const admitUnload = (
    slotId: SlotId,
  ): Effect.Effect<ServiceOperationAdmission<LocalInferenceError>, LocalInferenceError> =>
    admitTransition(Effect.gen(function* () {
      const slot = yield* selectedSlot(slotId)
      if (slot._tag === "Unassigned" || slot.selection.providerId !== LOCAL_PROVIDER_ID) {
        return yield* reject(slotId, "The slot does not contain a local model")
      }
      const providerModelId = slot.selection.providerModelId
      const key = { _tag: "Unload" as const, providerModelId }
      return {
        key,
        whenIdle: Effect.gen(function* () {
          if (isModelSlotUnloadSatisfied(slot)) return Option.none()
          if (slot._tag !== "Ready" && slot._tag !== "UnloadingLocalModel") {
            return yield* reject(slotId, "The selected local model is not loaded")
          }
          if (!(yield* localRuntime.isResident(providerModelId))) return Option.none()
          return Option.some(unloadDefinition(key))
        }),
      }
    }))

  const awaitAdmission = (
    retry: Effect.Effect<ServiceOperationAdmission<LocalInferenceError>, LocalInferenceError>,
    admission: ServiceOperationAdmission<LocalInferenceError>,
  ): Effect.Effect<void, LocalInferenceError> => {
    switch (admission._tag) {
      case "Satisfied":
        return Effect.void
      case "Current":
        return admission.outcome.pipe(Effect.flatMap((exit) =>
          Exit.isSuccess(exit) ? Effect.void : Effect.failCause(exit.cause)))
      case "Conflicting":
        return admission.outcome.pipe(
          Effect.zipRight(Effect.suspend(() => retry)),
          Effect.flatMap((next) => awaitAdmission(retry, next)),
        )
    }
  }

  const awaitLoad = (slotId: SlotId): Effect.Effect<void, LocalInferenceError> =>
    Effect.suspend(() => admitLoad(slotId)).pipe(
      Effect.flatMap((admission) => awaitAdmission(admitLoad(slotId), admission)),
    )

  const awaitUnloadProviderModel = (
    providerModelId: ProviderModelId,
  ): Effect.Effect<void, LocalInferenceError> =>
    Effect.suspend(() => admitUnloadProviderModel(providerModelId)).pipe(
      Effect.flatMap((admission) =>
        awaitAdmission(admitUnloadProviderModel(providerModelId), admission)),
    )

  const updateModelSlot: ModelSlotCoordinatorApi["updateModelSlot"] = (slotId, selection) =>
    Effect.gen(function* () {
      const previous = yield* selectedSlot(slotId)
      if (Option.isNone(selection) && previous._tag === "Unassigned") return
      const normalizedSelection = Option.isSome(selection)
        ? Option.some(yield* normalizeAndValidateSelection(slotId, selection.value))
        : Option.none<SlotSelection>()
      if (Option.isSome(normalizedSelection) && previous._tag !== "Unassigned"
        && sameSelection(previous.selection, normalizedSelection.value)) return
      yield* configuration.updateSlot(slotId, normalizedSelection).pipe(
        Effect.mapError((error) => slotFailure(slotId, "model_slot_persistence_failed", error)),
      )
      yield* reconcileAll
      if (previous._tag === "Unassigned"
        || previous.selection.providerId !== LOCAL_PROVIDER_ID
        || previous._tag === "UnloadedLocalModel"
        || previous._tag === "Blocked") return
      const configured = (yield* configuration.get).slots
      const stillSelected = [configured.primary, configured.secondary].some((configuredSelection) =>
        Option.exists(configuredSelection, (value) => value.providerId === LOCAL_PROVIDER_ID
          && value.providerModelId === previous.selection.providerModelId))
      if (stillSelected) return
      yield* Effect.forkIn(
        awaitUnloadProviderModel(previous.selection.providerModelId).pipe(
          Effect.catchAll((error) =>
            Effect.logWarning("Follow-up local model unload failed").pipe(
              Effect.annotateLogs({
                slotId,
                providerModelId: previous.selection.providerModelId,
                error: error.message,
              }),
            ),
          ),
        ),
        scope,
      ).pipe(Effect.uninterruptible)
    })

  const setModelFavorite: ModelSlotCoordinatorApi["setModelFavorite"] = (model, favorite) =>
    Effect.gen(function* () {
      yield* configuration.setFavorite(model, favorite).pipe(
        Effect.mapError(() => new ModelPreferenceMutationFailed({
          message: "Failed to save model favorite",
        })),
      )
      yield* reconcileAll
    })

  const loadModel: ModelSlotCoordinatorApi["loadModel"] = (slotId) =>
    Effect.gen(function* () {
      const admission = yield* admitLoad(slotId)
      if (admission._tag === "Conflicting") {
        return yield* reject(slotId, "Another local model transition is already active")
      }
    })

  const unloadModel: ModelSlotCoordinatorApi["unloadModel"] = (slotId) =>
    Effect.gen(function* () {
      const admission = yield* admitUnload(slotId)
      if (admission._tag === "Conflicting") {
        return yield* reject(slotId, "Another local model transition is already active")
      }
    })

  const unloadModelAndWait: ModelSlotCoordinatorApi["unloadModelAndWait"] = (slotId) =>
    Effect.suspend(() => admitUnload(slotId)).pipe(
      Effect.flatMap((admission) => awaitAdmission(admitUnload(slotId), admission)),
    )

  const acquireLocalModel: ModelSlotCoordinatorApi["acquireLocalModel"] = (
    slotId,
    providerModelId,
  ) => {
    const acquireReadyAdmission: Effect.Effect<void, LocalInferenceError> = Effect.suspend(() =>
      Effect.gen(function* () {
        const slot = yield* selectedSlot(slotId)
        if (slot._tag === "Unassigned"
          || slot.selection.providerId !== LOCAL_PROVIDER_ID
          || slot.selection.providerModelId !== providerModelId) {
          return yield* reject(slotId, "The local model request no longer matches the selected slot")
        }
        yield* awaitLoad(slotId)
        const ready = yield* Effect.uninterruptible(Effect.gen(function* () {
          yield* modelAdmission.take(1)
          const current = yield* selectedSlot(slotId)
          const isReady = current._tag === "Ready"
            && current.selection.providerId === LOCAL_PROVIDER_ID
            && current.selection.providerModelId === providerModelId
            && (yield* localRuntime.isResident(providerModelId))
          if (!isReady) yield* modelAdmission.release(1)
          return isReady
        }))
        if (!ready) return yield* acquireReadyAdmission
      }),
    )
    return Effect.acquireRelease(
      acquireReadyAdmission,
      () => modelAdmission.release(1),
    ).pipe(Effect.asVoid)
  }

  return ModelSlotCoordinator.of({
    snapshot: mirror.get,
    changes: mirror.changes,
    agentModelConfiguration: SubscriptionRef.get(agentConfiguration),
    agentModelConfigurations: agentConfiguration.changes,
    acquireLocalModel,
    updateModelSlot,
    setModelFavorite,
    loadModel,
    unloadModel,
    unloadModelAndWait,
  })
}))
