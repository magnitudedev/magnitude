import { Context, Effect, Layer, Option, Schema, Scope, Stream, SubscriptionRef } from "effect"
import {
  buildConfigStateFromSlots,
  sameConfigStateValue,
  type ConfigState,
} from "@magnitudedev/agent"
import {
  modelSlotActions,
  ModelPreferenceMutationFailed,
  ModelSlotLifecycle,
  ModelSlotMutationFailed,
  ModelSlotMutationRejected,
  ModelSlotUnassigned,
  Models,
  ModelSlotsStateSchema,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  type ModelFailure,
  type ModelSlot,
  type ModelSlotAvailability,
  type ModelSlotDescriptor,
  type ModelSlotsState,
  type ModelSlotUpdateError,
  type ProviderCatalogEntry,
  type ProviderModelCatalogEntry,
  type ProviderModelCatalogState,
  type ProviderModelIdentity,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/acn-protocol"
import { MagnitudeStorage } from "@magnitudedev/storage"
import { ReasoningEffortSchema, type ProviderId, type ProviderModelId } from "@magnitudedev/providers/client"
import { projectInferenceResidency } from "@magnitudedev/acn-protocol"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import { IcnInstances } from "@magnitudedev/icn"
import { ModelSelection } from "./model-selection"
import { AcnChanges } from "./changes"
import { LocalProviderOfferings } from "./local-provider-offerings"
import { LocalModels } from "./local-models"
import { ProviderModelCatalog } from "./provider-model-catalog"
import {
  localModelSlotAvailability,
  selectableModelCapabilities,
} from "./model-slot-projection"

export interface ModelSlotControllerApi {
  readonly state: Effect.Effect<ModelSlotsState>
  readonly changes: Stream.Stream<ModelSlotsState>
  readonly agentModelConfiguration: Effect.Effect<ConfigState>
  readonly agentModelConfigurationChanges: Stream.Stream<ConfigState>
  readonly refresh: Effect.Effect<void>
  readonly updateModelSlot: (
    slotId: SlotId,
    selection: Option.Option<SlotSelection>,
  ) => Effect.Effect<void, ModelSlotUpdateError>
  readonly setModelFavorite: (
    model: ProviderModelIdentity,
    favorite: boolean,
  ) => Effect.Effect<void, ModelPreferenceMutationFailed>
}

export class ModelSlotController extends Context.Tag("ModelSlotController")<
  ModelSlotController,
  ModelSlotControllerApi
>() {}

interface ControllerAggregate {
  readonly state: ModelSlotsState
  readonly agentConfiguration: ConfigState
}

const sameSelection = (left: SlotSelection, right: SlotSelection): boolean =>
  left.providerId === right.providerId
  && left.providerModelId === right.providerModelId
  && left.reasoningEffort === right.reasoningEffort

const slotKey = (slotId: SlotId): "primary" | "secondary" =>
  slotId === PRIMARY_SLOT_ID ? "primary" : "secondary"

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

const catalogContents = (state: ProviderModelCatalogState): {
  readonly providers: readonly ProviderCatalogEntry[]
  readonly models: readonly ProviderModelCatalogEntry[]
  readonly failures: readonly { readonly _tag: string; readonly message: string; readonly providerId?: ProviderId }[]
} => {
  switch (state._tag) {
    case "Loading":
      return { providers: [], models: [], failures: [] }
    case "Ready":
      return { providers: state.providers, models: state.models, failures: [] }
    case "Refreshing":
    case "Degraded":
      return { providers: state.providers, models: state.models, failures: state.failures }
    case "Unavailable":
      return { providers: state.providers, models: [], failures: state.failures }
  }
}

const modelFailure = (
  code: string,
  message: string,
  retryable: boolean,
): ModelFailure => ({ code, message, retryable })


export const ModelSlotControllerLive: Layer.Layer<
  ModelSlotController,
  never,
  ModelSelection | MagnitudeStorage | LocalModels | LocalProviderOfferings
    | ProviderModelCatalog | IcnInstances | AcnChanges
> = Layer.scoped(ModelSlotController, Effect.gen(function* () {
  const modelSelection = yield* ModelSelection
  const storage = yield* MagnitudeStorage
  const localModels = yield* LocalModels
  const localOfferings = yield* LocalProviderOfferings
  const catalog = yield* ProviderModelCatalog
  const instances = yield* IcnInstances
  const changes = yield* AcnChanges
  const scope = yield* Scope.Scope
  const stateLock = yield* Effect.makeSemaphore(1)

  const initialSelection = yield* modelSelection.get
  const configuredContextLimits = yield* storage.config.getContextLimitPolicy().pipe(Effect.orDie)
  const initialCatalog = yield* catalog.state
  const emptyState: ModelSlotsState = {
    slots: {
      primary: new ModelSlotUnassigned({ slotId: PRIMARY_SLOT_ID }),
      secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
    },
    recentModels: initialSelection.recentModels,
    favoriteModels: initialSelection.favorites,
  }
  const aggregate = yield* SubscriptionRef.make<ControllerAggregate>({
    state: emptyState,
    agentConfiguration: buildConfigStateFromSlots(
      catalogContents(initialCatalog).models,
      emptyState.slots,
      configuredContextLimits,
    ),
  })
  const commit = (
    state: ModelSlotsState,
    catalogModels: readonly ProviderModelCatalogEntry[],
    contextLimits: typeof configuredContextLimits,
  ) => Effect.gen(function* () {
    const previous = yield* SubscriptionRef.get(aggregate)
    const stateChanged = !Schema.equivalence(
      ModelSlotsStateSchema,
    )(previous.state, state)
    const candidateAgentConfiguration = buildConfigStateFromSlots(
      catalogModels,
      state.slots,
      contextLimits,
    )
    const agentConfigurationChanged = !sameConfigStateValue(
      previous.agentConfiguration,
      candidateAgentConfiguration,
    )
    if (!stateChanged && !agentConfigurationChanged) {
      return previous
    }
    const next: ControllerAggregate = {
      state: stateChanged ? state : previous.state,
      agentConfiguration: agentConfigurationChanged
        ? candidateAgentConfiguration
        : previous.agentConfiguration,
    }
    yield* SubscriptionRef.set(aggregate, next)
    if (stateChanged) {
      yield* changes.publish({ operation: Models.getSlots._tag })
    }
    return next
  })

  const unavailable = (
    code: string,
    message: string,
    retryable = true,
  ): ModelSlotAvailability => ({
    _tag: "Unavailable",
    failure: modelFailure(code, message, retryable),
  })

  const providerAvailability = (
    selection: SlotSelection,
    catalogState: ProviderModelCatalogState,
  ): ModelSlotAvailability => {
    if (catalogState._tag === "Loading") return { _tag: "Pending" }
    const refreshing = catalogState._tag === "Refreshing"
    const contents = catalogContents(catalogState)
    const providerFailure = contents.failures.find((item) =>
      item._tag === "ProviderFailure" && item.providerId === selection.providerId)
    if (providerFailure) {
      return unavailable("provider_unavailable", providerFailure.message)
    }
    const provider = contents.providers.find((item) => item.providerId === selection.providerId)
    if (!provider) {
      if (refreshing) return { _tag: "Pending" }
      return unavailable("provider_unavailable", "The selected provider is unavailable")
    }
    if (provider.authentication === "NotConfigured") {
      return unavailable("provider_not_configured", "The selected provider is not configured", false)
    }
    if (provider.availability._tag !== "Available") {
      const message = provider.availability._tag === "Failed"
        ? provider.availability.message
        : Option.getOrElse(provider.availability.message, () => "The selected provider is unavailable")
      return unavailable("provider_unavailable", message)
    }
    const model = contents.models.find((item) =>
      item.providerId === selection.providerId
      && item.providerModelId === selection.providerModelId)
    if (!model) {
      if (refreshing) return { _tag: "Pending" }
      return unavailable("model_unavailable", "The selected model is unavailable")
    }
    if (model.availability._tag !== "Available") {
      return unavailable("model_unavailable", "The selected model is unavailable")
    }
    return { _tag: "Available" }
  }

  const descriptorFor = (
    selection: SlotSelection,
    models: readonly ProviderModelCatalogEntry[],
  ): Option.Option<ModelSlotDescriptor> => {
    const model = models.find((item) =>
      item.providerId === selection.providerId
      && item.providerModelId === selection.providerModelId)
    return model === undefined ? Option.none() : Option.some({
      providerId: selection.providerId,
      providerModelId: selection.providerModelId,
      displayName: model.displayName,
      variantLabel: model.variantLabel,
    })
  }

  const rebuild = stateLock.withPermits(1)(Effect.gen(function* () {
    const configured = yield* modelSelection.get
    const catalogState = yield* catalog.state
    const contents = catalogContents(catalogState)
    const localOfferingsReady = yield* localOfferings.ready
    const previousAggregate = yield* SubscriptionRef.get(aggregate)
    const previous = previousAggregate.state
    const offerings = yield* localOfferings.list
    const instanceState = yield* instances.get

    const buildSlot = (slotId: SlotId, selection: Option.Option<SlotSelection>): ModelSlot =>
      Option.match(selection, {
        onNone: () => {
          const current = previous.slots[slotKey(slotId)]
          switch (current._tag) {
            case "Unassigned":
              return ModelSlotLifecycle.hold(current, { slotId })
            case "Resolving":
            case "ConfiguredRemote":
            case "ConfiguredLocal":
              return ModelSlotLifecycle.transition(current, "Unassigned", { slotId })
          }
        },
        onSome: (selected) => {
          const descriptor = descriptorFor(selected, contents.models)
          const current = previous.slots[slotKey(slotId)]
          if (Option.isNone(descriptor)) {
            const props = { slotId, selection: selected } as const
            return current._tag === "Resolving"
              ? ModelSlotLifecycle.hold(current, props)
              : ModelSlotLifecycle.transition(current, "Resolving", props)
          }
          const baseAvailability = providerAvailability(selected, catalogState)
          if (selected.providerId !== LOCAL_PROVIDER_ID) {
            const props = {
              slotId,
              selection: selected,
              descriptor: descriptor.value,
              availability: baseAvailability,
              actions: [],
            } as const
            switch (current._tag) {
              case "ConfiguredRemote":
                return ModelSlotLifecycle.hold(current, props)
              case "Unassigned":
              case "Resolving":
              case "ConfiguredLocal":
                return ModelSlotLifecycle.transition(current, "ConfiguredRemote", props)
            }
          }
          const offering = offerings.find((item) =>
            item.providerModelId === selected.providerModelId)
          const availability = localModelSlotAvailability({
            catalogIdentityPending: baseAvailability._tag === "Pending",
            offeringsReady: localOfferingsReady,
            offeringExists: offering !== undefined,
          })
          const instance = instanceState.instances.findLast((candidate) =>
            candidate.modelId === selected.providerModelId)
          const residency = instance === undefined
            ? { _tag: "Unloaded" as const }
            : projectInferenceResidency(instance)
          const props = {
            slotId,
            selection: selected,
            descriptor: descriptor.value,
            availability,
            residency,
            actions: modelSlotActions(availability, residency),
          } as const
          switch (current._tag) {
            case "ConfiguredLocal":
              return ModelSlotLifecycle.hold(current, props)
            case "Unassigned":
            case "Resolving":
            case "ConfiguredRemote":
              return ModelSlotLifecycle.transition(current, "ConfiguredLocal", props)
          }
        },
      })

    const state: ModelSlotsState = {
      slots: {
        primary: buildSlot(PRIMARY_SLOT_ID, configured.slots.primary),
        secondary: buildSlot(SECONDARY_SLOT_ID, configured.slots.secondary),
      },
      recentModels: configured.recentModels,
      favoriteModels: configured.favorites,
    }
    return yield* commit(
      state,
      contents.models,
      configuredContextLimits,
    )
  }))

  const providerIdentityIsAuthoritative = (
    providerId: ProviderId,
    state: ProviderModelCatalogState,
  ): boolean => state._tag === "Ready"
    || state._tag === "Degraded"
      && !state.failures.some((item) =>
        item._tag === "ProviderFailure" && item.providerId === providerId)

  const reconcileSelections = Effect.gen(function* () {
    const configured = yield* modelSelection.get
    const catalogState = yield* catalog.state
    const contents = catalogContents(catalogState)
    const localReady = yield* localOfferings.ready
    const localIds = new Set<string>((yield* localModels.state).models.map(({ modelId }) => modelId))
    for (const slotId of [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID] as const) {
      const selected = configured.slots[slotKey(slotId)]
      if (Option.isNone(selected)) continue
      const selection = selected.value
      if (selection.providerId === LOCAL_PROVIDER_ID && !localReady) continue
      if (!providerIdentityIsAuthoritative(selection.providerId, catalogState)) continue
      const exists = selection.providerId === LOCAL_PROVIDER_ID
        ? localIds.has(selection.providerModelId)
        : contents.models.some((model) => model.providerId === selection.providerId
          && model.providerModelId === selection.providerModelId)
      if (exists === false) {
        yield* modelSelection.updateSlot(slotId, Option.none()).pipe(Effect.orDie)
      }
    }
  })

  const reconcileAndRebuild = reconcileSelections.pipe(Effect.zipRight(rebuild))

  yield* reconcileAndRebuild
  yield* Effect.forkIn(modelSelection.changes.pipe(
    Stream.runForEach(() => reconcileAndRebuild),
  ), scope)
  yield* Effect.forkIn(localOfferings.changes.pipe(
    Stream.runForEach(() => reconcileAndRebuild),
  ), scope)
  yield* Effect.forkIn(catalog.changes.pipe(
    Stream.runForEach(() => reconcileAndRebuild),
  ), scope)
  yield* Effect.forkIn(instances.changes.pipe(Stream.runForEach(() => rebuild)), scope)

  const reject = (slotId: SlotId, message: string) =>
    new ModelSlotMutationRejected({ slotId, message })

  const normalizeAndValidateSelection = (
    slotId: SlotId,
    selection: SlotSelection,
  ): Effect.Effect<SlotSelection, ModelSlotMutationRejected> => Effect.gen(function* () {
    const contents = catalogContents(yield* catalog.state)
    const model = contents.models.find((item) =>
      item.providerId === selection.providerId
      && item.providerModelId === selection.providerModelId)
    const offering = selection.providerId === LOCAL_PROVIDER_ID
      ? yield* Effect.option(localOfferings.resolve(selection.providerModelId))
      : Option.none()
    if (selection.providerId === LOCAL_PROVIDER_ID && Option.isNone(offering)) {
      return yield* reject(slotId, "The selected local model configuration is unavailable")
    }
    const capabilities = selectableModelCapabilities(
      slotId,
      model,
      Option.getOrUndefined(Option.map(offering, (value) => ({
        capabilities: value.capabilities,
      }))),
    )
    if (!capabilities) {
      return yield* reject(slotId, "The selected model is unavailable for this slot")
    }
    return normalizeSelectionReasoning(selection, { capabilities })
  })

  const requireSelectedLocalOffering = (
    slotId: SlotId,
    selection: SlotSelection,
  ): Effect.Effect<void, ModelSlotUpdateError> => {
    if (selection.providerId !== LOCAL_PROVIDER_ID) return Effect.void
    return localOfferings.resolve(selection.providerModelId).pipe(
      Effect.asVoid,
      Effect.mapError(() => new ModelSlotMutationRejected({
        slotId,
        message: "The selected local model offering is unavailable",
      })),
    )
  }

  const updateModelSlot: ModelSlotControllerApi["updateModelSlot"] = (slotId, selection) =>
    Effect.gen(function* () {
      if (Option.isSome(selection)) {
        yield* requireSelectedLocalOffering(slotId, selection.value)
      }
      const normalized = Option.isSome(selection)
        ? Option.some(yield* normalizeAndValidateSelection(slotId, selection.value))
        : Option.none<SlotSelection>()
      const previous = (yield* SubscriptionRef.get(aggregate)).state.slots[slotKey(slotId)]
      if (Option.isNone(normalized) && previous._tag === "Unassigned") return
      if (Option.isSome(normalized) && previous._tag !== "Unassigned"
        && sameSelection(previous.selection, normalized.value)) return
      yield* modelSelection.updateSlot(slotId, normalized).pipe(
        Effect.mapError((error) => new ModelSlotMutationFailed({
          slotId,
          code: "model_slot_persistence_failed",
          message: String(error),
          retryable: true,
        })),
      )
      yield* rebuild
    })

  const setModelFavorite: ModelSlotControllerApi["setModelFavorite"] = (model, favorite) =>
    modelSelection.setFavorite(model, favorite).pipe(
      Effect.mapError(() => new ModelPreferenceMutationFailed({
        message: "Failed to save model favorite",
      })),
      Effect.zipRight(rebuild),
      Effect.asVoid,
    )

  return ModelSlotController.of({
    state: SubscriptionRef.get(aggregate).pipe(Effect.map(({ state }) => state)),
    changes: aggregate.changes.pipe(
      Stream.map(({ state }) => state),
      Stream.changesWith(Schema.equivalence(ModelSlotsStateSchema)),
    ),
    agentModelConfiguration: SubscriptionRef.get(aggregate).pipe(
      Effect.map(({ agentConfiguration }) => agentConfiguration),
    ),
    agentModelConfigurationChanges: aggregate.changes.pipe(
      Stream.map(({ agentConfiguration }) => agentConfiguration),
      Stream.changesWith(sameConfigStateValue),
    ),
    refresh: rebuild.pipe(Effect.asVoid),
    updateModelSlot,
    setModelFavorite,
  })
}))
