import {
  Cause,
  Context,
  Deferred,
  Either,
  Effect,
  Exit,
  Layer,
  Option,
  Ref,
  Schema,
  Scope,
  Stream,
  SubscriptionRef,
} from "effect"
import {
  buildConfigStateFromSlots,
  sameConfigStateValue,
  type ConfigState,
} from "@magnitudedev/agent"
import {
  LocalModelMutationFailed,
  modelSlotActions,
  servableModelBundlePackageIds,
  ModelInstanceIdSchema,
  ModelPreferenceMutationFailed,
  ModelSlotLifecycle,
  ModelSlotMutationFailed,
  ModelSlotMutationRejected,
  ModelSlotUnassigned,
  ModelSlotsMirror,
  ModelServingConfigurationIdSchema,
  ModelServingConfigurationSchema,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  type LocalInferenceError,
  type LocalProviderOffering,
  type MirroredSnapshot,
  type ModelFailure,
  type ModelInstanceId,
  type ModelLoadPlan,
  type ModelReleaseReason,
  type ModelSlot,
  type ModelSlotAvailability,
  type ModelSlotDescriptor,
  type ModelResidency,
  type ModelSlotsState,
  type ModelSlotUpdateError,
  type ModelServingConfigurationId,
  type ProviderCatalogEntry,
  type ProviderModelCatalogEntry,
  type ProviderModelCatalogState,
  type ProviderModelIdentity,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/acn-protocol"
import {
  CurrentModelInstance,
  IcnClient,
  IcnInstances,
} from "@magnitudedev/icn"
import type * as Generated from "@magnitudedev/icn-protocol/schemas"
import { MagnitudeStorage } from "@magnitudedev/storage"
import {
  ReasoningEffortSchema,
  type ProviderId,
  type ProviderModelId,
} from "@magnitudedev/sdk"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import { ModelSelection } from "./model-selection"
import { MirroredStateChanges } from "./mirrored-state"
import { LocalModelPackages } from "./local-model-packages"
import { LocalProviderOfferings } from "./local-provider-offerings"
import { ProviderModelCatalog } from "./provider-model-catalog"
import { modelServingConfigurationToIcn } from "./local-model-icn-adapter"
import {
  localModelSlotAvailability,
  projectModelResidency,
  projectModelLoadPlan,
  selectableModelCapabilities,
} from "./model-slot-projection"

export interface ModelSlotControllerApi {
  readonly snapshot: Effect.Effect<MirroredSnapshot<ModelSlotsState>>
  readonly changes: Stream.Stream<MirroredSnapshot<ModelSlotsState>>
  readonly agentModelConfiguration: Effect.Effect<ConfigState>
  readonly agentModelConfigurationChanges: Stream.Stream<ConfigState>
  readonly acquireLocalModel: (
    slotId: SlotId,
    providerModelId: ProviderModelId,
  ) => Effect.Effect<ModelLoadResult, LocalInferenceError, Scope.Scope>
  readonly updateModelSlot: (
    slotId: SlotId,
    selection: Option.Option<SlotSelection>,
  ) => Effect.Effect<void, ModelSlotUpdateError>
  readonly setModelFavorite: (
    model: ProviderModelIdentity,
    favorite: boolean,
  ) => Effect.Effect<void, ModelPreferenceMutationFailed>
  readonly loadModel: (slotId: SlotId) => Effect.Effect<void, LocalInferenceError>
  readonly previewModelLoad: (slotId: SlotId) => Effect.Effect<ModelLoadPlan, LocalInferenceError>
  readonly stopModel: (slotId: SlotId) => Effect.Effect<void, LocalInferenceError>
}

export class ModelSlotController extends Context.Tag("ModelSlotController")<
  ModelSlotController,
  ModelSlotControllerApi
>() {}

interface ControllerAggregate {
  readonly snapshot: MirroredSnapshot<ModelSlotsState>
  readonly agentConfiguration: ConfigState
  readonly loadTargets: Readonly<Record<"primary" | "secondary", Option.Option<SlotLoadTarget>>>
  readonly instanceBindings: Readonly<Record<
    "primary" | "secondary",
    Option.Option<SlotInstanceBinding>
  >>
}

interface SlotLoadTarget {
  readonly selection: SlotSelection
  readonly configuration: LocalProviderOffering["configuration"]
}

interface SlotInstanceBinding {
  readonly configurationId: ModelServingConfigurationId
  readonly instanceId: ModelInstanceId
}

interface ModelLoadCommand {
  readonly requestedBy: SlotId
  readonly configuration: LocalProviderOffering["configuration"]
  readonly instanceId: ModelInstanceId
  readonly admission: Deferred.Deferred<void, LocalInferenceError>
}

type ModelLoadResult =
  | {
      readonly _tag: "Ready"
      readonly instanceId: ModelInstanceId
      readonly configurationId: ModelServingConfigurationId
    }
  | {
      readonly _tag: "Cancelled"
      readonly reason: ModelReleaseReason
    }

interface PendingLoadRequest {
  readonly _tag: "Pending"
  readonly selection: SlotSelection
  readonly cancellation: Deferred.Deferred<void>
}

interface FailedLoadRequest {
  readonly _tag: "Failed"
  readonly selection: SlotSelection
  readonly failure: ModelFailure
}

type LoadRequest = PendingLoadRequest | FailedLoadRequest

type LoadDecision =
  | { readonly _tag: "Waiting" }
  | { readonly _tag: "Cancelled" }
  | { readonly _tag: "Satisfied" }
  | { readonly _tag: "Load"; readonly waitOnRetryableFailure: boolean }
  | { readonly _tag: "Failed"; readonly failure: ModelFailure }

type ModelLoadClaim =
  | { readonly _tag: "Complete"; readonly result: Extract<ModelLoadResult, { _tag: "Ready" }> }
  | { readonly _tag: "Observe"; readonly instanceId: ModelInstanceId }
  | { readonly _tag: "Join"; readonly command: ModelLoadCommand }
  | { readonly _tag: "Owner"; readonly command: ModelLoadCommand }

const sameSelection = (left: SlotSelection, right: SlotSelection): boolean =>
  left.providerId === right.providerId
  && left.providerModelId === right.providerModelId
  && left.reasoningEffort === right.reasoningEffort

const sameServingConfiguration = Schema.equivalence(ModelServingConfigurationSchema)

const sameLoadTarget = (
  left: Option.Option<SlotLoadTarget>,
  right: Option.Option<SlotLoadTarget>,
): boolean => Option.match(left, {
  onNone: () => Option.isNone(right),
  onSome: (leftTarget) => Option.exists(right, (rightTarget) =>
    sameSelection(leftTarget.selection, rightTarget.selection)
    && sameServingConfiguration(leftTarget.configuration, rightTarget.configuration)),
})

const sameInstanceBinding = (
  left: Option.Option<SlotInstanceBinding>,
  right: Option.Option<SlotInstanceBinding>,
): boolean => Option.match(left, {
  onNone: () => Option.isNone(right),
  onSome: (leftBinding) => Option.exists(right, (rightBinding) =>
    leftBinding.configurationId === rightBinding.configurationId
    && leftBinding.instanceId === rightBinding.instanceId),
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

const failure = (
  code: string,
  error: string | { readonly message: string },
  retryable = true,
): LocalModelMutationFailed => new LocalModelMutationFailed({
  code,
  message: typeof error === "string" ? error : error.message,
  retryable,
})

const modelFailure = (
  code: string,
  message: string,
  retryable: boolean,
): ModelFailure => ({ code, message, retryable })

export const ModelSlotControllerLive: Layer.Layer<
  ModelSlotController,
  never,
  ModelSelection | MagnitudeStorage | LocalModelPackages | LocalProviderOfferings
    | ProviderModelCatalog | MirroredStateChanges | IcnClient | IcnInstances
> = Layer.scoped(ModelSlotController, Effect.gen(function* () {
  const modelSelection = yield* ModelSelection
  const storage = yield* MagnitudeStorage
  const localPackages = yield* LocalModelPackages
  const localOfferings = yield* LocalProviderOfferings
  const catalog = yield* ProviderModelCatalog
  const mirroredChanges = yield* MirroredStateChanges
  const client = yield* IcnClient
  const observedInstances = yield* IcnInstances
  const scope = yield* Scope.Scope
  const stateLock = yield* Effect.makeSemaphore(1)
  const commandLock = yield* Effect.makeSemaphore(1)
  const loadCommands = yield* Ref.make<
    ReadonlyMap<ModelServingConfigurationId, ModelLoadCommand>
  >(new Map())
  const loadRequests = yield* Ref.make<Readonly<Record<
    "primary" | "secondary",
    Option.Option<LoadRequest>
  >>>({ primary: Option.none(), secondary: Option.none() })
  const clearLoadRequestUnsafe = (slotId: SlotId) => Effect.gen(function* () {
    const key = slotKey(slotId)
    const requests = yield* Ref.get(loadRequests)
    const current = requests[key]
    if (Option.isSome(current) && current.value._tag === "Pending") {
      yield* Deferred.succeed(current.value.cancellation, undefined)
    }
    if (Option.isSome(current)) {
      yield* Ref.set(loadRequests, { ...requests, [key]: Option.none() })
    }
    return current
  })

  const initialSelection = yield* modelSelection.get
  const configuredContextLimits = yield* storage.config.getContextLimitPolicy().pipe(Effect.orDie)
  const initialCatalog = (yield* catalog.snapshot).state
  const emptyState: ModelSlotsState = {
    slots: {
      primary: new ModelSlotUnassigned({ slotId: PRIMARY_SLOT_ID }),
      secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
    },
    recentModels: initialSelection.recentModels,
    favoriteModels: initialSelection.favorites,
  }
  const aggregate = yield* SubscriptionRef.make<ControllerAggregate>({
    snapshot: { revision: 0, state: emptyState },
    agentConfiguration: buildConfigStateFromSlots(
      catalogContents(initialCatalog).models,
      emptyState.slots,
      configuredContextLimits,
    ),
    loadTargets: {
      primary: Option.none(),
      secondary: Option.none(),
    },
    instanceBindings: {
      primary: Option.none(),
      secondary: Option.none(),
    },
  })
  const commit = (
    state: ModelSlotsState,
    catalogModels: readonly ProviderModelCatalogEntry[],
    contextLimits: typeof configuredContextLimits,
    loadTargets: ControllerAggregate["loadTargets"],
    instanceBindings: ControllerAggregate["instanceBindings"],
  ) => Effect.gen(function* () {
    const previous = yield* SubscriptionRef.get(aggregate)
    const stateChanged = !Schema.equivalence(
      ModelSlotsMirror.stateSchema,
    )(previous.snapshot.state, state)
    const candidateAgentConfiguration = buildConfigStateFromSlots(
      catalogModels,
      state.slots,
      contextLimits,
    )
    const agentConfigurationChanged = !sameConfigStateValue(
      previous.agentConfiguration,
      candidateAgentConfiguration,
    )
    const loadTargetsChanged = !sameLoadTarget(
      previous.loadTargets.primary,
      loadTargets.primary,
    ) || !sameLoadTarget(
      previous.loadTargets.secondary,
      loadTargets.secondary,
    )
    const instanceBindingsChanged = !sameInstanceBinding(
      previous.instanceBindings.primary,
      instanceBindings.primary,
    ) || !sameInstanceBinding(
      previous.instanceBindings.secondary,
      instanceBindings.secondary,
    )
    if (!stateChanged && !agentConfigurationChanged && !loadTargetsChanged
      && !instanceBindingsChanged) {
      return previous
    }
    const revision = stateChanged
      ? previous.snapshot.revision + 1
      : previous.snapshot.revision
    const next: ControllerAggregate = {
      snapshot: stateChanged ? { revision, state } : previous.snapshot,
      agentConfiguration: agentConfigurationChanged
        ? candidateAgentConfiguration
        : previous.agentConfiguration,
      loadTargets: loadTargetsChanged ? loadTargets : previous.loadTargets,
      instanceBindings: instanceBindingsChanged
        ? instanceBindings
        : previous.instanceBindings,
    }
    yield* SubscriptionRef.set(aggregate, next)
    if (stateChanged) {
      yield* mirroredChanges.publish({
        _tag: "changed",
        id: ModelSlotsMirror.id,
        revision,
      })
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
  ): ModelSlotDescriptor => {
    const model = models.find((item) =>
      item.providerId === selection.providerId
      && item.providerModelId === selection.providerModelId)
    return {
      providerId: selection.providerId,
      providerModelId: selection.providerModelId,
      displayName: model?.displayName || selection.providerModelId,
      variantLabel: model?.variantLabel ?? Option.none(),
    }
  }

  const rebuild = stateLock.withPermits(1)(Effect.gen(function* () {
    const configured = yield* modelSelection.get
    const catalogState = (yield* catalog.snapshot).state
    const contents = catalogContents(catalogState)
    const localOfferingsReady = yield* localOfferings.ready
    const packageState = (yield* localPackages.snapshot).state
    const packages = yield* localPackages.installedPackageIds
    const native = yield* observedInstances.get
    const previousAggregate = yield* SubscriptionRef.get(aggregate)
    const requests = yield* Ref.get(loadRequests)
    const previous = previousAggregate.snapshot.state
    const offerings = yield* localOfferings.list.pipe(Effect.orElseSucceed(() => []))

    const buildSlot = (slotId: SlotId, selection: Option.Option<SlotSelection>): ModelSlot =>
      Option.match(selection, {
        onNone: () => {
          const current = previous.slots[slotKey(slotId)]
          switch (current._tag) {
            case "Unassigned":
              return ModelSlotLifecycle.hold(current, { slotId })
            case "ConfiguredRemote":
            case "ConfiguredLocal":
              return ModelSlotLifecycle.transition(current, "Unassigned", { slotId })
          }
        },
        onSome: (selected) => {
          const descriptor = descriptorFor(selected, contents.models)
          const baseAvailability = providerAvailability(selected, catalogState)
          const current = previous.slots[slotKey(slotId)]
          if (selected.providerId !== LOCAL_PROVIDER_ID) {
            const props = {
              slotId,
              selection: selected,
              descriptor,
              availability: baseAvailability,
              actions: [],
            } as const
            switch (current._tag) {
              case "ConfiguredRemote":
                return ModelSlotLifecycle.hold(current, props)
              case "Unassigned":
              case "ConfiguredLocal":
                return ModelSlotLifecycle.transition(current, "ConfiguredRemote", props)
            }
          }
          const offering = offerings.find((item) =>
            item.providerModelId === selected.providerModelId)
          const downloaded = offering !== undefined
            && servableModelBundlePackageIds(offering.configuration.bundle)
              .every((packageId) => packages.has(packageId))
          const availability = localModelSlotAvailability({
            catalogIdentityPending: baseAvailability._tag === "Pending",
            offeringsReady: localOfferingsReady,
            inventory: packageState.inventory,
            offeringExists: offering !== undefined,
            installed: downloaded,
          })
          const configurationId = offering?.configuration.id
          const binding = previousAggregate.instanceBindings[slotKey(slotId)]
          const exact = Option.flatMap(binding, ({ configurationId: boundConfigurationId, instanceId }) =>
            boundConfigurationId === configurationId
              ? Option.fromNullable(native.instances.find((instance) =>
                instance.id === instanceId && instance.configurationId === configurationId))
              : Option.none())
          const request = Option.flatMap(
            requests[slotKey(slotId)],
            (candidate) => sameSelection(candidate.selection, selected)
              ? Option.some(candidate)
              : Option.none(),
          )
          const nativeResidency = Option.map(exact, projectModelResidency)
          const activeResidency = Option.filter(nativeResidency, (current) =>
            current._tag === "Loading"
            || current._tag === "Ready"
            || current._tag === "Stopping")
          const residency: ModelResidency = Option.getOrElse(activeResidency, () =>
            Option.match(request, {
              onSome: (current) => current._tag === "Pending"
                ? { _tag: "Requested" }
                : { _tag: "Failed", failure: current.failure },
              onNone: () => Option.getOrElse(
                nativeResidency,
                () => ({ _tag: "Unloaded" }),
              ),
            }))
          const props = {
            slotId,
            selection: selected,
            descriptor,
            availability,
            residency,
            actions: modelSlotActions(availability, residency),
          } as const
          switch (current._tag) {
            case "ConfiguredLocal":
              return ModelSlotLifecycle.hold(current, props)
            case "Unassigned":
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
    const loadTargetFor = (
      selection: Option.Option<SlotSelection>,
    ): Option.Option<SlotLoadTarget> => Option.flatMap(selection, (selected) =>
      selected.providerId === LOCAL_PROVIDER_ID
        ? Option.fromNullable(offerings.find((item) =>
            item.providerModelId === selected.providerModelId)).pipe(
            Option.map((selectedOffering) => ({
              selection: selected,
              configuration: selectedOffering.configuration,
            })),
          )
        : Option.none())
    const loadTargets: ControllerAggregate["loadTargets"] = {
      primary: loadTargetFor(configured.slots.primary),
      secondary: loadTargetFor(configured.slots.secondary),
    }
    const nextBindingFor = (
      key: "primary" | "secondary",
    ): Option.Option<SlotInstanceBinding> => Option.flatMap(
      loadTargets[key],
      (target) => Option.filter(
        previousAggregate.instanceBindings[key],
        (binding) => binding.configurationId === target.configuration.id,
      ),
    )
    return yield* commit(
      state,
      contents.models,
      configuredContextLimits,
      loadTargets,
      {
        primary: nextBindingFor("primary"),
        secondary: nextBindingFor("secondary"),
      },
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
    const catalogState = (yield* catalog.snapshot).state
    const contents = catalogContents(catalogState)
    const localReady = yield* localOfferings.ready
    const localResult = yield* Effect.either(localOfferings.list)
    const localIds = Either.isRight(localResult)
      ? new Set(localResult.right.map(({ providerModelId }) => providerModelId))
      : undefined
    for (const slotId of [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID] as const) {
      const selected = configured.slots[slotKey(slotId)]
      if (Option.isNone(selected)) continue
      const selection = selected.value
      if (selection.providerId === LOCAL_PROVIDER_ID && !localReady) continue
      if (!providerIdentityIsAuthoritative(selection.providerId, catalogState)) continue
      const exists = selection.providerId === LOCAL_PROVIDER_ID
        ? localIds?.has(selection.providerModelId)
        : contents.models.some((model) => model.providerId === selection.providerId
          && model.providerModelId === selection.providerModelId)
      if (exists === false) {
        yield* clearLoadRequestUnsafe(slotId)
        yield* modelSelection.updateSlot(slotId, Option.none()).pipe(Effect.orDie)
      }
    }
  })

  const reconcileAndRebuild = commandLock.withPermits(1)(
    reconcileSelections.pipe(Effect.zipRight(rebuild)),
  )

  yield* reconcileAndRebuild
  yield* Effect.forkIn(modelSelection.changes.pipe(
    Stream.runForEach(() => reconcileAndRebuild),
  ), scope)
  yield* Effect.forkIn(localPackages.changes.pipe(Stream.runForEach(() => rebuild)), scope)
  yield* Effect.forkIn(localOfferings.changes.pipe(
    Stream.runForEach(() => reconcileAndRebuild),
  ), scope)
  yield* Effect.forkIn(catalog.changes.pipe(
    Stream.runForEach(() => reconcileAndRebuild),
  ), scope)
  yield* Effect.forkIn(observedInstances.changes.pipe(
    Stream.runForEach(() => rebuild),
  ), scope)

  const selectedSlot = (slotId: SlotId) => SubscriptionRef.get(aggregate).pipe(
    Effect.map(({ snapshot }) => snapshot.state.slots[slotKey(slotId)]),
  )
  const bindSlotInstance = (
    slotId: SlotId,
    configurationId: ModelServingConfigurationId,
    instanceId: ModelInstanceId,
  ) => stateLock.withPermits(1)(
    SubscriptionRef.update(aggregate, (current) => ({
      ...current,
      instanceBindings: {
        ...current.instanceBindings,
        [slotKey(slotId)]: Option.some({ configurationId, instanceId }),
      },
    })),
  )
  const reject = (slotId: SlotId, message: string) =>
    new ModelSlotMutationRejected({ slotId, message })

  const normalizeAndValidateSelection = (
    slotId: SlotId,
    selection: SlotSelection,
  ): Effect.Effect<SlotSelection, ModelSlotMutationRejected> => Effect.gen(function* () {
    const contents = catalogContents((yield* catalog.snapshot).state)
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

  const awaitInstance = (
    instanceId: ModelInstanceId,
    predicate: (instance: Generated.ModelInstance) => boolean,
  ): Effect.Effect<Generated.ModelInstance, LocalInferenceError> => Effect.gen(function* () {
    yield* observedInstances.refresh.pipe(
      Effect.mapError((error) => failure("refresh_model_instances_failed", error)),
    )
    const current = (yield* observedInstances.get).instances.find((item) => item.id === instanceId)
    if (current && predicate(current)) return current
    const observed = yield* observedInstances.changes.pipe(
      Stream.filterMap((snapshot) => Option.fromNullable(
        snapshot.instances.find((item) => item.id === instanceId && predicate(item)),
      )),
      Stream.runHead,
    )
    if (Option.isSome(observed)) return observed.value
    return yield* failure(
      "model_instance_observation_ended",
      "Model-instance observation ended before the requested state was published",
    )
  })

  const awaitReadyInstance = (
    instanceId: ModelInstanceId,
  ): Effect.Effect<ModelLoadResult, LocalInferenceError> => Effect.gen(function* () {
    const instance = yield* awaitInstance(
      instanceId,
      (candidate) => candidate.lifecycle._tag === "Ready"
        || candidate.lifecycle._tag === "Stopped"
        || candidate.lifecycle._tag === "Failed",
    )
    switch (instance.lifecycle._tag) {
      case "Ready":
        return {
          _tag: "Ready" as const,
          instanceId: ModelInstanceIdSchema.make(instance.id),
          configurationId: ModelServingConfigurationIdSchema.make(instance.configurationId),
        }
      case "Failed":
        return yield* new LocalModelMutationFailed({
          code: instance.lifecycle.failure.code,
          message: instance.lifecycle.failure.message,
          retryable: instance.lifecycle.failure.retryable,
        })
      case "Stopped":
        return {
          _tag: "Cancelled" as const,
          instanceId: ModelInstanceIdSchema.make(instance.id),
          reason: instance.lifecycle.reason,
        }
      case "Loading":
      case "Stopping":
        return yield* failure(
          "invalid_model_instance_terminal_state",
          "Model-instance observation returned before the instance reached a terminal load state",
          false,
        )
    }
  })

  const commandIsCurrent = (command: ModelLoadCommand) =>
    SubscriptionRef.get(aggregate).pipe(
      Effect.map((current) => {
        const targetSelectedBy = (slotId: SlotId) => {
          const key = slotKey(slotId)
          const slot = current.snapshot.state.slots[key]
          const target = current.loadTargets[key]
          return slot._tag === "ConfiguredLocal"
            && Option.exists(target, (currentTarget) =>
              sameSelection(currentTarget.selection, slot.selection)
              && sameServingConfiguration(currentTarget.configuration, command.configuration))
        }
        return targetSelectedBy(PRIMARY_SLOT_ID) || targetSelectedBy(SECONDARY_SLOT_ID)
      }),
    )

  const runLoadCommand = (command: ModelLoadCommand) => Effect.gen(function* () {
    const stillCurrent = yield* commandLock.withPermits(1)(
      commandIsCurrent(command),
    )
    if (!stillCurrent) {
      return yield* reject(command.requestedBy, "The selected local model changed before loading")
    }
    const configuration = yield* modelServingConfigurationToIcn(command.configuration).pipe(
      Effect.mapError((error) => failure("encode_local_model_configuration_failed", error, false)),
    )
    const response = yield* client.models.loadModelInstance({
      payload: {
        instanceId: command.instanceId,
        configuration,
      },
    }).pipe(Effect.mapError((error) => failure("load_model_instance_failed", error)))
    yield* Effect.forkIn(response.events.pipe(
      Stream.runDrain,
      Effect.catchAll((error) => Effect.logWarning("ICN load observation stream ended").pipe(
        Effect.annotateLogs({ instanceId: command.instanceId, error: String(error) }),
      )),
    ), scope)
    yield* observedInstances.refresh.pipe(
      Effect.mapError((error) => failure("refresh_model_instances_failed", error)),
    )
    const remainsCurrent = yield* commandLock.withPermits(1)(
      commandIsCurrent(command),
    )
    if (!remainsCurrent) {
      yield* stopModelInstance(command.instanceId).pipe(
        Effect.catchAll((error) => Effect.logWarning("Superseded model instance stop failed").pipe(
          Effect.annotateLogs({ instanceId: command.instanceId, error: error.message }),
        )),
      )
      return yield* reject(command.requestedBy, "The selected local model changed while loading")
    }
    const published = yield* rebuild
    const publishedSlot = published.snapshot.state.slots[slotKey(command.requestedBy)]
    if (publishedSlot._tag !== "ConfiguredLocal"
      || (publishedSlot.residency._tag !== "Loading"
        && publishedSlot.residency._tag !== "Ready")
      || publishedSlot.residency.instanceId !== command.instanceId) {
      return yield* failure(
        "model_load_admission_not_published",
        "The admitted model instance was not published to its slot",
        false,
      )
    }
    yield* Deferred.succeed(command.admission, undefined)
    const result = yield* awaitReadyInstance(command.instanceId)
    yield* rebuild
    return result
  })

  const admitModelLoad = (
    slotId: SlotId,
    expectedSelection?: SlotSelection,
  ): Effect.Effect<ModelInstanceId, LocalInferenceError> => Effect.gen(function* () {
    yield* rebuild
    const candidateAdmission = yield* Deferred.make<void, LocalInferenceError>()
    const claim: ModelLoadClaim = yield* commandLock.withPermits(1)(Effect.gen(function* () {
      const current = yield* SubscriptionRef.get(aggregate)
      const key = slotKey(slotId)
      const slot = current.snapshot.state.slots[key]
      if (slot._tag !== "ConfiguredLocal") {
        return yield* reject(slotId, "The slot does not contain a local model")
      }
      if (expectedSelection !== undefined && !sameSelection(slot.selection, expectedSelection)) {
        return yield* reject(slotId, "The selected local model changed before load admission")
      }
      const loadTarget = current.loadTargets[key]
      if (Option.isNone(loadTarget)
        || !sameSelection(loadTarget.value.selection, slot.selection)) {
        return yield* reject(slotId, "The selected local model configuration is unavailable")
      }
      if (slot.residency._tag === "Ready") {
        return {
          _tag: "Complete" as const,
          result: {
            _tag: "Ready" as const,
            instanceId: slot.residency.instanceId,
            configurationId: slot.residency.configurationId,
          },
        }
      }
      if (slot.residency._tag === "Loading") {
        return {
          _tag: "Observe" as const,
          instanceId: slot.residency.instanceId,
        }
      }
      const candidate: ModelLoadCommand = {
        requestedBy: slotId,
        configuration: loadTarget.value.configuration,
        instanceId: ModelInstanceIdSchema.make(crypto.randomUUID()),
        admission: candidateAdmission,
      }
      const currentCommands = yield* Ref.get(loadCommands)
      const existing = currentCommands.get(candidate.configuration.id)
      if (existing) {
        if (!sameServingConfiguration(existing.configuration, candidate.configuration)) {
          return yield* reject(
            slotId,
            "The serving configuration ID is already in use for different configuration data",
          )
        }
        yield* bindSlotInstance(
          slotId,
          existing.configuration.id,
          existing.instanceId,
        )
        return { _tag: "Join" as const, command: existing }
      }
      yield* Ref.set(
        loadCommands,
        new Map(currentCommands).set(candidate.configuration.id, candidate),
      )
      yield* bindSlotInstance(
        slotId,
        candidate.configuration.id,
        candidate.instanceId,
      )
      return { _tag: "Owner" as const, command: candidate }
    }))
    if (claim._tag === "Complete") {
      return claim.result.instanceId
    }
    if (claim._tag === "Observe") {
      return claim.instanceId
    }
    yield* rebuild
    if (claim._tag === "Owner") {
      yield* Effect.forkIn(Effect.gen(function* () {
        const result = yield* Effect.exit(runLoadCommand(claim.command))
        yield* Deferred.done(claim.command.admission, Exit.asVoid(result))
        yield* commandLock.withPermits(1)(
          Ref.update(loadCommands, (current) => {
            const configurationId = claim.command.configuration.id
            if (current.get(configurationId) !== claim.command) return current
            const next = new Map(current)
            next.delete(configurationId)
            return next
          }),
        )
      }), scope)
    }
    yield* Deferred.await(claim.command.admission)
    const current = yield* SubscriptionRef.get(aggregate)
    const admittedSlot = current.snapshot.state.slots[slotKey(slotId)]
    if (admittedSlot._tag !== "ConfiguredLocal"
      || (admittedSlot.residency._tag !== "Loading"
        && admittedSlot.residency._tag !== "Ready")
      || admittedSlot.residency.instanceId !== claim.command.instanceId) {
      return yield* failure(
        "model_load_admission_not_published",
        "The admitted model instance was not published to its slot",
        false,
      )
    }
    return claim.command.instanceId
  })

  const previewModelLoad: ModelSlotControllerApi["previewModelLoad"] = (slotId) =>
    Effect.gen(function* () {
      const current = yield* SubscriptionRef.get(aggregate)
      const key = slotKey(slotId)
      const slot = current.snapshot.state.slots[key]
      const target = current.loadTargets[key]
      if (slot._tag !== "ConfiguredLocal" || Option.isNone(target)
        || !sameSelection(slot.selection, target.value.selection)) {
        return yield* reject(slotId, "The slot does not contain a previewable local model")
      }
      if (slot.availability._tag === "Pending") {
        return yield* reject(slotId, "The selected local model is still initializing")
      }
      if (slot.availability._tag === "Unavailable") {
        return yield* new LocalModelMutationFailed(slot.availability.failure)
      }
      if (slot.residency._tag === "Loading"
        || slot.residency._tag === "Ready"
        || slot.residency._tag === "Stopping") {
        return yield* reject(slotId, "The selected local model already has an active instance")
      }
      const configuration = yield* modelServingConfigurationToIcn(
        target.value.configuration,
      ).pipe(
        Effect.mapError((error) =>
          failure("encode_local_model_configuration_failed", error, false)),
      )
      const plan = yield* client.models.previewModelLoad({
        payload: { configuration },
      }).pipe(
        Effect.mapError((error) => failure("preview_model_load_failed", error)),
      )
      return projectModelLoadPlan(plan)
    })

  const stopModelInstance = (instanceId: ModelInstanceId) =>
    client.models.stopModelInstance({
      path: { instance_id: instanceId },
    }).pipe(
      Effect.mapError((error) => failure("stop_model_instance_failed", error)),
      Effect.asVoid,
    )

  const loadRequestFailure = (error: LocalInferenceError): ModelFailure => {
    switch (error._tag) {
      case "LocalModelMutationFailed":
      case "ModelSlotMutationFailed":
        return modelFailure(error.code, error.message, error.retryable)
      case "ModelSlotMutationRejected":
        return modelFailure("model_load_rejected", error.message, true)
    }
  }

  const transitionLoadRequest = (
    slotId: SlotId,
    expected: LoadRequest,
    next: Option.Option<LoadRequest>,
  ) => commandLock.withPermits(1)(Effect.gen(function* () {
    const key = slotKey(slotId)
    const requests = yield* Ref.get(loadRequests)
    if (!Option.exists(requests[key], (current) => current === expected)) return false
    yield* Ref.set(loadRequests, { ...requests, [key]: next })
    yield* rebuild
    return true
  }))

  const decideLoadRequest = (
    slotId: SlotId,
    request: PendingLoadRequest,
    current: ControllerAggregate,
  ): LoadDecision => {
    const key = slotKey(slotId)
    const slot = current.snapshot.state.slots[key]
    if (slot._tag !== "ConfiguredLocal"
      || !sameSelection(slot.selection, request.selection)) {
      return { _tag: "Cancelled" }
    }
    if (slot.residency._tag === "Loading" || slot.residency._tag === "Ready") {
      return { _tag: "Satisfied" }
    }
    if (slot.residency._tag === "Stopping") return { _tag: "Waiting" }
    if (slot.residency._tag === "Failed" && !slot.residency.failure.retryable) {
      return { _tag: "Failed", failure: slot.residency.failure }
    }
    if (slot.availability._tag === "Pending") return { _tag: "Waiting" }
    if (slot.availability._tag === "Unavailable") {
      if (!slot.availability.failure.retryable) {
        return { _tag: "Failed", failure: slot.availability.failure }
      }
      return Option.exists(current.loadTargets[key], (target) =>
        sameSelection(target.selection, request.selection))
        ? { _tag: "Load", waitOnRetryableFailure: true }
        : { _tag: "Waiting" }
    }
    return Option.exists(current.loadTargets[key], (target) =>
      sameSelection(target.selection, request.selection))
      ? { _tag: "Load", waitOnRetryableFailure: false }
      : { _tag: "Waiting" }
  }

  const awaitLoadDecision = (
    slotId: SlotId,
    request: PendingLoadRequest,
  ): Effect.Effect<Exclude<LoadDecision, { readonly _tag: "Waiting" }>> =>
    aggregate.changes.pipe(
      Stream.map((current) => decideLoadRequest(slotId, request, current)),
      Stream.filter((decision): decision is Exclude<
        LoadDecision,
        { readonly _tag: "Waiting" }
      > => decision._tag !== "Waiting"),
      Stream.runHead,
      Effect.flatMap(Option.match({
        onNone: () => Effect.die("Model-slot state ended before load request completed"),
        onSome: Effect.succeed,
      })),
    )

  const runLoadRequest = (
    slotId: SlotId,
    request: PendingLoadRequest,
  ): Effect.Effect<void> => Effect.suspend(() => Effect.gen(function* () {
    const decision = yield* Effect.raceFirst(
      Deferred.await(request.cancellation).pipe(
        Effect.as({ _tag: "Cancelled" } as const),
      ),
      awaitLoadDecision(slotId, request),
    )
    if (decision._tag === "Cancelled" || decision._tag === "Satisfied") {
      yield* transitionLoadRequest(slotId, request, Option.none())
      return
    }
    if (decision._tag === "Failed") {
      yield* transitionLoadRequest(slotId, request, Option.some({
        _tag: "Failed",
        selection: request.selection,
        failure: decision.failure,
      }))
      return
    }
    if (Option.isSome(yield* Deferred.poll(request.cancellation))) {
      yield* transitionLoadRequest(slotId, request, Option.none())
      return
    }

    const admission = yield* Effect.exit(admitModelLoad(slotId, request.selection))
    if (Exit.isFailure(admission)) {
      const typed = Cause.failureOption(admission.cause)
      const requestFailure = Option.match(typed, {
        onNone: () => modelFailure(
          "model_load_request_failed",
          Cause.pretty(admission.cause),
          false,
        ),
        onSome: loadRequestFailure,
      })
      if (decision.waitOnRetryableFailure && requestFailure.retryable) {
        const failedAtRevision = (yield* SubscriptionRef.get(aggregate)).snapshot.revision
        const resumed = yield* Effect.raceFirst(
          Deferred.await(request.cancellation).pipe(Effect.as("Cancelled" as const)),
          aggregate.changes.pipe(
            Stream.filter(({ snapshot }) => snapshot.revision > failedAtRevision),
            Stream.runHead,
            Effect.as("Changed" as const),
          ),
        )
        if (resumed === "Changed") return yield* runLoadRequest(slotId, request)
        yield* transitionLoadRequest(slotId, request, Option.none())
        return
      }
      yield* transitionLoadRequest(slotId, request, Option.some({
        _tag: "Failed",
        selection: request.selection,
        failure: requestFailure,
      }))
      return
    }

    const cancelled = Option.isSome(yield* Deferred.poll(request.cancellation))
    yield* transitionLoadRequest(slotId, request, Option.none())
    if (cancelled) {
      yield* stopModelInstance(admission.value).pipe(
        Effect.catchAll((error) => Effect.logWarning(
          "Cancelled model load request could not stop its admitted instance",
        ).pipe(Effect.annotateLogs({ slotId, error: error.message }))),
      )
    }
  })).pipe(
    Effect.catchAllCause((cause) => Cause.isInterruptedOnly(cause)
      ? Effect.void
      : transitionLoadRequest(
          slotId,
          request,
          Option.some({
            _tag: "Failed",
            selection: request.selection,
            failure: modelFailure("model_load_request_failed", Cause.pretty(cause), false),
          }),
        ).pipe(Effect.asVoid)),
  )

  const loadModel: ModelSlotControllerApi["loadModel"] = (slotId) =>
    Effect.uninterruptibleMask((restore) => Effect.gen(function* () {
      yield* restore(rebuild)
      yield* restore(commandLock.withPermits(1)(Effect.gen(function* () {
        const key = slotKey(slotId)
        const current = yield* SubscriptionRef.get(aggregate)
        const slot = current.snapshot.state.slots[key]
        if (slot._tag === "Unassigned") {
          return yield* reject(slotId, "Select a model before requesting a load")
        }
        if (slot._tag === "ConfiguredRemote") {
          return yield* reject(slotId, "The selected remote model does not require loading")
        }
        if (slot.residency._tag === "Requested"
          || slot.residency._tag === "Loading"
          || slot.residency._tag === "Ready") return

        const requests = yield* Ref.get(loadRequests)
        const existing = requests[key]
        if (Option.exists(existing, (request) =>
          request._tag === "Pending"
          && sameSelection(request.selection, slot.selection))) return
        const request: PendingLoadRequest = {
          _tag: "Pending",
          selection: slot.selection,
          cancellation: yield* Deferred.make<void>(),
        }
        yield* Effect.uninterruptible(Effect.gen(function* () {
          if (Option.isSome(existing) && existing.value._tag === "Pending") {
            yield* Deferred.succeed(existing.value.cancellation, undefined)
          }
          yield* Ref.set(loadRequests, {
            ...requests,
            [key]: Option.some(request),
          })
          yield* rebuild
          yield* Effect.forkIn(
            Effect.interruptible(runLoadRequest(slotId, request)),
            scope,
          )
        }))
      })))
    }))

  const stopModel: ModelSlotControllerApi["stopModel"] = (slotId) =>
    Effect.gen(function* () {
      const instanceId = yield* commandLock.withPermits(1)(Effect.gen(function* () {
        const slot = (yield* SubscriptionRef.get(aggregate)).snapshot.state.slots[slotKey(slotId)]
        if (slot._tag === "Unassigned") {
          return yield* reject(slotId, "Select a model before requesting a stop")
        }
        if (slot._tag === "ConfiguredRemote") {
          return yield* reject(slotId, "The selected remote model does not run locally")
        }
        switch (slot.residency._tag) {
          case "Requested":
            yield* clearLoadRequestUnsafe(slotId)
            yield* rebuild
            return Option.none<ModelInstanceId>()
          case "Loading":
          case "Ready":
            yield* clearLoadRequestUnsafe(slotId)
            yield* rebuild
            return Option.some(slot.residency.instanceId)
          case "Stopping":
            return Option.none<ModelInstanceId>()
          case "Unloaded":
          case "Failed":
            return yield* reject(slotId, "No selected model load is active")
        }
      }))
      if (Option.isSome(instanceId)) yield* stopModelInstance(instanceId.value)
    })

  const awaitLocalModelReady = (
    slotId: SlotId,
    providerModelId: ProviderModelId,
  ): Effect.Effect<ModelLoadResult, LocalInferenceError> => {
    const evaluate = (
      current: ControllerAggregate,
    ): Effect.Effect<Option.Option<ModelLoadResult>, LocalInferenceError> => Effect.gen(function* () {
      const slot = current.snapshot.state.slots[slotKey(slotId)]
      if (slot._tag !== "ConfiguredLocal"
        || slot.selection.providerId !== LOCAL_PROVIDER_ID
        || slot.selection.providerModelId !== providerModelId) {
        return yield* reject(slotId, "The local request no longer matches the selected slot")
      }
      switch (slot.residency._tag) {
        case "Ready":
          return Option.some({
            _tag: "Ready" as const,
            instanceId: slot.residency.instanceId,
            configurationId: slot.residency.configurationId,
          })
        case "Requested":
        case "Loading":
        case "Stopping":
          return Option.none()
        case "Failed":
          return yield* failure(
            slot.residency.failure.code,
            slot.residency.failure.message,
            slot.residency.failure.retryable,
          )
        case "Unloaded":
          return Option.some({
            _tag: "Cancelled" as const,
            reason: "user_stop",
          })
      }
    })

    return aggregate.changes.pipe(
      Stream.mapEffect(evaluate),
      Stream.filterMap((result) => result),
      Stream.runHead,
      Effect.flatMap(Option.match({
        onNone: () => failure(
          "model_slot_observation_ended",
          "Model-slot observation ended before the requested model became ready",
          false,
        ),
        onSome: Effect.succeed,
      })),
    )
  }

  const acquireLocalModel: ModelSlotControllerApi["acquireLocalModel"] = (
    slotId,
    providerModelId,
  ) => Effect.gen(function* () {
    yield* loadModel(slotId)
    const result = yield* awaitLocalModelReady(slotId, providerModelId)
    if (result._tag === "Ready") {
      yield* Effect.locallyScoped(CurrentModelInstance, Option.some({
        instanceId: result.instanceId,
        configurationId: result.configurationId,
      }))
    }
    return result
  })

  const updateModelSlot: ModelSlotControllerApi["updateModelSlot"] = (slotId, selection) =>
    Effect.gen(function* () {
      if (Option.isSome(selection)) {
        yield* requireSelectedLocalOffering(slotId, selection.value)
      }
      const normalized = Option.isSome(selection)
        ? Option.some(yield* normalizeAndValidateSelection(slotId, selection.value))
        : Option.none<SlotSelection>()
      const previous = yield* commandLock.withPermits(1)(Effect.gen(function* () {
        const previous = yield* selectedSlot(slotId)
        if (Option.isNone(normalized) && previous._tag === "Unassigned") return previous
        if (Option.isSome(normalized) && previous._tag !== "Unassigned"
          && sameSelection(previous.selection, normalized.value)) return previous
        yield* clearLoadRequestUnsafe(slotId)
        yield* modelSelection.updateSlot(slotId, normalized).pipe(
          Effect.mapError((error) => new ModelSlotMutationFailed({
            slotId,
            code: "model_slot_persistence_failed",
            message: String(error),
            retryable: true,
          })),
        )
        yield* rebuild
        return previous
      }))
      if (previous._tag === "ConfiguredLocal"
        && (previous.residency._tag === "Loading"
          || previous.residency._tag === "Ready")) {
        const configured = (yield* modelSelection.get).slots
        const stillSelected = [configured.primary, configured.secondary].some((candidate) =>
          Option.exists(candidate, (value) =>
            value.providerId === LOCAL_PROVIDER_ID
            && value.providerModelId === previous.selection.providerModelId))
        if (!stillSelected) {
          yield* Effect.forkIn(stopModelInstance(previous.residency.instanceId).pipe(
            Effect.catchAll((error) => Effect.logWarning("Follow-up local model stop failed").pipe(
              Effect.annotateLogs({ slotId, error: error.message }),
            )),
          ), scope)
        }
      }
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
    snapshot: SubscriptionRef.get(aggregate).pipe(Effect.map(({ snapshot }) => snapshot)),
    changes: aggregate.changes.pipe(
      Stream.map(({ snapshot }) => snapshot),
      Stream.changesWith((left, right) => left.revision === right.revision),
    ),
    agentModelConfiguration: SubscriptionRef.get(aggregate).pipe(
      Effect.map(({ agentConfiguration }) => agentConfiguration),
    ),
    agentModelConfigurationChanges: aggregate.changes.pipe(
      Stream.map(({ agentConfiguration }) => agentConfiguration),
      Stream.changesWith(sameConfigStateValue),
    ),
    acquireLocalModel,
    updateModelSlot,
    setModelFavorite,
    loadModel,
    previewModelLoad,
    stopModel,
  })
}))
  const slotKey = (slotId: SlotId): "primary" | "secondary" =>
    slotId === PRIMARY_SLOT_ID ? "primary" : "secondary"
