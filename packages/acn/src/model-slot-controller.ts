import {
  Context,
  Deferred,
  Effect,
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
  ModelInstanceIdSchema,
  ModelPreferenceMutationFailed,
  ModelSlotConfiguredLocal,
  ModelSlotLifecycle,
  ModelSlotMutationFailed,
  ModelSlotMutationRejected,
  ModelSlotUnassigned,
  ModelSlotsMirror,
  ModelServingConfigurationSchema,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  type LocalInferenceError,
  type LocalProviderOffering,
  type MirroredSnapshot,
  type ModelFailure,
  type ModelInstanceId,
  type ModelSlot,
  type ModelSlotAvailability,
  type ModelSlotDescriptor,
  type ModelSlotsState,
  type ModelSlotUpdateError,
  type ModelServingConfigurationId,
  type ProviderCatalogEntry,
  type ProviderModelCatalogEntry,
  type ProviderModelCatalogState,
  type ProviderModelIdentity,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/protocol"
import {
  CurrentModelInstance,
  IcnClient,
  IcnInstances,
  type Generated,
} from "@magnitudedev/icn"
import {
  ReasoningEffortSchema,
  type ProviderId,
  type ProviderModelId,
} from "@magnitudedev/sdk"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import { ModelConfiguration } from "./model-configuration"
import { MirroredStateChanges } from "./mirrored-state"
import { LocalModelPackages } from "./local-model-packages"
import { LocalModelRecommendations } from "./local-model-recommendations"
import { LocalProviderOfferings } from "./local-provider-offerings"
import { ProviderModelCatalog } from "./provider-model-catalog"
import { LocalInferenceHardware } from "./local-inference-hardware"
import { offeringTargetToIcn, servingProfileToIcn } from "./local-model-icn-adapter"
import { modelOfferingTargetPackageIds } from "@magnitudedev/protocol"
import {
  localModelAvailability,
  modelSlotActions,
  projectModelInstance,
  projectModelLoadPreview,
  selectableModelCapabilities,
} from "./model-slot-projection"

export interface ModelSlotControllerApi {
  readonly snapshot: Effect.Effect<MirroredSnapshot<ModelSlotsState>>
  readonly changes: Stream.Stream<MirroredSnapshot<ModelSlotsState>>
  readonly agentModelConfiguration: Effect.Effect<ConfigState>
  readonly agentModelConfigurations: Stream.Stream<ConfigState>
  readonly acquireLocalModel: (
    slotId: SlotId,
    providerModelId: ProviderModelId,
  ) => Effect.Effect<void, LocalInferenceError, Scope.Scope>
  readonly ensureLocalModelReady: (
    slotId: SlotId,
    providerModelId: ProviderModelId,
  ) => Effect.Effect<void, LocalInferenceError>
  readonly updateModelSlot: (
    slotId: SlotId,
    selection: Option.Option<SlotSelection>,
  ) => Effect.Effect<void, ModelSlotUpdateError>
  readonly setModelFavorite: (
    model: ProviderModelIdentity,
    favorite: boolean,
  ) => Effect.Effect<void, ModelPreferenceMutationFailed>
  readonly loadModel: (slotId: SlotId) => Effect.Effect<void, LocalInferenceError>
  readonly stopModel: (instanceId: ModelInstanceId) => Effect.Effect<void, LocalInferenceError>
}

export class ModelSlotController extends Context.Tag("ModelSlotController")<
  ModelSlotController,
  ModelSlotControllerApi
>() {}

interface ControllerAggregate {
  readonly snapshot: MirroredSnapshot<ModelSlotsState>
  readonly agentConfiguration: ConfigState
  readonly loadTargets: Readonly<Record<"primary" | "secondary", Option.Option<SlotLoadTarget>>>
}

interface SlotLoadTarget {
  readonly selection: SlotSelection
  readonly configuration: LocalProviderOffering["configuration"]
}

interface ModelLoadCommand {
  readonly requestedBy: SlotId
  readonly configuration: LocalProviderOffering["configuration"]
  readonly instanceId: ModelInstanceId
  readonly result: Deferred.Deferred<void, LocalInferenceError>
}

type ModelLoadClaim =
  | { readonly _tag: "Complete" }
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
  error: unknown,
  retryable = true,
): LocalModelMutationFailed => new LocalModelMutationFailed({
  code,
  message: error instanceof Error ? error.message : String(error),
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
  ModelConfiguration | LocalModelPackages | LocalModelRecommendations | LocalProviderOfferings
    | ProviderModelCatalog | MirroredStateChanges | IcnClient | IcnInstances
    | LocalInferenceHardware
> = Layer.scoped(ModelSlotController, Effect.gen(function* () {
  const configuration = yield* ModelConfiguration
  const localPackages = yield* LocalModelPackages
  const recommendations = yield* LocalModelRecommendations
  const localOfferings = yield* LocalProviderOfferings
  const catalog = yield* ProviderModelCatalog
  const mirroredChanges = yield* MirroredStateChanges
  const client = yield* IcnClient
  const observedInstances = yield* IcnInstances
  const hardware = yield* LocalInferenceHardware
  const scope = yield* Scope.Scope
  const stateLock = yield* Effect.makeSemaphore(1)
  const commandLock = yield* Effect.makeSemaphore(1)
  const loadCommands = yield* Ref.make<
    ReadonlyMap<ModelServingConfigurationId, ModelLoadCommand>
  >(new Map())
  const previewLock = yield* Effect.makeSemaphore(1)

  const initialConfiguration = yield* configuration.get
  const initialCatalog = (yield* catalog.snapshot).state
  const emptyState: ModelSlotsState = {
    slots: {
      primary: new ModelSlotUnassigned({ slotId: PRIMARY_SLOT_ID }),
      secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
    },
    recentModelIds: initialConfiguration.localModelRecency,
    favoriteModels: initialConfiguration.favoriteModels,
  }
  const aggregate = yield* SubscriptionRef.make<ControllerAggregate>({
    snapshot: { revision: 0, state: emptyState },
    agentConfiguration: buildConfigStateFromSlots(
      catalogContents(initialCatalog).models,
      emptyState.slots,
      initialConfiguration.contextLimits,
      0,
    ),
    loadTargets: {
      primary: Option.none(),
      secondary: Option.none(),
    },
  })

  const commit = (
    state: ModelSlotsState,
    catalogModels: readonly ProviderModelCatalogEntry[],
    contextLimits: typeof initialConfiguration.contextLimits,
    loadTargets: ControllerAggregate["loadTargets"],
  ) => Effect.gen(function* () {
    const previous = yield* SubscriptionRef.get(aggregate)
    const stateChanged = !Schema.equivalence(
      ModelSlotsMirror.stateSchema,
    )(previous.snapshot.state, state)
    const candidateAgentConfiguration = buildConfigStateFromSlots(
      catalogModels,
      state.slots,
      contextLimits,
      previous.agentConfiguration.revision,
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
    if (!stateChanged && !agentConfigurationChanged && !loadTargetsChanged) {
      return previous
    }
    const revision = stateChanged
      ? previous.snapshot.revision + 1
      : previous.snapshot.revision
    const agentRevision = agentConfigurationChanged
      ? previous.agentConfiguration.revision + 1
      : previous.agentConfiguration.revision
    const next: ControllerAggregate = {
      snapshot: stateChanged ? { revision, state } : previous.snapshot,
      agentConfiguration: agentConfigurationChanged
        ? { ...candidateAgentConfiguration, revision: agentRevision }
        : previous.agentConfiguration,
      loadTargets: loadTargetsChanged ? loadTargets : previous.loadTargets,
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
    const contents = catalogContents(catalogState)
    const providerFailure = contents.failures.find((item) =>
      item._tag === "ProviderFailure" && item.providerId === selection.providerId)
    if (providerFailure) {
      return unavailable("provider_unavailable", providerFailure.message)
    }
    const provider = contents.providers.find((item) => item.providerId === selection.providerId)
    if (!provider) return unavailable("provider_unavailable", "The selected provider is unavailable")
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
    if (!model || model.availability._tag !== "Available") {
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
    }
  }

  const rebuild = stateLock.withPermits(1)(Effect.gen(function* () {
    const configured = yield* configuration.get
    const catalogState = (yield* catalog.snapshot).state
    const contents = catalogContents(catalogState)
    const packages = yield* localPackages.installedPackageIds
    const native = yield* observedInstances.get
    const previous = (yield* SubscriptionRef.get(aggregate)).snapshot.state
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
            && modelOfferingTargetPackageIds(offering.configuration.target)
              .every((packageId) => packages.has(packageId))
          const availability = localModelAvailability(
            { _tag: "Available" },
            offering !== undefined,
            downloaded,
          )
          const previousSlot = current
          const samePrevious = previousSlot._tag === "ConfiguredLocal"
            && sameSelection(previousSlot.selection, selected)
          const configurationId = offering?.configuration.id
          const previousInstanceId = samePrevious
            ? Option.map(previousSlot.instance, (instance) => instance.id)
            : Option.none<ModelInstanceId>()
          const exact = Option.flatMap(previousInstanceId, (id) =>
            Option.fromNullable(native.instances.find((instance) =>
              instance.id === id && instance.configurationId === configurationId)))
          const recoverable = native.instances.findLast((instance) =>
            instance.configurationId === configurationId
            && (instance.lifecycle._tag === "Loading"
              || instance.lifecycle._tag === "Ready"
              || instance.lifecycle._tag === "Stopping"))
          const instance = Option.map(
            Option.orElse(exact, () => Option.fromNullable(recoverable)),
            projectModelInstance,
          )
          const readiness = samePrevious
            ? previousSlot.readiness
            : { _tag: "Assessing" as const }
          const props = {
            slotId,
            selection: selected,
            descriptor,
            availability,
            readiness,
            instance,
            actions: modelSlotActions(availability, readiness, instance),
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
      recentModelIds: configured.localModelRecency,
      favoriteModels: configured.favoriteModels,
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
    return yield* commit(
      state,
      contents.models,
      configured.contextLimits,
      {
        primary: loadTargetFor(configured.slots.primary),
        secondary: loadTargetFor(configured.slots.secondary),
      },
    )
  }))

  const updateReadiness = (
    slotId: SlotId,
    target: SlotLoadTarget,
    readiness: ModelSlotConfiguredLocal["readiness"],
  ) => stateLock.withPermits(1)(Effect.gen(function* () {
    const current = yield* SubscriptionRef.get(aggregate)
    const key = slotKey(slotId)
    const slot = current.snapshot.state.slots[key]
    if (slot._tag !== "ConfiguredLocal"
      || !sameSelection(slot.selection, target.selection)
      || !Option.exists(current.loadTargets[key], (currentTarget) =>
        sameSelection(currentTarget.selection, target.selection)
        && sameServingConfiguration(currentTarget.configuration, target.configuration))) {
      return
    }
    const nextSlot = ModelSlotLifecycle.hold(slot, {
      readiness,
      actions: modelSlotActions(slot.availability, readiness, slot.instance),
    })
    const configured = yield* configuration.get
    const models = catalogContents((yield* catalog.snapshot).state).models
    yield* commit({
      ...current.snapshot.state,
      slots: { ...current.snapshot.state.slots, [key]: nextSlot },
    }, models, configured.contextLimits, current.loadTargets)
  }))

  const assessReadinessFor = (
    slotId: SlotId,
    onlyIfAssessing: boolean,
  ) => previewLock.withPermits(1)(
    Effect.gen(function* () {
      const current = yield* SubscriptionRef.get(aggregate)
      const key = slotKey(slotId)
      const slot = current.snapshot.state.slots[key]
      const target = current.loadTargets[key]
      if (slot._tag !== "ConfiguredLocal"
        || slot.availability._tag !== "Available"
        || Option.isNone(target)
        || !sameSelection(slot.selection, target.value.selection)) {
        return
      }
      if (onlyIfAssessing && slot.readiness._tag !== "Assessing") return
      yield* updateReadiness(slotId, target.value, { _tag: "Assessing" })
      const readiness = yield* Effect.gen(function* () {
        const encodedTarget = yield* offeringTargetToIcn(target.value.configuration.target)
        const allocation = yield* client.models.previewModelLoad({
          payload: {
            configuration: {
              id: target.value.configuration.id,
              target: encodedTarget,
              profile: servingProfileToIcn(target.value.configuration.profile),
            },
          },
        })
        return {
          _tag: "Loadable" as const,
          allocation: projectModelLoadPreview(allocation),
        }
      }).pipe(
        Effect.catchAllCause((cause) => Effect.succeed({
          _tag: "Unavailable" as const,
          failure: modelFailure("model_load_unavailable", String(cause), true),
        })),
      )
      yield* updateReadiness(slotId, target.value, readiness)
    }),
  )

  const refreshReadinessFor = (slotId: SlotId) =>
    assessReadinessFor(slotId, false)

  const completeReadinessAssessment = (slotId: SlotId) =>
    assessReadinessFor(slotId, true)

  const refreshReadiness = Effect.forEach(
    [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID],
    refreshReadinessFor,
    { concurrency: 2, discard: true },
  )
  const reconcile = rebuild.pipe(Effect.zipRight(refreshReadiness))

  yield* rebuild
  yield* Effect.forkIn(refreshReadiness, scope)
  yield* Effect.forkIn(configuration.changes.pipe(
    Stream.drop(1),
    Stream.runForEach(() => reconcile),
  ), scope)
  yield* Effect.forkIn(localPackages.changes.pipe(Stream.runForEach(() => reconcile)), scope)
  yield* Effect.forkIn(localOfferings.changes.pipe(Stream.runForEach(() => reconcile)), scope)
  yield* Effect.forkIn(catalog.changes.pipe(Stream.drop(1), Stream.runForEach(() => reconcile)), scope)
  yield* Effect.forkIn(observedInstances.changes.pipe(
    Stream.drop(1),
    Stream.runForEach(() => rebuild),
  ), scope)
  yield* Effect.forkIn(hardware.changes.pipe(
    Stream.drop(1),
    Stream.runForEach(() => refreshReadiness),
  ), scope)

  const selectedSlot = (slotId: SlotId) => SubscriptionRef.get(aggregate).pipe(
    Effect.map(({ snapshot }) => snapshot.state.slots[slotKey(slotId)]),
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
    const installed = Option.isSome(offering)
      ? yield* localPackages.installedPackageIds
      : new Set<string>()
    const capabilities = selectableModelCapabilities(
      slotId,
      model,
      Option.getOrUndefined(Option.map(offering, (value) => ({
        capabilities: value.capabilities,
        packageIds: modelOfferingTargetPackageIds(value.configuration.target),
      }))),
      installed,
    )
    if (!capabilities) {
      return yield* reject(slotId, "The selected model is unavailable for this slot")
    }
    return normalizeSelectionReasoning(selection, { capabilities })
  })

  const retainSelectedLocalOffering = (
    slotId: SlotId,
    selection: SlotSelection,
  ): Effect.Effect<void, ModelSlotUpdateError> => {
    if (selection.providerId !== LOCAL_PROVIDER_ID) return Effect.void
    return Effect.gen(function* () {
      const existing = (yield* localOfferings.list).find((offering) =>
        offering.providerModelId === selection.providerModelId)
      if (existing) return
      const entry = yield* recommendations.getCatalogByProviderModelId(
        selection.providerModelId,
      )
      if (!entry) {
        return yield* reject(slotId, "The selected local model configuration is unavailable")
      }
      yield* localOfferings.save(
        entry.candidate.targetId,
        entry.configuration,
        Option.match(entry.recommendation, {
          onNone: () => ({ _tag: "UserConfigured" as const }),
          onSome: ({ id: recommendationId }) => ({
            _tag: "Recommendation" as const,
            recommendationId,
          }),
        }),
      )
    }).pipe(
      Effect.mapError((error) => error instanceof ModelSlotMutationRejected
        ? error
        : new ModelSlotMutationFailed({
            slotId,
            code: "local_model_offering_persistence_failed",
            message: error.message,
            retryable: error.retryable,
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
    return yield* observedInstances.changes.pipe(
      Stream.filterMap((snapshot) => Option.fromNullable(
        snapshot.instances.find((item) => item.id === instanceId && predicate(item)),
      )),
      Stream.runHead,
      Effect.flatMap(Option.match({
        onNone: () => Effect.die("ICN model-instance observation ended"),
        onSome: Effect.succeed,
      })),
    )
  })

  const awaitReadyInstance = (
    instanceId: ModelInstanceId,
  ): Effect.Effect<void, LocalInferenceError> => awaitInstance(
    instanceId,
    (instance) => instance.lifecycle._tag === "Ready"
      || instance.lifecycle._tag === "Stopped"
      || instance.lifecycle._tag === "Failed",
  ).pipe(
    Effect.flatMap((instance) => {
      switch (instance.lifecycle._tag) {
        case "Ready":
          return Effect.void
        case "Failed":
          return Effect.fail(new LocalModelMutationFailed({
            code: instance.lifecycle.failure.code,
            message: instance.lifecycle.failure.message,
            retryable: instance.lifecycle.failure.retryable,
          }))
        case "Stopped":
          return Effect.fail(failure(
            "model_instance_stopped_before_ready",
            "The model stopped before it became ready",
          ))
        case "Loading":
        case "Stopping":
          return Effect.die("Terminal model-instance observation returned a nonterminal state")
      }
    }),
  )

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
    const target = yield* offeringTargetToIcn(command.configuration.target).pipe(
      Effect.mapError((error) => failure("encode_local_model_target_failed", error, false)),
    )
    const response = yield* client.models.loadModelInstance({
      payload: {
        instanceId: command.instanceId,
        configuration: {
          id: command.configuration.id,
          target,
          profile: servingProfileToIcn(command.configuration.profile),
        },
      },
    }).pipe(Effect.mapError((error) => failure("load_model_instance_failed", error)))
    yield* Effect.forkIn(response.events.pipe(
      Stream.runDrain,
      Effect.catchAll((error) => Effect.logWarning("ICN load observation stream ended").pipe(
        Effect.annotateLogs({ instanceId: command.instanceId, error: String(error) }),
      )),
    ), scope)
    yield* awaitInstance(command.instanceId, () => true).pipe(
      Effect.mapError((error) => failure("observe_admitted_model_instance_failed", error)),
    )
    const remainsCurrent = yield* commandLock.withPermits(1)(
      commandIsCurrent(command),
    )
    if (!remainsCurrent) {
      yield* stopModel(command.instanceId).pipe(
        Effect.catchAll((error) => Effect.logWarning("Superseded model instance stop failed").pipe(
          Effect.annotateLogs({ instanceId: command.instanceId, error: error.message }),
        )),
      )
      return yield* reject(command.requestedBy, "The selected local model changed while loading")
    }
    yield* rebuild
    yield* awaitReadyInstance(command.instanceId)
    yield* rebuild
  })

  const loadModel: ModelSlotControllerApi["loadModel"] = (slotId) => Effect.gen(function* () {
    yield* rebuild
    yield* refreshReadinessFor(slotId)
    const candidateResult = yield* Deferred.make<void, LocalInferenceError>()
    const claim: ModelLoadClaim = yield* commandLock.withPermits(1)(Effect.gen(function* () {
      const current = yield* SubscriptionRef.get(aggregate)
      const key = slotKey(slotId)
      const slot = current.snapshot.state.slots[key]
      if (slot._tag !== "ConfiguredLocal") {
        return yield* reject(slotId, "The slot does not contain a local model")
      }
      const loadTarget = current.loadTargets[key]
      if (Option.isNone(loadTarget)
        || !sameSelection(loadTarget.value.selection, slot.selection)) {
        return yield* reject(slotId, "The selected local model configuration is unavailable")
      }
      if (Option.exists(slot.instance, (instance) =>
        instance.lifecycle._tag === "Ready")) {
        return { _tag: "Complete" as const }
      }
      if (Option.exists(slot.instance, (instance) =>
        instance.lifecycle._tag === "Loading")) {
        return {
          _tag: "Observe" as const,
          instanceId: Option.getOrThrow(slot.instance).id,
        }
      }
      if (slot.availability._tag === "Unavailable") {
        return yield* new LocalModelMutationFailed(slot.availability.failure)
      }
      if (slot.readiness._tag === "Unavailable") {
        return yield* new LocalModelMutationFailed(slot.readiness.failure)
      }
      if (slot.readiness._tag !== "Loadable") {
        return yield* reject(slotId, "The selected local model is still being assessed")
      }
      const candidate: ModelLoadCommand = {
        requestedBy: slotId,
        configuration: loadTarget.value.configuration,
        instanceId: ModelInstanceIdSchema.make(crypto.randomUUID()),
        result: candidateResult,
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
        return { _tag: "Join" as const, command: existing }
      }
      yield* Ref.set(
        loadCommands,
        new Map(currentCommands).set(candidate.configuration.id, candidate),
      )
      return { _tag: "Owner" as const, command: candidate }
    }))
    if (claim._tag === "Complete") return
    if (claim._tag === "Observe") {
      yield* awaitReadyInstance(claim.instanceId)
      yield* rebuild
      return
    }
    if (claim._tag === "Owner") {
      yield* Effect.forkIn(Effect.gen(function* () {
        const result = yield* Effect.exit(runLoadCommand(claim.command))
        yield* Deferred.done(claim.command.result, result)
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
    return yield* Deferred.await(claim.command.result)
  })

  const stopModel: ModelSlotControllerApi["stopModel"] = (instanceId) =>
    client.models.stopModelInstance({
      path: { instance_id: instanceId },
    }).pipe(
      Effect.mapError((error) => failure("stop_model_instance_failed", error)),
      Effect.asVoid,
    )

  const ensureLocalModelReady = (
    slotId: SlotId,
    providerModelId: ProviderModelId,
  ): Effect.Effect<{
    readonly instanceId: ModelInstanceId
    readonly configurationId: ModelServingConfigurationId
  }, LocalInferenceError> =>
    Effect.suspend(() => Effect.gen(function* () {
      yield* rebuild
      const current = yield* SubscriptionRef.get(aggregate)
      const slot = current.snapshot.state.slots[slotKey(slotId)]
      if (slot._tag !== "ConfiguredLocal"
        || slot.selection.providerModelId !== providerModelId) {
        return yield* reject(slotId, "The local request no longer matches the selected slot")
      }

      if (Option.isSome(slot.instance)) {
        const instance = slot.instance.value
        switch (instance.lifecycle._tag) {
          case "Ready":
            return {
              instanceId: instance.id,
              configurationId: instance.configurationId,
            }
          case "Loading": {
            const terminal = yield* awaitInstance(
              instance.id,
              (observed) => observed.lifecycle._tag === "Ready"
                || observed.lifecycle._tag === "Stopped"
                || observed.lifecycle._tag === "Failed",
            )
            if (terminal.lifecycle._tag === "Failed") {
              return yield* failure(
                terminal.lifecycle.failure.code,
                terminal.lifecycle.failure.message,
                terminal.lifecycle.failure.retryable,
              )
            }
            if (terminal.lifecycle._tag === "Stopped") {
              return yield* failure(
                "model_instance_stopped",
                "The model instance stopped before becoming ready",
                false,
              )
            }
            return yield* ensureLocalModelReady(slotId, providerModelId)
          }
          case "Stopping":
            yield* awaitInstance(
              instance.id,
              (observed) => observed.lifecycle._tag === "Stopped"
                || observed.lifecycle._tag === "Failed",
            )
            return yield* ensureLocalModelReady(slotId, providerModelId)
          case "Failed":
            if (!instance.lifecycle.failure.retryable) {
              return yield* failure(
                instance.lifecycle.failure.code,
                instance.lifecycle.failure.message,
                false,
              )
            }
            break
          case "Stopped":
            break
        }
      }

      if (slot.readiness._tag === "Loadable") {
        yield* loadModel(slotId)
        return yield* ensureLocalModelReady(slotId, providerModelId)
      }
      if (slot.availability._tag === "Unavailable") {
        return yield* failure(
          slot.availability.failure.code,
          slot.availability.failure.message,
          slot.availability.failure.retryable,
        )
      }
      if (slot.readiness._tag === "Unavailable") {
        return yield* failure(
          slot.readiness.failure.code,
          slot.readiness.failure.message,
          slot.readiness.failure.retryable,
        )
      }
      if (slot.readiness._tag !== "Assessing") {
        return yield* reject(slotId, "The selected local model is not loadable")
      }
      yield* completeReadinessAssessment(slotId)
      return yield* ensureLocalModelReady(slotId, providerModelId)
    }))

  const acquireLocalModel: ModelSlotControllerApi["acquireLocalModel"] = (
    slotId,
    providerModelId,
  ) => Effect.gen(function* () {
    const binding = yield* ensureLocalModelReady(slotId, providerModelId)
    yield* Effect.locallyScoped(CurrentModelInstance, Option.some(binding))
  })

  const updateModelSlot: ModelSlotControllerApi["updateModelSlot"] = (slotId, selection) =>
    Effect.gen(function* () {
      if (Option.isSome(selection)) {
        yield* retainSelectedLocalOffering(slotId, selection.value)
      }
      const normalized = Option.isSome(selection)
        ? Option.some(yield* normalizeAndValidateSelection(slotId, selection.value))
        : Option.none<SlotSelection>()
      const admittedReadiness = Option.isSome(normalized)
        && normalized.value.providerId === LOCAL_PROVIDER_ID
        ? Option.some(yield* Effect.gen(function* () {
            const offering = yield* localOfferings.resolve(normalized.value.providerModelId).pipe(
              Effect.mapError(() => reject(
                slotId,
                "The selected local model configuration is unavailable",
              )),
            )
            const encodedTarget = yield* offeringTargetToIcn(offering.configuration.target).pipe(
              Effect.mapError((error) => new ModelSlotMutationFailed({
                slotId,
                code: "encode_local_model_target_failed",
                message: String(error),
                retryable: false,
              })),
            )
            const allocation = yield* client.models.previewModelLoad({
              payload: {
                configuration: {
                  id: offering.configuration.id,
                  target: encodedTarget,
                  profile: servingProfileToIcn(offering.configuration.profile),
                },
              },
            }).pipe(
              Effect.mapError((error) => new ModelSlotMutationRejected({
                slotId,
                message: `The selected local model is not loadable: ${String(error)}`,
              })),
            )
            return {
              target: {
                selection: normalized.value,
                configuration: offering.configuration,
              },
              readiness: {
                _tag: "Loadable" as const,
                allocation: projectModelLoadPreview(allocation),
              },
            }
          }))
        : Option.none()
      const previous = yield* commandLock.withPermits(1)(Effect.gen(function* () {
        const previous = yield* selectedSlot(slotId)
        if (Option.isNone(normalized) && previous._tag === "Unassigned") return previous
        if (Option.isSome(normalized) && previous._tag !== "Unassigned"
          && sameSelection(previous.selection, normalized.value)) return previous
        yield* configuration.updateSlot(slotId, normalized).pipe(
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
      if (Option.isSome(admittedReadiness)) {
        yield* updateReadiness(
          slotId,
          admittedReadiness.value.target,
          admittedReadiness.value.readiness,
        )
      }
      if (previous._tag === "ConfiguredLocal" && Option.isSome(previous.instance)) {
        const configured = (yield* configuration.get).slots
        const stillSelected = [configured.primary, configured.secondary].some((candidate) =>
          Option.exists(candidate, (value) =>
            value.providerId === LOCAL_PROVIDER_ID
            && value.providerModelId === previous.selection.providerModelId))
        if (!stillSelected) {
          yield* stopModel(previous.instance.value.id).pipe(
            Effect.catchAll((error) => Effect.logWarning("Follow-up local model stop failed").pipe(
              Effect.annotateLogs({ slotId, error: error.message }),
            )),
          )
        }
      }
    })

  const setModelFavorite: ModelSlotControllerApi["setModelFavorite"] = (model, favorite) =>
    configuration.setFavorite(model, favorite).pipe(
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
    agentModelConfigurations: aggregate.changes.pipe(
      Stream.map(({ agentConfiguration }) => agentConfiguration),
      Stream.changesWith((left, right) => left.revision === right.revision),
    ),
    acquireLocalModel,
    ensureLocalModelReady: (slotId, providerModelId) =>
      ensureLocalModelReady(slotId, providerModelId).pipe(Effect.asVoid),
    updateModelSlot,
    setModelFavorite,
    loadModel,
    stopModel,
  })
}))
  const slotKey = (slotId: SlotId): "primary" | "secondary" =>
    slotId === PRIMARY_SLOT_ID ? "primary" : "secondary"
