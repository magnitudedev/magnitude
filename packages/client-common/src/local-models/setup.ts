import { Atom, Registry, Result } from "@effect-atom/atom-react"
import { Context, Deferred, Effect, Exit, Layer, Option, Stream } from "effect"
import {
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ReasoningEffortSchema,
  type CatalogModelReconciliationAdmission,
  type LocalModelsState,
  type ModelServingConfigurationId,
  type ModelSlotsState,
  type ProviderModelId,
  type ReasoningEffort,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { OnboardingPersistence } from "../onboarding/persistence"
import { ModelSlots, sameSlotSelection } from "../model-slots/service"
import { LocalModels } from "./service"
import {
  findLocalModelByConfigurationId,
  localModelProviderModelId,
} from "./projection"
import {
  OnboardingModelChoiceRejected,
  OnboardingModelResourceChanged,
  OnboardingModelSetupAlreadyActive,
  OnboardingModelSetupCancellationUnavailable,
  OnboardingModelSetupNotActive,
  projectOnboardingModelSetup,
  type OnboardingModelSetupExecution,
  type OnboardingModelSetupFailure,
} from "./setup-state"

export * from "./setup-state"

interface PreparedModel {
  readonly configurationId: ModelServingConfigurationId
  readonly reasoningEffort: ReasoningEffort
}

interface InstalledModel extends PreparedModel {
  readonly providerModelId: ProviderModelId
}

interface AssignedModel extends InstalledModel {
  readonly selection: SlotSelection
}

interface ResolvedChoice {
  readonly prepared: PreparedModel
  readonly installed: Option.Option<InstalledModel>
  readonly ready: Option.Option<AssignedModel>
}

type TerminalFact<Value> =
  | { readonly _tag: "Waiting" }
  | { readonly _tag: "Ready"; readonly value: Value }
  | { readonly _tag: "Failed"; readonly failure: OnboardingModelSetupFailure }

interface ActiveInvocation {
  readonly cancellation: Deferred.Deferred<void>
  readonly done: Deferred.Deferred<void, OnboardingModelSetupFailure>
}

const resolveChoice = (
  configurationId: ModelServingConfigurationId,
  models: LocalModelsState,
  slots: ModelSlotsState,
): Effect.Effect<ResolvedChoice, OnboardingModelChoiceRejected> => {
  const model = findLocalModelByConfigurationId(models.models, configurationId)
  if (Option.isNone(model)) {
    return Effect.fail(new OnboardingModelChoiceRejected({ configurationId, reason: "missing" }))
  }
  const serving = model.value.servingState
  if (serving._tag !== "Assessed") {
    return Effect.fail(new OnboardingModelChoiceRejected({ configurationId, reason: "unresolved" }))
  }
  if (serving.assessment._tag !== "Fits") {
    return Effect.fail(new OnboardingModelChoiceRejected({ configurationId, reason: "ineligible" }))
  }
  const providerModelId = localModelProviderModelId(model.value)
  const primary = slots.slots.primary
  const reasoningEffort = primary._tag !== "Unassigned"
      && Option.contains(providerModelId, primary.selection.providerModelId)
    ? primary.selection.reasoningEffort
    : Option.getOrElse(
        serving.capabilities.reasoning.defaultEffort,
        () => ReasoningEffortSchema.make("none"),
      )
  const prepared = { configurationId, reasoningEffort }
  const installed = model.value.acquisitionState._tag === "Installed"
      && serving.availabilityState._tag === "Selectable"
    ? Option.some({
        ...prepared,
        providerModelId: serving.availabilityState.providerModelId,
      })
    : Option.none()
  const ready = Option.flatMap(installed, (exact) => {
    const selection: SlotSelection = {
      providerId: ProviderIdSchema.make("local"),
      providerModelId: exact.providerModelId,
      reasoningEffort: exact.reasoningEffort,
    }
    if (primary._tag !== "ConfiguredLocal" || !sameSlotSelection(primary.selection, selection)) {
      return Option.none()
    }
    return primary.residency._tag === "Ready"
      && primary.residency.configurationId === configurationId
      ? Option.some({ ...exact, selection })
      : Option.none()
  })
  return Effect.succeed({ prepared, installed, ready })
}

const terminalFact = <Value>(
  stream: Stream.Stream<TerminalFact<Value>, OnboardingModelSetupFailure>,
): Effect.Effect<Value, OnboardingModelSetupFailure> => stream.pipe(
  Stream.filter((fact) => fact._tag !== "Waiting"),
  Stream.runHead,
  Effect.flatMap(Option.match({
    onNone: () => Effect.die("The authoritative query ended before publishing a terminal fact"),
    onSome: (fact) => fact._tag === "Ready"
      ? Effect.succeed(fact.value)
      : Effect.fail(fact.failure),
  })),
)

const makeOnboardingModelSetup = Effect.gen(function* () {
  const localModels = yield* LocalModels
  const slots = yield* ModelSlots
  const onboarding = yield* OnboardingPersistence
  const registry = yield* Registry.AtomRegistry
  const execution = Atom.keepAlive(Atom.make<OnboardingModelSetupExecution>({ _tag: "Choosing" }))
  const active = Atom.keepAlive(Atom.make<Option.Option<ActiveInvocation>>(Option.none()))

  const state = Atom.make((get) => {
    return Result.map(Result.all({
      models: get(localModels.state),
      slots: get(slots.state),
    }), ({ models, slots: response }) => projectOnboardingModelSetup(
      get(execution),
      models,
      response.state,
    ))
  })

  const start = (configurationId: ModelServingConfigurationId) => Effect.gen(function* () {
      const cancellation = yield* Deferred.make<void>()
      const done = yield* Deferred.make<void, OnboardingModelSetupFailure>()
      const invocation = { cancellation, done }
      const claimed = yield* Effect.sync(() => {
        if (Option.isSome(registry.get(active))) return false
        registry.set(active, Option.some(invocation))
        return true
      })
      if (!claimed) return yield* new OnboardingModelSetupAlreadyActive()

      const setExecution = (next: OnboardingModelSetupExecution) => Effect.sync(() => {
        registry.set(execution, next)
      })
      const awaitInstalled = (
        prepared: PreparedModel,
        admission: CatalogModelReconciliationAdmission,
      ) => terminalFact(Registry.toStreamResult(registry, localModels.state).pipe(
        Stream.map((current): TerminalFact<InstalledModel> => {
          const model = findLocalModelByConfigurationId(current.models, prepared.configurationId)
          if (Option.isNone(model)) {
            return { _tag: "Failed", failure: new OnboardingModelResourceChanged({
              configurationId: prepared.configurationId,
              resource: "installation",
            }) }
          }
          const acquisition = model.value.acquisitionState
          if (acquisition._tag === "Installed") {
            const serving = model.value.servingState
            if (serving._tag !== "Assessed") return { _tag: "Waiting" }
            const availability = serving.availabilityState
            if (availability._tag === "Installable") return { _tag: "Waiting" }
            const providerModelId = availability._tag === "Unavailable"
              ? Option.getOrUndefined(availability.providerModelId)
              : availability.providerModelId
            if (providerModelId !== admission.providerModelId) {
              return providerModelId === undefined
                ? { _tag: "Waiting" }
                : { _tag: "Failed", failure: new OnboardingModelResourceChanged({
                  configurationId: prepared.configurationId,
                  resource: "installation",
                }) }
            }
            if (availability._tag === "Preparing") return { _tag: "Waiting" }
            if (availability._tag === "Unavailable") {
              return { _tag: "Failed", failure: availability.failure }
            }
            return {
              _tag: "Ready",
              value: { ...prepared, providerModelId: admission.providerModelId },
            }
          }
          if (admission._tag === "Current"
            || acquisition._tag === "NotInstalled"
            || acquisition.downloadId !== admission.downloadId) {
            return { _tag: "Failed", failure: new OnboardingModelResourceChanged({
              configurationId: prepared.configurationId,
              resource: "installation",
            }) }
          }
          if (acquisition._tag === "Downloading") return { _tag: "Waiting" }
          return acquisition._tag === "Failed"
            ? { _tag: "Failed", failure: acquisition.failure }
            : { _tag: "Failed", failure: new OnboardingModelResourceChanged({
                configurationId: prepared.configurationId,
                resource: "installation",
              }) }
        }),
      ))
      const ensureInstalled = (resolved: ResolvedChoice) => Option.match(resolved.installed, {
        onSome: Effect.succeed,
        onNone: () => Effect.uninterruptibleMask((restore) => localModels.install(
          resolved.prepared.configurationId,
        ).pipe(
          Effect.flatMap((admission) => {
            const publish = admission._tag === "DownloadAdmitted"
              ? setExecution({
                  _tag: "Installing",
                  configurationId: resolved.prepared.configurationId,
                  cancelling: false,
                })
              : Effect.void
            const cancel = admission._tag === "DownloadAdmitted"
              ? localModels.cancelDownload(admission.downloadId).pipe(
                  Effect.tapError((failure) => setExecution({
                    _tag: "Failed",
                    configurationId: resolved.prepared.configurationId,
                    failure,
                  })),
                  Effect.orDie,
                )
              : Effect.void
            return publish.pipe(
              Effect.zipRight(restore(awaitInstalled(resolved.prepared, admission))),
              Effect.onInterrupt(() => cancel),
            )
          }),
        )),
      })
      const assign = (installed: InstalledModel) => {
        const selection: SlotSelection = {
          providerId: ProviderIdSchema.make("local"),
          providerModelId: installed.providerModelId,
          reasoningEffort: installed.reasoningEffort,
        }
        return setExecution({
          _tag: "Configuring",
          configurationId: installed.configurationId,
          cancelling: false,
        }).pipe(
          Effect.zipRight(Effect.uninterruptible(slots.assign(PRIMARY_SLOT_ID, selection))),
          Effect.as({ ...installed, selection }),
        )
      }
      const awaitReady = (loading: AssignedModel) => terminalFact(
        Registry.toStreamResult(registry, slots.state).pipe(
          Stream.map(({ state: current }): TerminalFact<AssignedModel> => {
            const slot = current.slots.primary
            if (slot._tag !== "ConfiguredLocal"
              || !sameSlotSelection(slot.selection, loading.selection)) {
              return { _tag: "Failed", failure: new OnboardingModelResourceChanged({
                configurationId: loading.configurationId,
                resource: "instance",
              }) }
            }
            switch (slot.residency._tag) {
              case "Requested":
              case "Loading":
              case "Stopping": return { _tag: "Waiting" }
              case "Ready":
                return slot.residency.configurationId === loading.configurationId
                  ? { _tag: "Ready", value: loading }
                  : { _tag: "Failed", failure: new OnboardingModelResourceChanged({
                      configurationId: loading.configurationId,
                      resource: "instance",
                    }) }
              case "Failed": return { _tag: "Failed", failure: slot.residency.failure }
              case "Unloaded": return { _tag: "Failed", failure: new OnboardingModelResourceChanged({
                configurationId: loading.configurationId,
                resource: "instance",
              }) }
            }
          }),
        ),
      )
      const load = (assigned: AssignedModel) => Effect.uninterruptibleMask((restore) =>
        slots.load(PRIMARY_SLOT_ID).pipe(
          Effect.tap(() => setExecution({
            _tag: "Loading",
            configurationId: assigned.configurationId,
            cancelling: false,
          })),
          Effect.zipRight(restore(awaitReady(assigned)).pipe(
            Effect.onInterrupt(() => slots.stop(PRIMARY_SLOT_ID).pipe(
              Effect.tapError((failure) => setExecution({
                _tag: "Failed",
                configurationId: assigned.configurationId,
                failure,
              })),
              Effect.orDie,
            )),
          )),
        ))
      const complete = (ready: AssignedModel) => setExecution({
        _tag: "Completing",
        configurationId: ready.configurationId,
      }).pipe(
        Effect.zipRight(Effect.uninterruptible(onboarding.setCompleted(true))),
      )
      const run = Effect.gen(function* () {
        yield* setExecution({ _tag: "Preparing", configurationId, cancelling: false })
        const models = yield* Registry.getResult(registry, localModels.state)
        const slotResponse = yield* Registry.getResult(registry, slots.state)
        const resolved = yield* resolveChoice(configurationId, models, slotResponse.state)
        if (Option.isSome(resolved.ready)) return yield* complete(resolved.ready.value)
        const installed = yield* ensureInstalled(resolved)
        const assigned = yield* assign(installed)
        const ready = yield* load(assigned)
        yield* complete(ready)
      })

      const outcome = yield* Effect.raceFirst(
        run.pipe(Effect.as("Completed" as const)),
        Deferred.await(cancellation).pipe(Effect.as("Cancelled" as const)),
      ).pipe(
        Effect.tapError((failure) => setExecution({
          _tag: "Failed",
          configurationId,
          failure,
        })),
        Effect.onInterrupt(() => setExecution({ _tag: "Choosing" })),
        Effect.onExit((exit) => Deferred.done(done, Exit.asVoid(exit)).pipe(
          Effect.zipRight(Effect.sync(() => {
            if (Option.exists(registry.get(active), (current) => current === invocation)) {
              registry.set(active, Option.none())
            }
          })),
        )),
      )
      if (outcome === "Cancelled") yield* setExecution({ _tag: "Choosing" })
    })

  const cancel = Effect.gen(function* () {
    const invocation = registry.get(active)
    if (Option.isNone(invocation)) return yield* new OnboardingModelSetupNotActive()
    const current = registry.get(execution)
    if (current._tag === "Completing") {
      return yield* new OnboardingModelSetupCancellationUnavailable()
    }
    if (current._tag !== "Choosing" && current._tag !== "Failed") {
      registry.set(execution, { ...current, cancelling: true })
    }
    yield* Deferred.succeed(invocation.value.cancellation, undefined)
    yield* Deferred.await(invocation.value.done)
  })

  const skip = Effect.gen(function* () {
    if (Option.isSome(registry.get(active))) {
      return yield* new OnboardingModelSetupAlreadyActive()
    }
    yield* onboarding.setCompleted(true)
    registry.set(execution, { _tag: "Choosing" })
  })

  return {
    state,
    start,
    cancel,
    skip,
  }
})

export interface OnboardingModelSetup extends Effect.Effect.Success<typeof makeOnboardingModelSetup> {}

export const OnboardingModelSetup = Context.GenericTag<OnboardingModelSetup>(
  "client/OnboardingModelSetup",
)

export const OnboardingModelSetupLive = Layer.effect(
  OnboardingModelSetup,
  makeOnboardingModelSetup,
)
