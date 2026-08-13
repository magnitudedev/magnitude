import { Atom, Result } from "@effect-atom/atom-react"
import { Deferred, Effect, Exit, Option, Stream } from "effect"
import { Mutation } from "@magnitudedev/effect-query"
import {
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ReasoningEffortSchema,
  type LocalModelInstallationAdmission,
  type LocalModelsState,
  type ModelInstanceId,
  type ModelServingConfigurationId,
  type ModelSlotsState,
  type ProviderModelId,
  type ReasoningEffort,
  type SlotSelection,
} from "@magnitudedev/sdk"
import type { AgentClientInstance } from "../state/agent-client"
import { onboardingAtoms } from "../onboarding/atoms"
import { modelSlotAtoms, sameSlotSelection } from "../model-slots/atoms"
import { localModelAtoms } from "./atoms"
import {
  findLocalModelByConfigurationId,
  localModelProviderModelId,
} from "./projection"
import {
  OnboardingModelChoiceRejected,
  OnboardingModelResourceChanged,
  OnboardingModelSetupAlreadyActive,
  OnboardingModelSetupCancellationUnavailable,
  OnboardingModelSetupFailed,
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

interface LoadingModel extends AssignedModel {
  readonly instanceId: ModelInstanceId
}

interface ResolvedChoice {
  readonly prepared: PreparedModel
  readonly installed: Option.Option<InstalledModel>
  readonly ready: Option.Option<LoadingModel>
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
  if (model.value.acquisitionState._tag === "Installed" && Option.isNone(providerModelId)) {
    return Effect.fail(new OnboardingModelChoiceRejected({
      configurationId,
      reason: "missing_provider_identity",
    }))
  }
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
    ? Option.map(providerModelId, (id) => ({ ...prepared, providerModelId: id }))
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
    return Option.flatMap(primary.instance, (instance) =>
      instance.configurationId === configurationId && instance.lifecycle._tag === "Ready"
        ? Option.some({
            ...exact,
            selection,
            instanceId: instance.id,
          })
        : Option.none())
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

const makeService = (client: AgentClientInstance) => {
  const localModels = localModelAtoms(client)
  const slots = modelSlotAtoms(client)
  const onboarding = onboardingAtoms(client)
  const execution = Atom.keepAlive(Atom.make<OnboardingModelSetupExecution>({ _tag: "Choosing" }))
  const active = Atom.keepAlive(Atom.make<Option.Option<ActiveInvocation>>(Option.none()))
  const sources = Atom.make((get) => {
    get.mount(localModels.mirrorInvalidationWatchAtom)
    get.mount(localModels.invalidationBridgeAtom)
    get.mount(slots.mirrorInvalidationWatchAtom)
    get.mount(slots.invalidationBridgeAtom)
  })

  const state = Atom.make((get) => {
    get.mount(sources)
    return Result.map(Result.all({
      models: get(localModels.localModelsQueryAtom).result,
      slots: get(slots.modelSlotsQueryAtom).result,
    }), ({ models, slots: response }) => projectOnboardingModelSetup(
      get(execution),
      models,
      response.state,
    ))
  })

  const start = Atom.keepAlive(client.effectQuery.runtime.fn<ModelServingConfigurationId>()(
    (configurationId, get) => Effect.gen(function* () {
      get.mount(sources)
      const cancellation = yield* Deferred.make<void>()
      const done = yield* Deferred.make<void, OnboardingModelSetupFailure>()
      const invocation = { cancellation, done }
      const claimed = yield* Effect.sync(() => {
        if (Option.isSome(get.registry.get(active))) return false
        get.registry.set(active, Option.some(invocation))
        return true
      })
      if (!claimed) return yield* new OnboardingModelSetupAlreadyActive()

      const setExecution = (next: OnboardingModelSetupExecution) => Effect.sync(() => {
        get.registry.set(execution, next)
      })
      const observeFailure = (cause: unknown) => new OnboardingModelSetupFailed({
        phase: "observe",
        cause,
      })
      const awaitInstalled = (
        prepared: PreparedModel,
        admission: LocalModelInstallationAdmission,
      ) => terminalFact(get.streamResult(localModels.localModelsResultAtom).pipe(
        Stream.mapError(observeFailure),
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
            const providerModelId = localModelProviderModelId(model.value)
            return Option.exists(providerModelId, (id) => id === admission.providerModelId)
              ? { _tag: "Ready", value: { ...prepared, providerModelId: admission.providerModelId } }
              : { _tag: "Failed", failure: new OnboardingModelResourceChanged({
                  configurationId: prepared.configurationId,
                  resource: "installation",
                }) }
          }
          if (admission._tag === "AlreadyInstalled"
            || acquisition._tag === "NotInstalled"
            || acquisition.downloadId !== admission.downloadId) {
            return { _tag: "Failed", failure: new OnboardingModelResourceChanged({
              configurationId: prepared.configurationId,
              resource: "installation",
            }) }
          }
          if (acquisition._tag === "Downloading") return { _tag: "Waiting" }
          return { _tag: "Failed", failure: new OnboardingModelSetupFailed({
            phase: "install",
            cause: acquisition._tag === "Failed" ? acquisition.failure : acquisition,
          }) }
        }),
      ))
      const ensureInstalled = (resolved: ResolvedChoice) => Option.match(resolved.installed, {
        onSome: Effect.succeed,
        onNone: () => Effect.uninterruptibleMask((restore) => Mutation.execute(
          localModels.installMutation,
          { configurationId: resolved.prepared.configurationId },
        ).pipe(
          Effect.mapError((cause) => new OnboardingModelSetupFailed({ phase: "install", cause })),
          Effect.flatMap((admission) => {
            const publish = admission._tag === "DownloadAdmitted"
              ? setExecution({
                  _tag: "Installing",
                  configurationId: resolved.prepared.configurationId,
                  cancelling: false,
                })
              : Effect.void
            const cancel = admission._tag === "DownloadAdmitted"
              ? Mutation.execute(localModels.cancelDownloadMutation, {
                  downloadId: admission.downloadId,
                }).pipe(
                  Effect.mapError((cause) => new OnboardingModelSetupFailed({ phase: "cancel", cause })),
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
          Effect.zipRight(Effect.uninterruptible(Mutation.execute(slots.assignMutation, {
            slotId: PRIMARY_SLOT_ID,
            selection,
          }))),
          Effect.mapError((cause) => new OnboardingModelSetupFailed({ phase: "assign", cause })),
          Effect.as({ ...installed, selection }),
        )
      }
      const awaitReady = (loading: LoadingModel) => terminalFact(
        get.streamResult(slots.modelSlotsResultAtom).pipe(
          Stream.mapError(observeFailure),
          Stream.map(({ state: current }): TerminalFact<LoadingModel> => {
            const slot = current.slots.primary
            if (slot._tag !== "ConfiguredLocal"
              || !sameSlotSelection(slot.selection, loading.selection)) {
              return { _tag: "Failed", failure: new OnboardingModelResourceChanged({
                configurationId: loading.configurationId,
                resource: "instance",
              }) }
            }
            const instance = Option.filter(slot.instance, (candidate) =>
              candidate.id === loading.instanceId
              && candidate.configurationId === loading.configurationId)
            if (Option.isNone(instance)) {
              return { _tag: "Failed", failure: new OnboardingModelResourceChanged({
                configurationId: loading.configurationId,
                resource: "instance",
              }) }
            }
            switch (instance.value.lifecycle._tag) {
              case "Loading":
              case "Stopping": return { _tag: "Waiting" }
              case "Ready": return { _tag: "Ready", value: loading }
              case "Failed": return { _tag: "Failed", failure: new OnboardingModelSetupFailed({
                phase: "load",
                cause: instance.value.lifecycle.failure,
              }) }
              case "Stopped": return { _tag: "Failed", failure: new OnboardingModelSetupFailed({
                phase: "load",
                cause: instance.value.lifecycle,
              }) }
            }
          }),
        ),
      )
      const load = (assigned: AssignedModel) => Effect.uninterruptibleMask((restore) =>
        Mutation.execute(slots.loadMutation, {
          slotId: PRIMARY_SLOT_ID,
          selection: assigned.selection,
        }).pipe(
          Effect.mapError((cause) => new OnboardingModelSetupFailed({ phase: "load", cause })),
          Effect.map(({ instanceId }) => ({ ...assigned, instanceId })),
          Effect.tap((loading) => setExecution({
            _tag: "Loading",
            configurationId: loading.configurationId,
            instanceId: loading.instanceId,
            cancelling: false,
          })),
          Effect.flatMap((loading) => restore(awaitReady(loading)).pipe(
            Effect.onInterrupt(() => Mutation.execute(slots.stopMutation, {
              instanceId: loading.instanceId,
            }).pipe(
              Effect.mapError((cause) => new OnboardingModelSetupFailed({ phase: "cancel", cause })),
              Effect.tapError((failure) => setExecution({
                _tag: "Failed",
                configurationId: loading.configurationId,
                failure,
              })),
              Effect.orDie,
            )),
          )),
        ))
      const complete = (ready: LoadingModel) => setExecution({
        _tag: "Completing",
        configurationId: ready.configurationId,
      }).pipe(
        Effect.zipRight(Effect.uninterruptible(Mutation.execute(
          onboarding.updateMutation,
          { completed: true },
        ))),
        Effect.mapError((cause) => new OnboardingModelSetupFailed({ phase: "complete", cause })),
      )
      const run = Effect.gen(function* () {
        yield* setExecution({ _tag: "Preparing", configurationId, cancelling: false })
        const models = yield* get.result(localModels.localModelsResultAtom).pipe(
          Effect.mapError(observeFailure),
        )
        const slotResponse = yield* get.result(slots.modelSlotsResultAtom).pipe(
          Effect.mapError(observeFailure),
        )
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
            if (Option.exists(get.registry.get(active), (current) => current === invocation)) {
              get.registry.set(active, Option.none())
            }
          })),
        )),
      )
      if (outcome === "Cancelled") yield* setExecution({ _tag: "Choosing" })
    }),
    { concurrent: true },
  ))

  const cancel = Atom.keepAlive(client.effectQuery.runtime.fn<void>()((_, get) => Effect.gen(function* () {
    const invocation = get.registry.get(active)
    if (Option.isNone(invocation)) return yield* new OnboardingModelSetupNotActive()
    const current = get.registry.get(execution)
    if (current._tag === "Completing") {
      return yield* new OnboardingModelSetupCancellationUnavailable()
    }
    if (current._tag !== "Choosing" && current._tag !== "Failed") {
      get.registry.set(execution, { ...current, cancelling: true })
    }
    yield* Deferred.succeed(invocation.value.cancellation, undefined)
    yield* Deferred.await(invocation.value.done)
  }), { concurrent: true }))

  const skip = Atom.keepAlive(client.effectQuery.runtime.fn<void>()((_, get) => Effect.gen(function* () {
    if (Option.isSome(get.registry.get(active))) {
      return yield* new OnboardingModelSetupAlreadyActive()
    }
    yield* Mutation.execute(onboarding.updateMutation, { completed: true }).pipe(
      Effect.mapError((cause) => new OnboardingModelSetupFailed({ phase: "complete", cause })),
    )
    get.registry.set(execution, { _tag: "Choosing" })
  }), { concurrent: true }))

  return {
    state,
    start,
    cancel,
    skip,
  }
}

export type OnboardingModelSetupService = ReturnType<typeof makeService>

const servicesByClient = new WeakMap<object, OnboardingModelSetupService>()

export const onboardingModelSetupService = (
  client: AgentClientInstance,
): OnboardingModelSetupService => {
  const existing = servicesByClient.get(client)
  if (existing !== undefined) return existing
  const service = makeService(client)
  servicesByClient.set(client, service)
  return service
}
