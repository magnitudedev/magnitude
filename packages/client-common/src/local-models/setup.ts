import { Atom, Registry, Result } from "@effect-atom/atom-react"
import { Cause, Context, Data, Deferred, Effect, Exit, Layer, Option, Schema, Scope, Stream } from "effect"
import {
  PRIMARY_SLOT_ID,
  ModelInstanceFailureSchema,
  ProviderIdSchema,
  ReasoningEffortSchema,
  installedAcquisition,
  type CatalogModelReconciliationAdmission,
  type LocalModelsState,
  type ModelInstanceFailure,
  type ModelSlotsState,
  type ProviderModelId,
  type ReasoningEffort,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { OnboardingPersistence } from "../onboarding/persistence"
import { ModelSlots, sameSlotSelection } from "../model-slots/service"
import { localModelOptions, type LocalModelOption } from "./options"
import { LocalModels } from "./service"
import {
  HarnessConnection,
  type HarnessDestination,
  type HarnessId,
  type HarnessLaunchPlan,
} from "../harness-connections/service"
import {
  OnboardingModelChoiceRejected,
  OnboardingModelResourceChanged,
  OnboardingModelSetupAlreadyActive,
  OnboardingModelSetupCancellationUnavailable,
  OnboardingModelSetupNotActive,
  OnboardingModelSetupNotOpen,
  defaultOnboardingModelRankingControls,
  normalizeOnboardingModelRankingControls,
  projectOnboardingModelSetupContent,
  tagOnboardingModelSetupObservation,
  type OnboardingModelSetupExecution,
  type OnboardingModelSetupAttempt,
  type OnboardingModelSetupFailure,
  type OnboardingModelSetupNotice,
  type OnboardingModelRankingControls,
  type OnboardingModelSetupState,
} from "./setup-state"

export * from "./setup-state"

interface PreparedModel {
  readonly modelId: ProviderModelId
  readonly reasoningEffort: ReasoningEffort
  readonly option: LocalModelOption
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

const isModelInstanceFailure = (
  failure: OnboardingModelSetupFailure,
): failure is ModelInstanceFailure => Schema.is(ModelInstanceFailureSchema)(failure)
  // ModelFailure is intentionally untagged, so exclude tagged mutation
  // errors that are structurally compatible with its three fields.
  && (!("_tag" in failure) || failure._tag === "LowMemory")

class OnboardingModelSelectionCancelled extends Data.TaggedError(
  "OnboardingModelSelectionCancelled",
)<{}> {}

interface SelectionInvocation {
  readonly cancellation: Deferred.Deferred<void>
  readonly done: Deferred.Deferred<void, OnboardingModelSetupFailure>
}

type ClosingInvocation = object

type OnboardingModelSetupLifecycle =
  | {
      readonly _tag: "Resting"
      readonly retainedOpen: boolean
      readonly notice: Option.Option<OnboardingModelSetupNotice>
    }
  | {
      readonly _tag: "Selecting"
      readonly invocation: SelectionInvocation
      readonly execution: OnboardingModelSetupExecution
    }
  | {
      readonly _tag: "LoadFailed"
      readonly attempt: Extract<OnboardingModelSetupAttempt, { readonly _tag: "LoadFailed" }>
    }
  | {
      readonly _tag: "Closing"
      readonly invocation: ClosingInvocation
    }
  | {
      readonly _tag: "ChoosingHarness"
      readonly selected: AssignedModel
      readonly destinations: ReadonlyArray<HarnessDestination>
      readonly completeOnFinish: boolean
    }
  | {
      readonly _tag: "ApplyingHarness"
      readonly selected: AssignedModel
      readonly harness: HarnessId
      readonly completeOnFinish: boolean
    }
  | {
      readonly _tag: "HarnessHandoff"
      readonly plan: HarnessLaunchPlan
    }

export interface OnboardingModelSetupConfig {
  readonly initiallyOpen: boolean
}

export const OnboardingModelSetupConfig = Context.GenericTag<OnboardingModelSetupConfig>(
  "client/OnboardingModelSetupConfig",
)

const resolveChoice = (
  modelId: ProviderModelId,
  models: LocalModelsState,
  slots: ModelSlotsState,
): Effect.Effect<ResolvedChoice, OnboardingModelChoiceRejected> => {
  const option = localModelOptions(models, slots).find((candidate) => candidate.model.modelId === modelId)
  if (option === undefined) {
    return Effect.fail(new OnboardingModelChoiceRejected({ modelId, reason: "missing" }))
  }
  const serving = option.model.servingState
  if (serving._tag !== "Assessed") {
    return Effect.fail(new OnboardingModelChoiceRejected({ modelId, reason: "unresolved" }))
  }
  if (serving.assessment._tag !== "Fits") {
    return Effect.fail(new OnboardingModelChoiceRejected({ modelId, reason: "ineligible" }))
  }
  const primary = slots.slots.primary
  const providerModelId = serving.availabilityState._tag === "Selectable"
    ? Option.some(serving.availabilityState.providerModelId)
    : Option.none()
  const reasoningEffort = primary._tag !== "Unassigned"
      && Option.contains(providerModelId, primary.selection.providerModelId)
    ? primary.selection.reasoningEffort
    : Option.getOrElse(
        serving.capabilities.reasoning.defaultEffort,
        () => ReasoningEffortSchema.make("none"),
      )
  const prepared = { modelId, reasoningEffort, option }
  const installed = option.model.acquisitionState._tag === "Installed"
      && Option.isSome(providerModelId)
    ? Option.some({ ...prepared, providerModelId: providerModelId.value })
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
      ? Option.some({ ...exact, selection })
      : Option.none()
  })
  return Effect.succeed({ prepared, installed, ready })
}

const awaitTerminalFact = <Snapshot, QueryError, Value>(
  registry: Registry.Registry,
  atom: Atom.Atom<Result.Result<Snapshot, QueryError>>,
  project: (snapshot: Snapshot) => TerminalFact<Value>,
): Effect.Effect<Value, OnboardingModelSetupFailure> => Registry.toStream(registry, atom).pipe(
  Stream.mapEffect((result): Effect.Effect<TerminalFact<Value>> => {
    if (Result.isInitial(result)) return Effect.succeed({ _tag: "Waiting" })
    if (Result.isFailure(result)) {
      const defects = Cause.keepDefects(result.cause)
      return Option.match(defects, {
        onNone: () => Effect.succeed<TerminalFact<Value>>({ _tag: "Waiting" }),
        onSome: Effect.failCause,
      })
    }
    return Effect.succeed(project(result.value))
  }),
  Stream.filter((fact) => fact._tag !== "Waiting"),
  Stream.runHead,
  Effect.flatMap(Option.match({
    onNone: () => Effect.die("The setup observation ended before publishing a terminal fact"),
    onSome: (fact) => fact._tag === "Ready"
      ? Effect.succeed(fact.value)
      : Effect.fail(fact.failure),
  })),
)

const makeOnboardingModelSetup = Effect.gen(function* () {
  const localModels = yield* LocalModels
  const slots = yield* ModelSlots
  const onboarding = yield* OnboardingPersistence
  const harnessConnection = yield* HarnessConnection
  const config = yield* OnboardingModelSetupConfig
  const registry = yield* Registry.AtomRegistry
  const scope = yield* Scope.Scope
  const admissionLock = yield* Effect.makeSemaphore(1)
  const closedLifecycle: OnboardingModelSetupLifecycle = {
    _tag: "Resting",
    retainedOpen: false,
    notice: Option.none(),
  }
  const lifecycle = Atom.keepAlive(Atom.make<OnboardingModelSetupLifecycle>({
    ...closedLifecycle,
    retainedOpen: config.initiallyOpen,
  }))
  const rankingControls = Atom.keepAlive(Atom.make<OnboardingModelRankingControls>(
    defaultOnboardingModelRankingControls,
  ))

  const view = Atom.make((get) => {
    const current = get(lifecycle)
    const onboardingResult = tagOnboardingModelSetupObservation(
      get(onboarding.state),
      "onboarding",
    )
    return Result.flatMap(onboardingResult, ({ completed }) => {
      const exitKind = completed ? "Close" as const : "Skip" as const
      if (current._tag === "Closing") {
        return Result.success<OnboardingModelSetupState>({
          _tag: "Open",
          exitKind,
          notice: Option.none(),
          content: { _tag: "Closing" },
        })
      }
      if (current._tag === "HarnessHandoff") {
        return Result.success<OnboardingModelSetupState>({
          _tag: "Open",
          exitKind: "Close",
          notice: Option.none(),
          content: { _tag: "HarnessHandoff", plan: current.plan },
        })
      }
      if (current._tag === "ChoosingHarness" || current._tag === "ApplyingHarness") {
        const selected = current.selected
        return Result.success<OnboardingModelSetupState>({
          _tag: "Open",
          exitKind,
          notice: Option.none(),
          content: current._tag === "ChoosingHarness"
            ? {
                _tag: "Harness",
                model: selected.option.model,
                modelId: selected.modelId,
                providerModelId: selected.providerModelId,
                destinations: current.destinations,
              }
            : {
                _tag: "ApplyingHarness",
                model: selected.option.model,
                modelId: selected.modelId,
                harness: current.harness,
              },
        })
      }
      if (current._tag === "Resting" && completed && !current.retainedOpen) {
        return Result.success<OnboardingModelSetupState>({ _tag: "Closed" })
      }
      const models = tagOnboardingModelSetupObservation(
        get(localModels.state),
        "local-models",
      )
      const slotResponse = tagOnboardingModelSetupObservation(
        get(slots.state),
        "model-slots",
      )
      return Result.flatMap(models, (modelState) => Result.map(
        slotResponse,
        (slotState) => ({
          _tag: "Open" as const,
          exitKind,
          notice: current._tag === "Resting" ? current.notice : Option.none(),
          content: projectOnboardingModelSetupContent(
            current._tag === "Selecting"
              ? Option.some(current.execution)
              : current._tag === "LoadFailed"
                ? Option.some(current.attempt)
                : Option.none(),
            modelState,
            slotState,
            get(rankingControls),
          ),
        }),
      ))
    })
  })

  const retry = Effect.suspend(() => {
    const retries: Effect.Effect<void>[] = []
    if (Result.isFailure(registry.get(onboarding.state))) retries.push(onboarding.retry)
    const current = registry.get(lifecycle)
    const onboardingValue = Result.value(registry.get(onboarding.state))
    const observesModels = current._tag === "Selecting" || current._tag === "LoadFailed"
      || current._tag === "Resting"
        && (!Option.exists(onboardingValue, ({ completed }) => completed) || current.retainedOpen)
    if (observesModels) {
      if (Result.isFailure(registry.get(localModels.state))) retries.push(localModels.retry)
      if (Result.isFailure(registry.get(slots.state))) retries.push(slots.retry)
    }
    return Effect.all(retries, { discard: true })
  })

  const setRankingControls = (next: OnboardingModelRankingControls) => Effect.sync(() => {
    registry.set(rankingControls, normalizeOnboardingModelRankingControls(next))
  })

  const setExecution = (
    invocation: SelectionInvocation,
    execution: OnboardingModelSetupExecution,
  ) => Effect.sync(() => {
    const current = registry.get(lifecycle)
    if (current._tag === "Selecting" && current.invocation === invocation) {
      registry.set(lifecycle, { ...current, execution })
    }
  })

  const awaitInstalled = (
    prepared: PreparedModel,
    admission: CatalogModelReconciliationAdmission,
  ) => {
    const project = (current: LocalModelsState): TerminalFact<InstalledModel> => {
      const model = current.models.find((candidate) => candidate.modelId === prepared.modelId)
      if (model === undefined) {
        return { _tag: "Failed", failure: new OnboardingModelResourceChanged({
          modelId: prepared.modelId,
          resource: "installation",
        }) }
      }
      const acquisition = model.acquisitionState
      if (installedAcquisition(acquisition) !== undefined) {
        const serving = model.servingState
        if (serving._tag !== "Assessed") return { _tag: "Waiting" }
        const availability = serving.availabilityState
        if (availability._tag === "Installable" || availability._tag === "Preparing") {
          return { _tag: "Waiting" }
        }
        const providerModelId = availability._tag === "Unavailable"
          ? Option.getOrUndefined(availability.providerModelId)
          : availability.providerModelId
        if (providerModelId !== admission.providerModelId) {
          return providerModelId === undefined
            ? { _tag: "Waiting" }
            : { _tag: "Failed", failure: new OnboardingModelResourceChanged({
                modelId: prepared.modelId,
                resource: "installation",
              }) }
        }
        if (availability._tag === "Unavailable") {
          return { _tag: "Failed", failure: availability.failure }
        }
        return {
          _tag: "Ready",
          value: { ...prepared, providerModelId: admission.providerModelId },
        }
      }
      if (admission._tag === "DownloadAdmitted" && acquisition._tag === "Installing") {
        return { _tag: "Waiting" }
      }
      if (acquisition._tag === "InstallFailed") {
        return { _tag: "Failed", failure: acquisition.failure }
      }
      return { _tag: "Failed", failure: new OnboardingModelResourceChanged({
        modelId: prepared.modelId,
        resource: "installation",
      }) }
    }
    return awaitTerminalFact(registry, localModels.state, project)
  }

  const cancelled = (invocation: SelectionInvocation) => Deferred.poll(
    invocation.cancellation,
  ).pipe(Effect.map(Option.isSome))

  const ensureInstalled = (
    invocation: SelectionInvocation,
    resolved: ResolvedChoice,
  ): Effect.Effect<InstalledModel, OnboardingModelSetupFailure | OnboardingModelSelectionCancelled> =>
    Option.match(resolved.installed, {
      onSome: (installed) => cancelled(invocation).pipe(
        Effect.flatMap((isCancelled) => isCancelled
          ? Effect.fail(new OnboardingModelSelectionCancelled())
          : Effect.succeed(installed)),
      ),
      onNone: () => localModels.install(resolved.prepared.modelId).pipe(
        Effect.flatMap((admission) => {
          const publish = admission._tag === "DownloadAdmitted"
            ? setExecution(invocation, {
                _tag: "Installing",
                option: resolved.prepared.option,
                modelId: resolved.prepared.modelId,
                cancelling: false,
              })
            : Effect.void
          const cancel = admission._tag === "DownloadAdmitted"
            ? localModels.cancelDownload(resolved.prepared.modelId)
            : Effect.void
          return publish.pipe(
            Effect.zipRight(cancelled(invocation)),
            Effect.flatMap((isCancelled) => isCancelled
              ? cancel.pipe(Effect.zipRight(Effect.fail(new OnboardingModelSelectionCancelled())))
              : admission._tag === "DownloadAdmitted"
                ? Effect.raceFirst(
                    awaitInstalled(resolved.prepared, admission).pipe(
                      Effect.map((installed) => ({ _tag: "Installed" as const, installed })),
                    ),
                    Deferred.await(invocation.cancellation).pipe(
                      Effect.as({ _tag: "Cancelled" as const }),
                    ),
                  ).pipe(Effect.flatMap((outcome) => outcome._tag === "Installed"
                    ? Effect.succeed(outcome.installed)
                    : cancel.pipe(Effect.zipRight(Effect.fail(new OnboardingModelSelectionCancelled())))))
                : awaitInstalled(resolved.prepared, admission)),
          )
        }),
      ),
    })

  const assign = (
    invocation: SelectionInvocation,
    installed: InstalledModel,
  ): Effect.Effect<AssignedModel, OnboardingModelSetupFailure | OnboardingModelSelectionCancelled> => {
    const selection: SlotSelection = {
      providerId: ProviderIdSchema.make("local"),
      providerModelId: installed.providerModelId,
      reasoningEffort: installed.reasoningEffort,
    }
    return setExecution(invocation, {
      _tag: "Configuring",
      option: installed.option,
      modelId: installed.modelId,
      cancelling: false,
    }).pipe(
      Effect.zipRight(slots.assign(PRIMARY_SLOT_ID, selection)),
      Effect.zipRight(cancelled(invocation)),
      Effect.flatMap((isCancelled) => isCancelled
        ? Effect.fail(new OnboardingModelSelectionCancelled())
        : Effect.succeed({ ...installed, selection })),
    )
  }

  const awaitReady = (
    assigned: AssignedModel,
    options: { readonly waitForLoadAdmission?: boolean } = {},
  ) => {
    const project = (state: ModelSlotsState): TerminalFact<AssignedModel> => {
        const slot = state.slots.primary
        if (slot._tag !== "ConfiguredLocal" || !sameSlotSelection(slot.selection, assigned.selection)) {
          return { _tag: "Failed", failure: new OnboardingModelResourceChanged({
            modelId: assigned.modelId,
            resource: "instance",
          }) }
        }
        switch (slot.residency._tag) {
          case "Requested":
          case "Loading":
          case "Stopping": return { _tag: "Waiting" }
          case "Ready": return { _tag: "Ready", value: assigned }
          case "Failed": return { _tag: "Failed", failure: slot.residency.failure }
          case "Unloaded": return options.waitForLoadAdmission
            ? { _tag: "Waiting" }
            : { _tag: "Failed", failure: new OnboardingModelResourceChanged({
                modelId: assigned.modelId,
                resource: "instance",
              }) }
        }
    }
    return awaitTerminalFact(registry, slots.state, project)
  }

  const load = (
    invocation: SelectionInvocation,
    assigned: AssignedModel,
  ): Effect.Effect<AssignedModel, OnboardingModelSetupFailure | OnboardingModelSelectionCancelled> =>
    setExecution(invocation, {
      _tag: "Loading",
      option: assigned.option,
      modelId: assigned.modelId,
      providerModelId: assigned.providerModelId,
      selection: assigned.selection,
      cancelling: false,
    }).pipe(
      Effect.zipRight(Effect.raceFirst(
        Effect.matchEffect(slots.load(PRIMARY_SLOT_ID), {
          onFailure: (commandFailure) => commandFailure._tag === "ModelSlotMutationRejected"
            ? Effect.fail(commandFailure)
            : awaitReady(assigned, {
                waitForLoadAdmission: true,
              }).pipe(
                Effect.timeoutFail({
                  duration: "2 seconds",
                  onTimeout: () => commandFailure,
                }),
              ),
          onSuccess: () => awaitReady(assigned),
        }),
        Deferred.await(invocation.cancellation).pipe(
          Effect.zipRight(Effect.fail(new OnboardingModelSelectionCancelled())),
        ),
      )),
    )

  const terminalizeSelection = (
    invocation: SelectionInvocation,
    exit: Exit.Exit<"Completed" | "Cancelled", OnboardingModelSetupFailure>,
  ) => admissionLock.withPermits(1)(Effect.gen(function* () {
    const current = registry.get(lifecycle)
    if (current._tag !== "Selecting" || current.invocation !== invocation) return exit
    const effectiveExit = Exit.isSuccess(exit)
        && exit.value === "Completed"
        && current.execution._tag !== "Completing"
        && Option.isSome(yield* Deferred.poll(invocation.cancellation))
      ? Exit.succeed("Cancelled" as const)
      : exit
    if (Exit.isSuccess(effectiveExit)) {
      registry.set(lifecycle, {
        _tag: "Resting",
        retainedOpen: effectiveExit.value === "Cancelled",
        notice: Option.none(),
      })
    } else {
      const failure = Option.getOrUndefined(Cause.failureOption(effectiveExit.cause))
      if (failure !== undefined
        && current.execution._tag === "Loading"
        && isModelInstanceFailure(failure)) {
        registry.set(lifecycle, {
          _tag: "LoadFailed",
          attempt: {
            _tag: "LoadFailed",
            execution: current.execution,
            failure,
          },
        })
      } else {
        const notice = Option.map(
          Cause.failureOption(effectiveExit.cause),
          (selectionFailure): OnboardingModelSetupNotice => ({
            failure: selectionFailure,
            subject: {
              _tag: "ModelOperation",
              operation: current.execution._tag,
              model: current.execution.option.model,
            },
          }),
        )
        registry.set(lifecycle, {
          _tag: "Resting",
          retainedOpen: true,
          notice,
        })
      }
    }
    return effectiveExit
  })).pipe(
    Effect.flatMap((effectiveExit) => Deferred.done(invocation.done, Exit.asVoid(effectiveExit))),
  )

  const runSelection = (
    invocation: SelectionInvocation,
    resolved: ResolvedChoice,
    completeOnFinish: boolean,
  ) => Effect.gen(function* () {
    const ready = Option.match(resolved.ready, {
      onSome: Effect.succeed,
      onNone: () => ensureInstalled(invocation, resolved).pipe(
        Effect.flatMap((installed) => assign(invocation, installed)),
        Effect.flatMap((assigned) => load(invocation, assigned)),
      ),
    })
    const selected = yield* ready
    const destinations = yield* harnessConnection.list
    yield* admissionLock.withPermits(1)(Effect.sync(() => {
      const current = registry.get(lifecycle)
      if (current._tag === "Selecting" && current.invocation === invocation) {
        registry.set(lifecycle, { _tag: "ChoosingHarness", selected, destinations, completeOnFinish })
      }
    }))
    return "Completed" as const
  }).pipe(
    Effect.catchTag("OnboardingModelSelectionCancelled", () =>
      Effect.succeed("Cancelled" as const)),
    Effect.exit,
    Effect.flatMap((exit) => terminalizeSelection(invocation, exit).pipe(
      Effect.zipRight(Exit.isFailure(exit) && Option.isNone(Cause.failureOption(exit.cause))
        ? Effect.failCause(exit.cause)
        : Effect.void),
    )),
  )

  const open = admissionLock.withPermits(1)(Effect.sync(() => {
    const current = registry.get(lifecycle)
    if (current._tag !== "Resting" || current.retainedOpen) return
    registry.set(lifecycle, { ...current, retainedOpen: true })
  }))

  const select = (modelId: ProviderModelId) =>
    admissionLock.withPermits(1)(Effect.uninterruptibleMask((restore) => Effect.gen(function* () {
      const current = registry.get(lifecycle)
      if (current._tag !== "Resting"
        && (current._tag !== "LoadFailed" || current.attempt.execution.modelId !== modelId)) {
        return yield* new OnboardingModelSetupAlreadyActive()
      }
      const onboardingState = yield* restore(Registry.getResult(registry, onboarding.state))
      if (onboardingState.completed && current._tag === "Resting" && !current.retainedOpen) {
        return yield* new OnboardingModelSetupNotOpen()
      }
      const models = yield* restore(Registry.getResult(registry, localModels.state))
      const slotResponse = yield* restore(Registry.getResult(registry, slots.state))
      const resolved = yield* resolveChoice(modelId, models, slotResponse)
      const invocation: SelectionInvocation = {
        cancellation: yield* Deferred.make<void>(),
        done: yield* Deferred.make<void, OnboardingModelSetupFailure>(),
      }
      registry.set(lifecycle, {
        _tag: "Selecting",
        invocation,
        execution: {
          _tag: "Preparing",
          option: resolved.prepared.option,
          modelId,
          cancelling: false,
        },
      })
      yield* Effect.forkIn(restore(runSelection(
        invocation,
        resolved,
        !onboardingState.completed,
      )), scope)
    })))

  const cancel = admissionLock.withPermits(1)(Effect.gen(function* () {
    const current = registry.get(lifecycle)
    if (current._tag === "Closing" || current._tag === "ApplyingHarness" || current._tag === "HarnessHandoff") {
      return yield* new OnboardingModelSetupCancellationUnavailable()
    }
    if (current._tag !== "Selecting") return yield* new OnboardingModelSetupNotActive()
    if (current.execution._tag === "Completing") {
      return yield* new OnboardingModelSetupCancellationUnavailable()
    }
    registry.set(lifecycle, {
      ...current,
      execution: { ...current.execution, cancelling: true },
    })
    yield* Deferred.succeed(current.invocation.cancellation, undefined)
    return current.invocation.done
  })).pipe(Effect.flatMap(Deferred.await))

  const chooseAnother = admissionLock.withPermits(1)(Effect.sync(() => {
    const current = registry.get(lifecycle)
    if (current._tag !== "LoadFailed") return false
    registry.set(lifecycle, {
      _tag: "Resting",
      retainedOpen: true,
      notice: Option.none(),
    })
    return true
  })).pipe(
    Effect.filterOrFail(
      (dismissed) => dismissed,
      () => new OnboardingModelSetupNotActive(),
    ),
    Effect.asVoid,
  )

  const back = admissionLock.withPermits(1)(Effect.gen(function* () {
    const current = registry.get(lifecycle)
    if (current._tag !== "ChoosingHarness") return yield* new OnboardingModelSetupNotActive()
    registry.set(lifecycle, {
      _tag: "Resting",
      retainedOpen: true,
      notice: Option.none(),
    })
  }))

  const continueWithHarness = (
    harness: HarnessId,
    options: { readonly launchOnStartup: boolean; readonly installSkill: boolean },
  ) => admissionLock.withPermits(1)(Effect.gen(function* () {
    const current = registry.get(lifecycle)
    if (current._tag !== "ChoosingHarness") return yield* new OnboardingModelSetupNotActive()
    if (!current.destinations.some((destination) => destination.id === harness && destination.selectable)) {
      return yield* new OnboardingModelSetupNotActive()
    }
    registry.set(lifecycle, { _tag: "ApplyingHarness", selected: current.selected, harness, completeOnFinish: current.completeOnFinish })
    return { selected: current.selected, completeOnFinish: current.completeOnFinish }
  })).pipe(Effect.flatMap(({ selected, completeOnFinish }) => Effect.gen(function* () {
    if (options.launchOnStartup) yield* harnessConnection.installStartup
    if (options.installSkill) yield* harnessConnection.installSkill(harness)
    const connection = yield* harnessConnection.connect(harness, { setCurrent: Option.some(selected.modelId) })
    const plan = Option.getOrThrow(connection.launchPlan)
    if (completeOnFinish) yield* onboarding.complete
    yield* admissionLock.withPermits(1)(Effect.sync(() => {
      registry.set(lifecycle, harness === "magnitude"
        ? closedLifecycle
        : { _tag: "HarnessHandoff", plan })
    }))
  }).pipe(Effect.catchAll((error) => admissionLock.withPermits(1)(Effect.sync(() => {
    registry.set(lifecycle, {
      _tag: "Resting",
      retainedOpen: true,
      notice: Option.some<OnboardingModelSetupNotice>({
        failure: error,
        subject: { _tag: "Setup" },
      }),
    })
  })).pipe(Effect.zipRight(Effect.fail(error)))))))

  const terminalizeClosing = (
    invocation: ClosingInvocation,
    exit: Exit.Exit<void, OnboardingModelSetupFailure>,
  ) => admissionLock.withPermits(1)(Effect.sync(() => {
    const current = registry.get(lifecycle)
    if (current._tag !== "Closing" || current.invocation !== invocation) return
    registry.set(lifecycle, Exit.isSuccess(exit)
      ? closedLifecycle
      : {
          _tag: "Resting",
          retainedOpen: true,
          notice: Option.map(
            Cause.failureOption(exit.cause),
            (failure): OnboardingModelSetupNotice => ({
              failure,
              subject: { _tag: "Setup" },
            }),
          ),
        })
  })).pipe(
    Effect.zipRight(Exit.isFailure(exit) && Option.isNone(Cause.failureOption(exit.cause))
      ? Effect.failCause(exit.cause)
      : Effect.void),
  )

  const exit = admissionLock.withPermits(1)(Effect.uninterruptibleMask((restore) =>
    Effect.gen(function* () {
      const current = registry.get(lifecycle)
      if (current._tag !== "Resting") return yield* new OnboardingModelSetupAlreadyActive()
      const onboardingState = yield* restore(Registry.getResult(registry, onboarding.state))
      if (onboardingState.completed) {
        if (!current.retainedOpen) return yield* new OnboardingModelSetupNotOpen()
        registry.set(lifecycle, closedLifecycle)
        return
      }
      const invocation: ClosingInvocation = {}
      registry.set(lifecycle, { _tag: "Closing", invocation })
      yield* Effect.forkIn(
        restore(Effect.exit(onboarding.complete.pipe(Effect.asVoid)).pipe(
          Effect.flatMap((exit) => terminalizeClosing(invocation, exit)),
        )),
        scope,
      )
    }),
  ))

  return {
    view,
    retry,
    setRankingControls,
    open,
    select,
    cancel,
    chooseAnother,
    back,
    continueWithHarness,
    exit,
  }
})

export interface OnboardingModelSetup extends Effect.Effect.Success<typeof makeOnboardingModelSetup> {}

export const OnboardingModelSetup = Context.GenericTag<OnboardingModelSetup>(
  "client/OnboardingModelSetup",
)

export const OnboardingModelSetupLive = Layer.scoped(
  OnboardingModelSetup,
  makeOnboardingModelSetup,
)
