import { useCallback, useMemo } from "react"
import { Atom, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Effect, Option, Schema, Stream } from "effect"
import {
  LocalInferenceHardwareMirror,
  LocalModelsMirror,
  ModelSlotsMirror,
  OnboardingMirror,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelCatalogMirror,
  type ProviderModelId,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { useMirroredStateAtom } from "./use-mirrored-state"
import { useModelSlotActions } from "./use-local-inference-state"
import {
  OnboardingIdle,
  OnboardingModelMachine,
  initialObservationCorrelation,
  onboardingCancellationPending,
  onboardingProviderModelId,
  onboardingSubmission,
  observeAdmittedDownload,
  observeAdmittedLoad,
  reduceDownloadObservation,
  reduceLoadObservation,
  requestOnboardingCancellation,
  resetOnboardingOperation,
  sameDownloadAttempts,
  type OnboardingConfigurationChoice,
  type OnboardingLoadModelChoice,
  type OnboardingModelOperation,
  type OnboardingModelSubmission,
} from "./onboarding-model-machine"

export type {
  OnboardingConfigurationChoice,
  OnboardingLoadModelChoice,
  OnboardingModelSubmission,
} from "./onboarding-model-machine"

export class OnboardingModelCommandFailed extends Schema.TaggedError<
  OnboardingModelCommandFailed
>()("OnboardingModelCommandFailed", {
  command: Schema.Literal(
    "install",
    "assign",
    "load",
    "complete",
    "cancel",
    "clear",
  ),
  message: Schema.String,
}) {}

export const useOnboardingModelSetup = () => {
  const client = useAgentClient()
  const hardwareAtom = useMirroredStateAtom(LocalInferenceHardwareMirror)
  const modelsAtom = useMirroredStateAtom(LocalModelsMirror)
  const catalogAtom = useMirroredStateAtom(ProviderModelCatalogMirror)
  const slotsAtom = useMirroredStateAtom(ModelSlotsMirror)
  const slotActions = useModelSlotActions()
  const mutations = useMemo(() => ({
    install: client.mutation("InstallModel"),
    assign: client.mutation("AssignSlot"),
    load: client.mutation("LoadModel"),
    complete: client.mutation("UpdateOnboardingState"),
    cancel: client.mutation("CancelModelDownload"),
    clear: client.mutation("ClearSlot"),
    stop: client.mutation("StopModel"),
  }), [client])
  const operationAtom = useMemo(
    () => Atom.make<OnboardingModelOperation>(new OnboardingIdle()),
    [],
  )

  const cancelAtom = useMemo(
    () => Atom.fn<"Cancel">()((_, get) => {
      const initial = get(operationAtom)
      if ((initial._tag === "DownloadAdmitted" || initial._tag === "DownloadCancellationFailed")
        && initial.submission._tag === "ConfigureThenLoad") {
        const currentModels = get(modelsAtom)
        const terminal = Result.isSuccess(currentModels) && !currentModels.waiting
          ? reduceDownloadObservation(
            initialObservationCorrelation,
            currentModels.value.state,
            initial.configurationId,
            initial.attemptIds,
          )[1]
          : Option.none()
        if (Option.exists(terminal, (observation) => observation !== "Superseded")) {
          get.set(operationAtom, OnboardingModelMachine.transition(initial, "Idle", {}))
          return Effect.void
        }
      }
      if (initial._tag === "LoadAdmitted" || initial._tag === "LoadStopFailed") {
        const currentSlots = get(slotsAtom)
        const terminal = Result.isSuccess(currentSlots) && !currentSlots.waiting
          ? reduceLoadObservation(
            initialObservationCorrelation,
            currentSlots.value.state,
            initial.providerModelId,
            initial.instanceId,
          )[1]
          : Option.none()
        if (Option.exists(terminal, (observation) =>
          observation === "Failed" || observation === "Stopped")) {
          get.set(operationAtom, OnboardingModelMachine.transition(initial, "Idle", {}))
          return Effect.void
        }
      }
      const request = requestOnboardingCancellation(initial)
      get.set(operationAtom, request.state)
      switch (request._tag) {
        case "Noop":
        case "Deferred":
          return Effect.void
        case "Download": {
          const requesting = request.state
          return Effect.gen(function* () {
            yield* get.setResult(mutations.cancel, {
              payload: { attemptIds: requesting.attemptIds },
              reactivityKeys: [LocalModelsMirror.id],
            }).pipe(
              Effect.mapError((error) => new OnboardingModelCommandFailed({
                command: "cancel",
                message: error.message,
              })),
              Effect.tapError(() => Effect.sync(() => {
                const current = get(operationAtom)
                if (current._tag === "RequestingDownloadCancellation"
                  && sameDownloadAttempts(current.attemptIds, requesting.attemptIds)) {
                  get.set(operationAtom, OnboardingModelMachine.transition(
                    current,
                    "DownloadCancellationFailed",
                    {},
                  ))
                }
              })),
            )
            const current = get(operationAtom)
            if (current._tag !== "RequestingDownloadCancellation"
              || !sameDownloadAttempts(current.attemptIds, requesting.attemptIds)) return
            get.set(operationAtom, OnboardingModelMachine.transition(
              current,
              "AwaitingDownloadCancellation",
              {},
            ))
            if (requesting.submission._tag !== "ConfigureThenLoad") {
              return yield* Effect.die("Download cancellation requires a download submission")
            }
            yield* observeAdmittedDownload(
              get.stream(modelsAtom),
              requesting.configurationId,
              requesting.attemptIds,
              (observation) => observation !== "Superseded",
            )
            const observed = get(operationAtom)
            if (observed._tag === "AwaitingDownloadCancellation"
              && sameDownloadAttempts(observed.attemptIds, requesting.attemptIds)) {
              get.set(operationAtom, OnboardingModelMachine.transition(observed, "Idle", {}))
            }
          })
        }
        case "Load": {
          const requesting = request.state
          return Effect.gen(function* () {
            yield* get.setResult(mutations.stop, {
              payload: { instanceId: requesting.instanceId },
              reactivityKeys: [ModelSlotsMirror.id],
            }).pipe(
              Effect.mapError((error) => new OnboardingModelCommandFailed({
                command: "cancel",
                message: error.message,
              })),
              Effect.tapError(() => Effect.sync(() => {
                const current = get(operationAtom)
                if (current._tag === "RequestingLoadStop"
                  && current.instanceId === requesting.instanceId) {
                  get.set(operationAtom, OnboardingModelMachine.transition(
                    current,
                    "LoadStopFailed",
                    {},
                  ))
                }
              })),
            )
            const current = get(operationAtom)
            if (current._tag !== "RequestingLoadStop"
              || current.instanceId !== requesting.instanceId) return
            get.set(operationAtom, OnboardingModelMachine.transition(
              current,
              "AwaitingLoadStop",
              {},
            ))
            yield* observeAdmittedLoad(
              get.stream(slotsAtom),
              requesting.providerModelId,
              requesting.instanceId,
              (observation) => observation === "Stopped" || observation === "Failed",
            )
            const observed = get(operationAtom)
            if (observed._tag === "AwaitingLoadStop"
              && observed.instanceId === requesting.instanceId) {
              get.set(operationAtom, OnboardingModelMachine.transition(observed, "Idle", {}))
            }
          })
        }
      }
    }),
    [modelsAtom, mutations, operationAtom, slotsAtom],
  )

  const workflowAtom = useMemo(
    () => Atom.fn<OnboardingModelSubmission>()((command, get) => {
      const idle = resetOnboardingOperation(get(operationAtom))
      get.set(operationAtom, idle)
      const awaitCancellation = Effect.gen(function* () {
        const outcome = yield* get.stream(operationAtom).pipe(
          Stream.filterMap((state) => state._tag === "Idle"
            ? Option.some("Cancelled" as const)
            : state._tag === "DownloadCancellationFailed" || state._tag === "LoadStopFailed"
              ? Option.some("CancellationFailed" as const)
              : Option.none()),
          Stream.runHead,
        )
        return yield* Option.match(outcome, {
          onNone: () => Effect.die("Onboarding cancellation observation ended"),
          onSome: Effect.succeed,
        })
      })

      const activate = (providerModelId: ProviderModelId) => Effect.gen(function* () {
        const beforeAssignment = get(operationAtom)
        const assigning = beforeAssignment._tag === "Idle"
          ? OnboardingModelMachine.transition(beforeAssignment, "Assigning", {
              submission: command,
              providerModelId,
              cancellationRequested: false,
            })
          : beforeAssignment._tag === "AdmittingDownload" || beforeAssignment._tag === "DownloadAdmitted"
            ? OnboardingModelMachine.transition(beforeAssignment, "Assigning", {
                submission: command,
                providerModelId,
                cancellationRequested: false,
              })
            : yield* Effect.die(`Cannot assign from onboarding state ${beforeAssignment._tag}`)
        get.set(operationAtom, assigning)
        yield* get.setResult(mutations.assign, {
          payload: {
            slotId: PRIMARY_SLOT_ID,
            selection: {
              providerId: ProviderIdSchema.make("local"),
              providerModelId,
              reasoningEffort: command.choice.reasoningEffort,
            },
          },
          reactivityKeys: [ModelSlotsMirror.id],
        }).pipe(
          Effect.mapError((error) => new OnboardingModelCommandFailed({
            command: "assign",
            message: error.message,
          })),
        )
        const afterAssignment = get(operationAtom)
        if (afterAssignment._tag !== "Assigning") return { _tag: "Superseded" as const }
        if (afterAssignment.cancellationRequested) {
          yield* get.setResult(mutations.clear, {
            payload: { slotId: PRIMARY_SLOT_ID },
            reactivityKeys: [ModelSlotsMirror.id],
          }).pipe(
            Effect.mapError((error) => new OnboardingModelCommandFailed({
              command: "clear",
              message: error.message,
            })),
          )
          get.set(operationAtom, OnboardingModelMachine.transition(afterAssignment, "Idle", {}))
          return { _tag: "Cancelled" as const }
        }
        const admitting = OnboardingModelMachine.transition(afterAssignment, "AdmittingLoad", {
          providerModelId,
          cancellationRequested: false,
        })
        get.set(operationAtom, admitting)
        const load = yield* get.setResult(mutations.load, {
          payload: { slotId: PRIMARY_SLOT_ID },
          reactivityKeys: [ModelSlotsMirror.id],
        }).pipe(
          Effect.mapError((error) => new OnboardingModelCommandFailed({
            command: "load",
            message: error.message,
          })),
        )
        const afterAdmission = get(operationAtom)
        if (afterAdmission._tag !== "AdmittingLoad") return { _tag: "Superseded" as const }
        const cancellationRequested = afterAdmission.cancellationRequested
        const admitted = OnboardingModelMachine.transition(afterAdmission, "LoadAdmitted", {
          providerModelId,
          instanceId: load.instanceId,
        })
        get.set(operationAtom, admitted)
        if (cancellationRequested) {
          yield* get.setResult(cancelAtom, "Cancel")
          return { _tag: "Cancelled" as const }
        }
        const loadState = yield* observeAdmittedLoad(
          get.stream(slotsAtom),
          providerModelId,
          load.instanceId,
        )
        const current = get(operationAtom)
        if (current._tag === "RequestingLoadStop" || current._tag === "AwaitingLoadStop"
          || current._tag === "LoadStopFailed") return yield* awaitCancellation
        if (current._tag !== "LoadAdmitted" || current.instanceId !== load.instanceId) {
          return { _tag: "Superseded" as const }
        }
        if (loadState !== "Ready") {
          if (loadState === "Stopped" || loadState === "Superseded") {
            get.set(operationAtom, OnboardingModelMachine.transition(current, "Idle", {}))
          }
          return { _tag: loadState }
        }
        const completing = OnboardingModelMachine.transition(current, "Completing", {})
        get.set(operationAtom, completing)
        yield* get.setResult(mutations.complete, {
          payload: { completed: true },
          reactivityKeys: [OnboardingMirror.id],
        }).pipe(
          Effect.mapError((error) => new OnboardingModelCommandFailed({
            command: "complete",
            message: error.message,
          })),
        )
        get.set(operationAtom, OnboardingModelMachine.transition(completing, "Idle", {}))
        return { _tag: "Completed" as const, instanceId: load.instanceId }
      }).pipe(
        Effect.tapError(() => Effect.sync(() => {
          const current = get(operationAtom)
          if (current._tag === "Assigning" || current._tag === "AdmittingLoad"
            || current._tag === "Completing") get.set(operationAtom, idle)
        })),
      )

      if (command._tag === "Load") return activate(command.choice.providerModelId)

      return Effect.gen(function* () {
        const beforeAdmission = get(operationAtom)
        if (beforeAdmission._tag !== "Idle") {
          return yield* Effect.die(`Cannot admit a download from ${beforeAdmission._tag}`)
        }
        const admitting = OnboardingModelMachine.transition(beforeAdmission, "AdmittingDownload", {
          submission: command,
          cancellationRequested: false,
        })
        get.set(operationAtom, admitting)
        const installation = yield* get.setResult(mutations.install, {
          payload: { configurationId: command.choice.configurationId },
          reactivityKeys: [LocalModelsMirror.id, ProviderModelCatalogMirror.id],
        }).pipe(
          Effect.mapError((error) => new OnboardingModelCommandFailed({
            command: "install",
            message: error.message,
          })),
        )
        const acquisition = installation.download
        if (acquisition._tag === "AlreadyInstalled") {
          const current = get(operationAtom)
          if (current._tag !== "AdmittingDownload") return { _tag: "Superseded" as const }
          if (current.cancellationRequested) {
            get.set(operationAtom, OnboardingModelMachine.transition(current, "Idle", {}))
            return { _tag: "Cancelled" as const }
          }
          return yield* activate(installation.providerModelId)
        }
        const afterAdmission = get(operationAtom)
        if (afterAdmission._tag !== "AdmittingDownload") return { _tag: "Superseded" as const }
        const cancellationRequested = afterAdmission.cancellationRequested
        const admitted = OnboardingModelMachine.transition(afterAdmission, "DownloadAdmitted", {
          configurationId: command.choice.configurationId,
          providerModelId: installation.providerModelId,
          attemptIds: acquisition.attemptIds,
        })
        get.set(operationAtom, admitted)
        if (cancellationRequested) {
          yield* get.setResult(cancelAtom, "Cancel")
          return { _tag: "Cancelled" as const }
        }
        const downloadState = yield* observeAdmittedDownload(
          get.stream(modelsAtom),
          command.choice.configurationId,
          acquisition.attemptIds,
        )
        const current = get(operationAtom)
        if (current._tag === "RequestingDownloadCancellation"
          || current._tag === "AwaitingDownloadCancellation"
          || current._tag === "DownloadCancellationFailed") return yield* awaitCancellation
        if (current._tag !== "DownloadAdmitted"
          || !sameDownloadAttempts(current.attemptIds, acquisition.attemptIds)) {
          return { _tag: "Superseded" as const }
        }
        if (downloadState !== "Downloaded") {
          if (downloadState === "Cancelled" || downloadState === "Superseded") {
            get.set(operationAtom, OnboardingModelMachine.transition(current, "Idle", {}))
          }
          return { _tag: downloadState }
        }
        return yield* activate(installation.providerModelId)
      }).pipe(
        Effect.tapError(() => Effect.sync(() => {
          const current = get(operationAtom)
          if (current._tag === "AdmittingDownload") {
            get.set(operationAtom, OnboardingModelMachine.transition(current, "Idle", {}))
          }
        })),
      )
    }),
    [cancelAtom, modelsAtom, mutations, operationAtom, slotsAtom],
  )
  const runWorkflow = useAtomSet(workflowAtom)
  const runCancel = useAtomSet(cancelAtom)
  const setupAtom = useMemo(
    () => Atom.make((get) => {
      const operation = get(operationAtom)
      return {
        hardware: Result.map(get(hardwareAtom), ({ state }) => state),
        models: Result.map(get(modelsAtom), ({ state }) => state),
        catalog: Result.map(get(catalogAtom), ({ state }) => state),
        slots: Result.map(get(slotsAtom), ({ state }) => state),
        workflowResult: get(workflowAtom),
        cancelResult: get(cancelAtom),
        submission: onboardingSubmission(operation),
        providerModelId: onboardingProviderModelId(operation),
        cancelling: onboardingCancellationPending(operation),
      }
    }),
    [cancelAtom, catalogAtom, hardwareAtom, modelsAtom, operationAtom, slotsAtom, workflowAtom],
  )
  const setup = useAtomValue(setupAtom)

  const load = useCallback((choice: OnboardingLoadModelChoice) => {
    if (Result.isWaiting(setup.workflowResult)) return
    runWorkflow({ _tag: "Load", choice })
  }, [runWorkflow, setup.workflowResult])

  const configureThenLoad = useCallback((choice: OnboardingConfigurationChoice) => {
    if (Result.isWaiting(setup.workflowResult)) return
    runWorkflow({ _tag: "ConfigureThenLoad", choice })
  }, [runWorkflow, setup.workflowResult])

  const cancel = useCallback(() => {
    runCancel("Cancel")
  }, [runCancel])

  return {
    ...setup,
    slotActions,
    load,
    configureThenLoad,
    cancel,
  }
}
