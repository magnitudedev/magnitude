import { Data, Effect, Option, Stream } from "effect"
import { Result } from "@effect-atom/atom-react"
import { FSM } from "@magnitudedev/utils"
import type {
  DownloadAttemptId,
  LocalModelsState,
  ModelInstanceId,
  ModelServingConfigurationId,
  ModelSlotsState,
  ProviderModelId,
  ReasoningEffort,
} from "@magnitudedev/sdk"

interface OnboardingModelChoiceBase {
  readonly displayName: string
  readonly reasoningEffort: ReasoningEffort
}

export interface OnboardingLoadModelChoice extends OnboardingModelChoiceBase {
  readonly providerModelId: ProviderModelId
}

export interface OnboardingConfigurationChoice extends OnboardingModelChoiceBase {
  readonly configurationId: ModelServingConfigurationId
}

export type OnboardingModelSubmission =
  | { readonly _tag: "Load"; readonly choice: OnboardingLoadModelChoice }
  | { readonly _tag: "InstallThenLoad"; readonly choice: OnboardingConfigurationChoice }

type SubmissionProps = { readonly submission: OnboardingModelSubmission }
type DownloadProps = SubmissionProps & {
  readonly configurationId: ModelServingConfigurationId
  readonly providerModelId: ProviderModelId
  readonly attemptIds: readonly [DownloadAttemptId, ...DownloadAttemptId[]]
}
type LoadProps = SubmissionProps & {
  readonly providerModelId: ProviderModelId
  readonly instanceId: ModelInstanceId
}

export class OnboardingIdle extends Data.TaggedClass("Idle")<{}> {}
export class OnboardingRequestingInstallation extends Data.TaggedClass("RequestingInstallation")<
  SubmissionProps & { readonly cancellationRequested: boolean }
> {}
export class OnboardingDownloadAdmitted extends Data.TaggedClass("DownloadAdmitted")<DownloadProps> {}
export class OnboardingRequestingDownloadCancellation extends Data.TaggedClass(
  "RequestingDownloadCancellation",
)<DownloadProps> {}
export class OnboardingAwaitingDownloadCancellation extends Data.TaggedClass(
  "AwaitingDownloadCancellation",
)<DownloadProps> {}
export class OnboardingDownloadCancellationFailed extends Data.TaggedClass(
  "DownloadCancellationFailed",
)<DownloadProps> {}
export class OnboardingAssigning extends Data.TaggedClass("Assigning")<
  SubmissionProps & { readonly providerModelId: ProviderModelId; readonly cancellationRequested: boolean }
> {}
export class OnboardingAdmittingLoad extends Data.TaggedClass("AdmittingLoad")<
  SubmissionProps & { readonly providerModelId: ProviderModelId; readonly cancellationRequested: boolean }
> {}
export class OnboardingLoadAdmitted extends Data.TaggedClass("LoadAdmitted")<LoadProps> {}
export class OnboardingRequestingLoadStop extends Data.TaggedClass("RequestingLoadStop")<LoadProps> {}
export class OnboardingAwaitingLoadStop extends Data.TaggedClass("AwaitingLoadStop")<LoadProps> {}
export class OnboardingLoadStopFailed extends Data.TaggedClass("LoadStopFailed")<LoadProps> {}
export class OnboardingCompleting extends Data.TaggedClass("Completing")<LoadProps> {}

export const OnboardingModelMachine = FSM.defineFSM(
  {
    Idle: OnboardingIdle,
    RequestingInstallation: OnboardingRequestingInstallation,
    DownloadAdmitted: OnboardingDownloadAdmitted,
    RequestingDownloadCancellation: OnboardingRequestingDownloadCancellation,
    AwaitingDownloadCancellation: OnboardingAwaitingDownloadCancellation,
    DownloadCancellationFailed: OnboardingDownloadCancellationFailed,
    Assigning: OnboardingAssigning,
    AdmittingLoad: OnboardingAdmittingLoad,
    LoadAdmitted: OnboardingLoadAdmitted,
    RequestingLoadStop: OnboardingRequestingLoadStop,
    AwaitingLoadStop: OnboardingAwaitingLoadStop,
    LoadStopFailed: OnboardingLoadStopFailed,
    Completing: OnboardingCompleting,
  },
  {
    Idle: ["RequestingInstallation", "Assigning"],
    RequestingInstallation: ["DownloadAdmitted", "Assigning", "Idle"],
    DownloadAdmitted: ["RequestingDownloadCancellation", "Assigning", "Idle"],
    RequestingDownloadCancellation: ["AwaitingDownloadCancellation", "DownloadCancellationFailed"],
    AwaitingDownloadCancellation: ["Idle"],
    DownloadCancellationFailed: ["RequestingDownloadCancellation", "Idle"],
    Assigning: ["AdmittingLoad", "Idle"],
    AdmittingLoad: ["LoadAdmitted", "Idle"],
    LoadAdmitted: ["RequestingLoadStop", "Completing", "Idle"],
    RequestingLoadStop: ["AwaitingLoadStop", "LoadStopFailed"],
    AwaitingLoadStop: ["Idle"],
    LoadStopFailed: ["RequestingLoadStop", "Idle"],
    Completing: ["Idle"],
  } as const,
)

export type OnboardingModelOperation = FSM.StateUnion<typeof OnboardingModelMachine.stateClasses>

export const resetOnboardingOperation = (
  state: OnboardingModelOperation,
): OnboardingIdle => {
  switch (state._tag) {
    case "Idle": return state
    case "DownloadAdmitted":
    case "DownloadCancellationFailed":
    case "LoadAdmitted":
    case "LoadStopFailed":
      return OnboardingModelMachine.transition(state, "Idle", {})
    case "RequestingInstallation":
    case "RequestingDownloadCancellation":
    case "AwaitingDownloadCancellation":
    case "Assigning":
    case "AdmittingLoad":
    case "RequestingLoadStop":
    case "AwaitingLoadStop":
    case "Completing":
      throw new Error(`Cannot replace active onboarding operation ${state._tag}`)
  }
}

export const onboardingSubmission = (
  state: OnboardingModelOperation,
): OnboardingModelSubmission | null => state._tag === "Idle" ? null : state.submission

export const onboardingProviderModelId = (
  state: OnboardingModelOperation,
): Option.Option<ProviderModelId> => {
  if (state._tag === "Idle") return Option.none()
  if (state.submission._tag === "Load") return Option.some(state.submission.choice.providerModelId)
  switch (state._tag) {
    case "DownloadAdmitted":
    case "RequestingDownloadCancellation":
    case "AwaitingDownloadCancellation":
    case "DownloadCancellationFailed":
      return Option.none()
    case "Assigning":
    case "AdmittingLoad":
    case "LoadAdmitted":
    case "RequestingLoadStop":
    case "AwaitingLoadStop":
    case "LoadStopFailed":
    case "Completing":
      return Option.some(state.providerModelId)
    case "RequestingInstallation":
      return Option.none()
  }
}

export const onboardingCancellationPending = (state: OnboardingModelOperation): boolean => {
  switch (state._tag) {
    case "RequestingInstallation":
    case "Assigning":
    case "AdmittingLoad":
      return state.cancellationRequested
    case "RequestingDownloadCancellation":
    case "AwaitingDownloadCancellation":
    case "RequestingLoadStop":
    case "AwaitingLoadStop":
      return true
    case "Idle":
    case "DownloadAdmitted":
    case "DownloadCancellationFailed":
    case "LoadAdmitted":
    case "LoadStopFailed":
    case "Completing":
      return false
  }
}

export type OnboardingCancellationRequest =
  | { readonly _tag: "Noop"; readonly state: OnboardingModelOperation }
  | { readonly _tag: "Deferred"; readonly state:
      OnboardingRequestingInstallation | OnboardingAssigning | OnboardingAdmittingLoad }
  | { readonly _tag: "Download"; readonly state: OnboardingRequestingDownloadCancellation }
  | { readonly _tag: "Load"; readonly state: OnboardingRequestingLoadStop }

export const requestOnboardingCancellation = (
  state: OnboardingModelOperation,
): OnboardingCancellationRequest => {
  switch (state._tag) {
    case "RequestingInstallation":
    case "Assigning":
    case "AdmittingLoad":
      return {
        _tag: "Deferred",
        state: OnboardingModelMachine.hold(state, { cancellationRequested: true }),
      }
    case "DownloadAdmitted":
    case "DownloadCancellationFailed":
      return {
        _tag: "Download",
        state: OnboardingModelMachine.transition(state, "RequestingDownloadCancellation", {}),
      }
    case "LoadAdmitted":
    case "LoadStopFailed":
      return {
        _tag: "Load",
        state: OnboardingModelMachine.transition(state, "RequestingLoadStop", {}),
      }
    case "Idle":
    case "RequestingDownloadCancellation":
    case "AwaitingDownloadCancellation":
    case "RequestingLoadStop":
    case "AwaitingLoadStop":
    case "Completing":
      return { _tag: "Noop", state }
  }
}

export interface ObservationCorrelation {
  readonly exactIdentitySeen: boolean
}

export const initialObservationCorrelation: ObservationCorrelation = {
  exactIdentitySeen: false,
}

export type DownloadObservation = "Downloaded" | "Failed" | "Cancelled" | "Superseded"

export const reduceDownloadObservation = (
  correlation: ObservationCorrelation,
  state: LocalModelsState,
  configurationId: ModelServingConfigurationId,
  attemptIds: readonly [DownloadAttemptId, ...DownloadAttemptId[]],
): readonly [ObservationCorrelation, Option.Option<DownloadObservation>] => {
  const model = state.models.find(({ servingState }) =>
    servingState._tag !== "Resolving"
    && (servingState._tag !== "Failed"
      ? servingState.configuration.id === configurationId
      : Option.exists(servingState.configuration, ({ id }) => id === configurationId)))
  return Option.match(
    Option.fromNullable(model),
    {
      onNone: () => [correlation, Option.none()],
      onSome: ({ acquisitionState }) => {
        if (acquisitionState._tag === "Installed") {
          return [correlation, Option.some("Downloaded")]
        }
        if (acquisitionState._tag === "NotInstalled") return [correlation, Option.none()]
        const admitted = new Set<DownloadAttemptId>(attemptIds)
        const exact = acquisitionState.attemptIds.some((attemptId) => admitted.has(attemptId))
        if (!exact) {
          return correlation.exactIdentitySeen
            ? [correlation, Option.some("Superseded")]
            : [correlation, Option.none()]
        }
        const next = { exactIdentitySeen: true }
        switch (acquisitionState._tag) {
          case "Downloading": return [next, Option.none()]
          case "Failed": return [next, Option.some("Failed")]
          case "Cancelled": return [next, Option.some("Cancelled")]
        }
      },
    },
  )
}

export type LoadObservation = "Ready" | "Failed" | "Stopped" | "Superseded"

export const reduceLoadObservation = (
  correlation: ObservationCorrelation,
  state: ModelSlotsState,
  providerModelId: ProviderModelId,
  instanceId: ModelInstanceId,
): readonly [ObservationCorrelation, Option.Option<LoadObservation>] => {
  const primary = state.slots.primary
  if (primary._tag !== "ConfiguredLocal"
    || primary.selection.providerId !== "local"
    || primary.selection.providerModelId !== providerModelId) {
    return correlation.exactIdentitySeen
      ? [correlation, Option.some("Superseded")]
      : [correlation, Option.none()]
  }
  if (Option.isNone(primary.instance)) return [correlation, Option.none()]
  if (primary.instance.value.id !== instanceId) {
    return correlation.exactIdentitySeen
      ? [correlation, Option.some("Superseded")]
      : [correlation, Option.none()]
  }
  const next = { exactIdentitySeen: true }
  switch (primary.instance.value.lifecycle._tag) {
    case "Loading":
    case "Stopping":
      return [next, Option.none()]
    case "Ready": return [next, Option.some("Ready")]
    case "Failed": return [next, Option.some("Failed")]
    case "Stopped": return [next, Option.some("Stopped")]
  }
}

export const observeAdmittedDownload = <E, R>(
  results: Stream.Stream<Result.Result<LocalModelsState, E>, never, R>,
  configurationId: ModelServingConfigurationId,
  attemptIds: readonly [DownloadAttemptId, ...DownloadAttemptId[]],
  accept: (observation: DownloadObservation) => boolean = () => true,
): Effect.Effect<DownloadObservation, never, R> => results.pipe(
  // Ignore a previous cached value while a synchronized refresh is in flight.
  Stream.filterMap((result) => Result.isSuccess(result) && !result.waiting
    ? Option.some(result.value)
    : Option.none()),
  Stream.mapAccum(initialObservationCorrelation, (correlation, state) => {
    const [next, observation] = reduceDownloadObservation(
      correlation,
      state,
      configurationId,
      attemptIds,
    )
    return [next, Option.filter(observation, accept)] as const
  }),
  Stream.filterMap((observation) => observation),
  Stream.runHead,
  Effect.flatMap(Option.match({
    onNone: () => Effect.die("Local-model query observation ended"),
    onSome: Effect.succeed,
  })),
)

export const observeAdmittedLoad = <E, R>(
  results: Stream.Stream<Result.Result<{ readonly state: ModelSlotsState }, E>, never, R>,
  providerModelId: ProviderModelId,
  instanceId: ModelInstanceId,
  accept: (observation: LoadObservation) => boolean = () => true,
): Effect.Effect<LoadObservation, never, R> => results.pipe(
  // Ignore the previous cached value while the admission-triggered reread is in flight.
  Stream.filterMap((result) => Result.isSuccess(result) && !result.waiting
    ? Option.some(result.value.state)
    : Option.none()),
  Stream.mapAccum(initialObservationCorrelation, (correlation, state) => {
    const [next, observation] = reduceLoadObservation(correlation, state, providerModelId, instanceId)
    return [next, Option.filter(observation, accept)] as const
  }),
  Stream.filterMap((observation) => observation),
  Stream.runHead,
  Effect.flatMap(Option.match({
    onNone: () => Effect.die("Model-slot query observation ended"),
    onSome: Effect.succeed,
  })),
)

export const sameDownloadAttempts = (
  left: readonly DownloadAttemptId[],
  right: readonly DownloadAttemptId[],
): boolean => left.length === right.length
  && left.every((attemptId, index) => attemptId === right[index])
