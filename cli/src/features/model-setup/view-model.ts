import { Option } from "effect"
import {
  findLocalModelByConfigurationId,
  formatLocalModelDisplayName,
  type OnboardingModelOperation,
  type OnboardingModelSubmission,
} from "@magnitudedev/client-common"
import type {
  LocalModel,
  LocalModelsState,
  ModelInstanceFailure,
  ModelSlotsState,
  ProviderModelId,
  ReasoningEffort,
} from "@magnitudedev/sdk"

export type OnboardingModelSetupView =
  | { readonly _tag: "Choosing" }
  | {
      readonly _tag: "Downloading"
      readonly model: LocalModel
      readonly starting: boolean
      readonly cancelling: boolean
    }
  | { readonly _tag: "DownloadFailed"; readonly model: LocalModel }
  | { readonly _tag: "Configuring"; readonly model: LocalModel }
  | {
      readonly _tag: "Activating"
      readonly providerModelId: ProviderModelId
      readonly displayName: string
      readonly reasoningEffort: ReasoningEffort
      readonly phase: "Loading" | "Stopping" | "Ready" | "Failed"
      readonly failure: ModelInstanceFailure | null
    }

type ActivatingView = Extract<OnboardingModelSetupView, { readonly _tag: "Activating" }>
type InstallationSubmission = Extract<
  OnboardingModelSubmission,
  { readonly _tag: "InstallThenLoad" }
>
type AdmittedLoadOperation = Extract<
  OnboardingModelOperation,
  { readonly _tag: "LoadAdmitted" | "LoadStopFailed" }
>

const activatingView = (
  choice: OnboardingModelSubmission["choice"],
  providerModelId: ProviderModelId,
  phase: ActivatingView["phase"],
  failure: ModelInstanceFailure | null = null,
): ActivatingView => ({
  _tag: "Activating",
  providerModelId,
  displayName: choice.displayName,
  reasoningEffort: choice.reasoningEffort,
  phase,
  failure,
})

const installationModel = (
  submission: InstallationSubmission,
  models: LocalModelsState,
): LocalModel => {
  const model = Option.getOrUndefined(findLocalModelByConfigurationId(
    models.models,
    submission.choice.configurationId,
  ))
  if (model === undefined) {
    throw new Error(
      `Onboarding model ${submission.choice.configurationId} is absent from LocalModelsState`,
    )
  }
  return model
}

const admittedDownloadView = (
  operation: Extract<OnboardingModelOperation, { readonly _tag: "DownloadAdmitted" }>,
  models: LocalModelsState,
): OnboardingModelSetupView => {
  const model = installationModel(operation.submission, models)
  const acquisition = model.acquisitionState
  if (acquisition._tag === "Installed") return { _tag: "Configuring", model }
  const admittedAttempts = new Set(operation.attemptIds)
  if (acquisition._tag === "NotInstalled"
    || !acquisition.attemptIds.some((attemptId) => admittedAttempts.has(attemptId))) {
    return { _tag: "Downloading", model, starting: true, cancelling: false }
  }
  switch (acquisition._tag) {
    case "Failed": return { _tag: "DownloadFailed", model }
    case "Cancelled": return { _tag: "Choosing" }
    case "Downloading":
      return { _tag: "Downloading", model, starting: false, cancelling: false }
  }
}

const admittedDownloadStarting = (
  model: LocalModel,
  operation: Extract<OnboardingModelOperation, {
    readonly _tag:
      | "RequestingDownloadCancellation"
      | "AwaitingDownloadCancellation"
      | "DownloadCancellationFailed"
  }>,
): boolean => {
  const acquisition = model.acquisitionState
  if (acquisition._tag === "NotInstalled" || acquisition._tag === "Installed") return true
  const admittedAttempts = new Set(operation.attemptIds)
  return !acquisition.attemptIds.some((attemptId) => admittedAttempts.has(attemptId))
}

const correlatedInstanceView = (
  operation: AdmittedLoadOperation,
  slots: ModelSlotsState,
): OnboardingModelSetupView => {
  const fallback = activatingView(
    operation.submission.choice,
    operation.providerModelId,
    "Loading",
  )
  const primary = slots.slots.primary
  if (primary._tag !== "ConfiguredLocal"
    || primary.selection.providerId !== "local"
    || primary.selection.providerModelId !== operation.providerModelId
    || Option.isNone(primary.instance)
    || primary.instance.value.id !== operation.instanceId) return fallback

  const lifecycle = primary.instance.value.lifecycle
  switch (lifecycle._tag) {
    case "Loading":
    case "Stopping":
    case "Ready":
      return activatingView(
        operation.submission.choice,
        operation.providerModelId,
        lifecycle._tag,
      )
    case "Failed":
      return activatingView(
        operation.submission.choice,
        operation.providerModelId,
        "Failed",
        lifecycle.failure,
      )
    case "Stopped": return { _tag: "Choosing" }
  }
}

export const deriveModelSetupActive = ({
  forceSetup,
  onboardingRequired,
  completionSucceeded,
}: {
  readonly forceSetup: boolean
  readonly onboardingRequired: boolean
  readonly completionSucceeded: boolean
}): boolean => {
  if (onboardingRequired) return true
  return forceSetup && !completionSucceeded
}

export const deriveOnboardingModelSetupView = ({
  operationState,
  models,
  slots,
}: {
  readonly operationState: OnboardingModelOperation
  readonly models: LocalModelsState
  readonly slots: ModelSlotsState
}): OnboardingModelSetupView => {
  switch (operationState._tag) {
    case "Idle": return { _tag: "Choosing" }
    case "RequestingInstallation":
      return {
        _tag: "Downloading",
        model: installationModel(operationState.submission, models),
        starting: true,
        cancelling: operationState.cancellationRequested,
      }
    case "DownloadAdmitted":
      return admittedDownloadView(operationState, models)
    case "RequestingDownloadCancellation":
    case "AwaitingDownloadCancellation": {
      const model = installationModel(operationState.submission, models)
      return {
        _tag: "Downloading",
        model,
        starting: admittedDownloadStarting(model, operationState),
        cancelling: true,
      }
    }
    case "DownloadCancellationFailed": {
      const model = installationModel(operationState.submission, models)
      return {
        _tag: "Downloading",
        model,
        starting: admittedDownloadStarting(model, operationState),
        cancelling: false,
      }
    }
    case "Assigning":
      return operationState.submission._tag === "InstallThenLoad"
        ? { _tag: "Configuring", model: installationModel(operationState.submission, models) }
        : activatingView(
            operationState.submission.choice,
            operationState.providerModelId,
            "Loading",
          )
    case "AdmittingLoad":
      return activatingView(
        operationState.submission.choice,
        operationState.providerModelId,
        operationState.cancellationRequested ? "Stopping" : "Loading",
      )
    case "LoadAdmitted":
    case "LoadStopFailed":
      return correlatedInstanceView(operationState, slots)
    case "RequestingLoadStop":
    case "AwaitingLoadStop":
      return activatingView(
        operationState.submission.choice,
        operationState.providerModelId,
        "Stopping",
      )
    case "Completing":
      return activatingView(
        operationState.submission.choice,
        operationState.providerModelId,
        "Ready",
      )
  }
}

export const onboardingModelSetupPlaceholder = (view: OnboardingModelSetupView): string => {
  switch (view._tag) {
    case "Choosing": return "Select a model to start coding…"
    case "Downloading": return `Downloading ${formatLocalModelDisplayName(view.model)}…`
    case "DownloadFailed": return `Couldn’t download ${formatLocalModelDisplayName(view.model)}`
    case "Configuring": return `Configuring ${formatLocalModelDisplayName(view.model)}…`
    case "Activating":
      return view.phase === "Loading"
        ? `Loading ${view.displayName}…`
        : view.phase === "Stopping"
          ? `Stopping ${view.displayName}…`
          : view.phase === "Ready"
            ? `Finishing setup for ${view.displayName}…`
            : `Couldn’t load ${view.displayName}`
  }
}
