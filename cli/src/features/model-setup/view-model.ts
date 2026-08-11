import { Option } from "effect"
import {
  findLocalModelByConfigurationId,
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
  | { readonly _tag: "Inactive" }
  | { readonly _tag: "Choosing" }
  | { readonly _tag: "Downloading"; readonly model: LocalModel }
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
  active,
  submission,
  providerModelId,
  submitting,
  models,
  slots,
}: {
  readonly active: boolean
  readonly submission: OnboardingModelSubmission | null
  readonly providerModelId: Option.Option<ProviderModelId>
  readonly submitting: boolean
  readonly models: LocalModelsState
  readonly slots: ModelSlotsState
}): OnboardingModelSetupView => {
  if (!active) return { _tag: "Inactive" }
  if (submission === null) return { _tag: "Choosing" }

  const choice = submission.choice
  const model = submission._tag === "InstallThenLoad"
    ? Option.getOrUndefined(findLocalModelByConfigurationId(
        models.models,
        submission.choice.configurationId,
      ))
    : undefined
  if (model?.acquisitionState._tag === "Failed") {
    return { _tag: "DownloadFailed", model }
  }
  if (model?.acquisitionState._tag === "Cancelled") return { _tag: "Choosing" }
  if (model?.acquisitionState._tag === "Downloading") {
    return { _tag: "Downloading", model }
  }
  if (submission._tag === "InstallThenLoad"
    && model?.acquisitionState._tag === "Installed"
    && submitting
    && Option.isNone(providerModelId)) {
    return { _tag: "Configuring", model }
  }

  const primary = slots.slots.primary
  const lifecycle = primary._tag === "ConfiguredLocal"
    && Option.contains(providerModelId, primary.selection.providerModelId)
    ? Option.getOrNull(Option.map(primary.instance, ({ lifecycle }) => lifecycle))
    : null
  if (primary._tag === "ConfiguredLocal") {
    if (lifecycle?._tag === "Loading"
      || lifecycle?._tag === "Stopping"
      || lifecycle?._tag === "Ready") {
      return activatingView(choice, primary.selection.providerModelId, lifecycle._tag)
    }
    if (lifecycle?._tag === "Failed") {
      return activatingView(
        choice,
        primary.selection.providerModelId,
        "Failed",
        lifecycle.failure,
      )
    }
  }
  if (lifecycle?._tag === "Stopped") return { _tag: "Choosing" }

  if (!submitting) return { _tag: "Choosing" }
  if (submission._tag === "InstallThenLoad"
    && model !== undefined
    && model.acquisitionState._tag !== "Installed") {
    return { _tag: "Downloading", model }
  }
  return Option.match(providerModelId, {
    onNone: () => ({ _tag: "Choosing" as const }),
    onSome: (id) => activatingView(choice, id, "Loading"),
  })
}

export const onboardingModelSetupPlaceholder = (view: OnboardingModelSetupView): string | null => {
  switch (view._tag) {
    case "Inactive": return null
    case "Choosing": return "Select a model to start coding…"
    case "Downloading": return `Downloading ${view.model.presentation.displayName}…`
    case "DownloadFailed": return `Couldn’t download ${view.model.presentation.displayName}`
    case "Configuring": return `Configuring ${view.model.presentation.displayName}…`
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
