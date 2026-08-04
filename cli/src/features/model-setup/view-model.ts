import { Option } from "effect"
import type { OnboardingModelSubmission } from "@magnitudedev/client-common"
import type {
  LocalModelCatalogCandidate,
  LocalModelsState,
  ModelSlotsState,
  ProviderModelId,
  ReasoningEffort,
} from "@magnitudedev/sdk"

export type OnboardingModelSetupView =
  | { readonly _tag: "Inactive" }
  | { readonly _tag: "Choosing" }
  | { readonly _tag: "Downloading"; readonly candidate: LocalModelCatalogCandidate }
  | { readonly _tag: "DownloadFailed"; readonly candidate: LocalModelCatalogCandidate }
  | {
      readonly _tag: "Activating"
      readonly providerModelId: ProviderModelId
      readonly displayName: string
      readonly reasoningEffort: ReasoningEffort
      readonly phase: "Loading" | "Stopping" | "Ready" | "Failed"
      readonly failure: string | null
    }

type ActivatingView = Extract<OnboardingModelSetupView, { readonly _tag: "Activating" }>

const activatingView = (
  choice: OnboardingModelSubmission["choice"],
  phase: ActivatingView["phase"],
  failure: string | null = null,
): ActivatingView => ({
  _tag: "Activating",
  providerModelId: choice.providerModelId,
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
  submitting,
  models,
  slots,
}: {
  readonly active: boolean
  readonly submission: OnboardingModelSubmission | null
  readonly submitting: boolean
  readonly models: LocalModelsState
  readonly slots: ModelSlotsState
}): OnboardingModelSetupView => {
  if (!active) return { _tag: "Inactive" }
  if (submission === null) return { _tag: "Choosing" }

  const choice = submission.choice
  const candidate = submission._tag === "DownloadThenLoad"
    && models.recommendations._tag === "Ready"
    ? models.recommendations.catalog.find(({ targetId }) =>
        targetId === submission.choice.targetId)
    : undefined
  if (candidate?.download._tag === "Failed") {
    return { _tag: "DownloadFailed", candidate }
  }
  if (candidate?.download._tag === "Cancelled") return { _tag: "Choosing" }
  if (candidate?.download._tag === "Downloading") {
    return { _tag: "Downloading", candidate }
  }

  const primary = slots.slots.primary
  const lifecycle = primary._tag === "ConfiguredLocal"
    && primary.selection.providerModelId === choice.providerModelId
    ? Option.getOrNull(Option.map(primary.instance, ({ lifecycle }) => lifecycle))
    : null
  if (lifecycle?._tag === "Loading"
    || lifecycle?._tag === "Stopping"
    || lifecycle?._tag === "Ready") {
    return activatingView(choice, lifecycle._tag)
  }
  if (lifecycle?._tag === "Failed") {
    return activatingView(choice, "Failed", lifecycle.failure.message)
  }
  if (lifecycle?._tag === "Stopped") return { _tag: "Choosing" }

  if (!submitting) return { _tag: "Choosing" }
  if (submission._tag === "DownloadThenLoad" && candidate !== undefined) {
    return { _tag: "Downloading", candidate }
  }
  return activatingView(choice, "Loading")
}

export const onboardingModelSetupPlaceholder = (view: OnboardingModelSetupView): string | null => {
  switch (view._tag) {
    case "Inactive": return null
    case "Choosing": return "Select a model to start coding…"
    case "Downloading": return `Downloading ${view.candidate.displayName}…`
    case "DownloadFailed": return `Couldn’t download ${view.candidate.displayName}`
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
