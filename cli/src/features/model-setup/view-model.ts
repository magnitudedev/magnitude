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
  | { readonly _tag: "Configuring"; readonly candidate: LocalModelCatalogCandidate }
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
  providerModelId: ProviderModelId,
  phase: ActivatingView["phase"],
  failure: string | null = null,
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
  const candidate = submission._tag === "ConfigureThenLoad"
    && models.recommendations._tag === "Ready"
    ? models.recommendations.catalog.find(({ configurationId }) =>
        configurationId === submission.choice.configurationId)
    : undefined
  if (candidate?.download._tag === "Failed") {
    return { _tag: "DownloadFailed", candidate }
  }
  if (candidate?.download._tag === "Cancelled") return { _tag: "Choosing" }
  if (candidate?.download._tag === "Downloading") {
    return { _tag: "Downloading", candidate }
  }
  if (submission._tag === "ConfigureThenLoad"
    && candidate?.download._tag === "Downloaded"
    && submitting
    && Option.isNone(providerModelId)) {
    return { _tag: "Configuring", candidate }
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
        lifecycle.failure.message,
      )
    }
  }
  if (lifecycle?._tag === "Stopped") return { _tag: "Choosing" }

  if (!submitting) return { _tag: "Choosing" }
  if (submission._tag === "ConfigureThenLoad"
    && candidate !== undefined
    && candidate.download._tag !== "Downloaded") {
    return { _tag: "Downloading", candidate }
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
    case "Downloading": return `Downloading ${view.candidate.displayName}…`
    case "DownloadFailed": return `Couldn’t download ${view.candidate.displayName}`
    case "Configuring": return `Configuring ${view.candidate.displayName}…`
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
