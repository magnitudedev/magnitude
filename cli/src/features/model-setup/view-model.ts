import { Option } from "effect"
import type { OnboardingModelWorkflowIntent } from "@magnitudedev/client-common"
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
      readonly phase: "Loading" | "Ready" | "Failed"
      readonly failure: string | null
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
  active,
  intent,
  models,
  slots,
}: {
  readonly active: boolean
  readonly intent: OnboardingModelWorkflowIntent
  readonly models: LocalModelsState
  readonly slots: ModelSlotsState
}): OnboardingModelSetupView => {
  if (!active) return { _tag: "Inactive" }
  if (intent._tag === "Idle") return { _tag: "Choosing" }

  const operation = intent._tag === "Cancelling" ? intent.operation : intent
  if (operation._tag === "Downloading") {
    const candidate = models.recommendations._tag === "Ready"
      ? models.recommendations.catalog.find(({ providerModelId }) =>
          providerModelId === operation.choice.providerModelId)
      : undefined
    if (candidate === undefined) return { _tag: "Choosing" }
    return candidate.download._tag === "Failed"
      ? { _tag: "DownloadFailed", candidate }
      : { _tag: "Downloading", candidate }
  }

  const primary = slots.slots.primary
  const lifecycle = primary._tag === "ConfiguredLocal"
    && primary.selection.providerModelId === operation.choice.providerModelId
    ? Option.getOrNull(Option.map(primary.instance, ({ lifecycle }) => lifecycle))
    : null
  if (lifecycle?._tag === "Ready" || lifecycle?._tag === "Failed") {
    return {
      _tag: "Activating",
      providerModelId: operation.choice.providerModelId,
      displayName: operation.choice.displayName,
      reasoningEffort: operation.choice.reasoningEffort,
      phase: lifecycle._tag,
      failure: lifecycle._tag === "Failed" ? lifecycle.failure.message : null,
    }
  }
  return {
    _tag: "Activating",
    providerModelId: operation.choice.providerModelId,
    displayName: operation.choice.displayName,
    reasoningEffort: operation.choice.reasoningEffort,
    phase: "Loading",
    failure: null,
  }
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
        : view.phase === "Ready"
          ? `Finishing setup for ${view.displayName}…`
          : `Couldn’t load ${view.displayName}`
  }
}
