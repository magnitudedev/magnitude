import { Option } from "effect"
import {
  deriveSelectedLocalModelCandidate,
  type LocalInferenceView,
} from "@magnitudedev/client-common"
import type { LocalModelCatalogCandidate, ProviderModelId } from "@magnitudedev/sdk"
import {
  buildLocalInferenceSelections,
  localInferenceSetupPhase,
} from "../local-inference/view-model"

export type OnboardingModelSetupView =
  | { readonly _tag: "Inactive" }
  | { readonly _tag: "Preparing"; readonly state: LocalInferenceView | null }
  | { readonly _tag: "Choosing"; readonly state: LocalInferenceView }
  | {
      readonly _tag: "Downloading"
      readonly state: LocalInferenceView
      readonly candidate: LocalModelCatalogCandidate
    }
  | {
      readonly _tag: "Activating"
      readonly state: LocalInferenceView
      readonly providerModelId: ProviderModelId
      readonly displayName: string
      readonly phase: "Preparing" | "Loading" | "Ready" | "Failed"
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
  onboardingRequired,
  submittedProviderModelId,
  state,
}: {
  readonly active: boolean
  readonly onboardingRequired: boolean
  readonly submittedProviderModelId: ProviderModelId | null
  readonly state: LocalInferenceView | null
}): OnboardingModelSetupView => {
  if (!active) return { _tag: "Inactive" }
  if (state === null || localInferenceSetupPhase(state) !== "ready") {
    return { _tag: "Preparing", state }
  }
  if (!onboardingRequired) return { _tag: "Choosing", state }

  const submittedCandidate = submittedProviderModelId === null
    || state.models.recommendations._tag !== "Ready"
    ? null
    : state.models.recommendations.catalog.find(({ providerModelId }) =>
        providerModelId === submittedProviderModelId) ?? null
  const download = submittedCandidate
    && (submittedCandidate.download._tag === "Downloading"
      || submittedCandidate.download._tag === "Failed"
      || submittedCandidate.download._tag === "Downloaded"
        && submittedCandidate.preparation._tag === "Calibrating")
    ? submittedCandidate
    : null
  if (download) return { _tag: "Downloading", state, candidate: download }

  const primary = state.slots.slots.primary
  if (submittedProviderModelId === null
    || primary._tag !== "ConfiguredLocal"
    || primary.selection.providerModelId !== submittedProviderModelId) {
    if (submittedProviderModelId !== null
      && submittedCandidate?.download._tag === "Downloaded") {
      return {
        _tag: "Activating",
        state,
        providerModelId: submittedProviderModelId,
        displayName: submittedCandidate.displayName,
        phase: "Preparing",
        failure: null,
      }
    }
    return { _tag: "Choosing", state }
  }
  const candidate = deriveSelectedLocalModelCandidate(state)
  const selection = buildLocalInferenceSelections(state).find((item) =>
    Option.exists(item.providerModelId, (providerModelId) =>
      providerModelId === primary.selection.providerModelId))
  const lifecycle = Option.map(primary.instance, ({ lifecycle }) => lifecycle)
  const failure = Option.match(lifecycle, {
    onNone: () => primary.readiness._tag === "Unavailable"
      ? primary.readiness.failure.message
      : primary.availability._tag === "Unavailable"
        && primary.availability.failure.code !== "local_model_not_installed"
        ? primary.availability.failure.message
        : null,
    onSome: (instanceLifecycle) => instanceLifecycle._tag === "Failed"
      ? instanceLifecycle.failure.message
      : null,
  })
  const phase = Option.match(lifecycle, {
    onNone: () => failure === null ? "Preparing" as const : "Failed" as const,
    onSome: (instanceLifecycle) => {
      switch (instanceLifecycle._tag) {
        case "Loading": return "Loading" as const
        case "Ready": return "Ready" as const
        case "Failed": return "Failed" as const
        case "Stopping":
        case "Stopped":
          return "Preparing" as const
      }
    },
  })
  return {
    _tag: "Activating",
    state,
    providerModelId: primary.selection.providerModelId,
    displayName: candidate?.displayName
      ?? selection?.model.displayName
      ?? primary.descriptor.displayName,
    phase,
    failure,
  }
}

export const onboardingModelSetupPlaceholder = (view: OnboardingModelSetupView): string | null => {
  switch (view._tag) {
    case "Inactive": return null
    case "Preparing": return "Preparing local models…"
    case "Choosing": return "Select a model to start coding…"
    case "Downloading": return `Downloading ${view.candidate.displayName}…`
    case "Activating":
      return view.phase === "Loading"
        ? `Loading ${view.displayName}…`
        : view.phase === "Ready"
          ? `Finishing setup for ${view.displayName}…`
          : `Preparing ${view.displayName}…`
  }
}
