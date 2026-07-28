import { Option } from "effect"
import {
  deriveSelectedLocalModelCandidate,
  deriveSelectedLocalModelSetup,
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
      readonly _tag: "Loading"
      readonly state: LocalInferenceView
      readonly providerModelId: ProviderModelId
      readonly displayName: string
      readonly failure: string | null
    }

/** Keeps forced setup mounted until its exact server-resolved selection is authoritatively ready. */
export const deriveModelSetupActive = ({
  forceSetup,
  onboardingRequired,
  completionSucceeded,
  selectedProviderModelId,
  primary,
}: {
  readonly forceSetup: boolean
  readonly onboardingRequired: boolean
  readonly completionSucceeded: boolean
  readonly selectedProviderModelId: ProviderModelId | null
  readonly primary: LocalInferenceView["slots"]["slots"]["primary"] | null
}): boolean => {
  if (onboardingRequired) return true
  if (!forceSetup || completionSucceeded) return false
  return selectedProviderModelId === null
    || primary?._tag !== "Ready"
    || primary.selection.providerModelId !== selectedProviderModelId
}

const blockedReasonMessage = (
  reason: Extract<
    LocalInferenceView["slots"]["slots"]["primary"],
    { readonly _tag: "Blocked" }
  >["reason"],
): string => "message" in reason ? reason.message : reason.error.message

export const deriveOnboardingModelSetupView = ({
  active,
  onboardingRequired,
  state,
}: {
  readonly active: boolean
  readonly onboardingRequired: boolean
  readonly state: LocalInferenceView | null
}): OnboardingModelSetupView => {
  if (!active) return { _tag: "Inactive" }
  if (state === null || localInferenceSetupPhase(state) !== "ready") {
    return { _tag: "Preparing", state }
  }
  if (!onboardingRequired) return { _tag: "Choosing", state }

  const download = deriveSelectedLocalModelSetup(state)
  if (download) return { _tag: "Downloading", state, candidate: download }

  const primary = state.slots.slots.primary
  if (primary._tag === "Unassigned" || primary._tag === "Ready") {
    return { _tag: "Choosing", state }
  }
  const candidate = deriveSelectedLocalModelCandidate(state)
  const selection = buildLocalInferenceSelections(state).find((item) =>
    Option.exists(item.providerModelId, (providerModelId) =>
      providerModelId === primary.selection.providerModelId))
  return {
    _tag: "Loading",
    state,
    providerModelId: primary.selection.providerModelId,
    displayName: candidate?.displayName ?? selection?.model.displayName ?? "local model",
    failure: primary._tag === "Blocked" && primary.reason._tag !== "ModelUnavailable"
      ? blockedReasonMessage(primary.reason)
      : null,
  }
}

export const onboardingModelSetupPlaceholder = (view: OnboardingModelSetupView): string | null => {
  switch (view._tag) {
    case "Inactive": return null
    case "Preparing": return "Preparing local models…"
    case "Choosing": return "Select a model to start coding…"
    case "Downloading": return `Downloading ${view.candidate.displayName}…`
    case "Loading": return `Loading ${view.displayName}…`
  }
}
