import { Option } from "effect"
import {
  deriveSelectedLocalModelCandidate,
} from "@magnitudedev/client-common"
import type {
  LocalModelCatalogCandidate,
  LocalModelsState,
  ModelSlotsState,
  ProviderModelCatalogState,
  ProviderModelId,
} from "@magnitudedev/sdk"
import {
  buildLocalInferenceSelections,
} from "../local-inference/view-model"

export type OnboardingModelSetupView =
  | { readonly _tag: "Inactive" }
  | { readonly _tag: "Choosing" }
  | {
      readonly _tag: "Downloading"
      readonly candidate: LocalModelCatalogCandidate
    }
  | {
      readonly _tag: "DownloadFailed"
      readonly candidate: LocalModelCatalogCandidate
    }
  | {
      readonly _tag: "Activating"
      readonly providerModelId: ProviderModelId
      readonly displayName: string
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
  submittedProviderModelId,
  models,
  catalog,
  slots,
}: {
  readonly active: boolean
  readonly submittedProviderModelId: ProviderModelId | null
  readonly models: LocalModelsState
  readonly catalog: ProviderModelCatalogState
  readonly slots: ModelSlotsState
}): OnboardingModelSetupView => {
  if (!active) return { _tag: "Inactive" }

  const submittedCandidate = submittedProviderModelId === null
    || models.recommendations._tag !== "Ready"
    ? null
    : models.recommendations.catalog.find(({ providerModelId }) =>
        providerModelId === submittedProviderModelId) ?? null
  if (submittedCandidate?.download._tag === "Failed") {
    return { _tag: "DownloadFailed", candidate: submittedCandidate }
  }
  if (submittedCandidate?.download._tag === "Downloading"
    || submittedCandidate?.download._tag === "Downloaded"
      && submittedCandidate.preparation._tag === "Calibrating") {
    return { _tag: "Downloading", candidate: submittedCandidate }
  }

  const primary = slots.slots.primary
  if (submittedProviderModelId === null
    || primary._tag !== "ConfiguredLocal"
    || primary.selection.providerModelId !== submittedProviderModelId) {
    return { _tag: "Choosing" }
  }
  const candidate = deriveSelectedLocalModelCandidate(models, slots)
  const selection = buildLocalInferenceSelections(models, catalog, slots).find((item) =>
    Option.exists(item.providerModelId, (providerModelId) =>
      providerModelId === primary.selection.providerModelId))
  const lifecycle = Option.getOrNull(Option.map(
    primary.instance,
    ({ lifecycle }) => lifecycle,
  ))
  if (lifecycle === null
    || lifecycle._tag === "Stopping"
    || lifecycle._tag === "Stopped") {
    return { _tag: "Choosing" }
  }
  return {
    _tag: "Activating",
    providerModelId: primary.selection.providerModelId,
    displayName: candidate?.displayName
      ?? selection?.model.displayName
      ?? primary.descriptor.displayName,
    phase: lifecycle._tag,
    failure: lifecycle._tag === "Failed" ? lifecycle.failure.message : null,
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
