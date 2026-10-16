import { Option } from "effect"
import type {
  CustomLocalModel,
  DrafterCandidate,
  DrafterReference,
  LocalModel,
  MeasuredDrafterAcceptance,
} from "@magnitudedev/sdk"
import { formatStorageSize } from "../utils/format-bytes"
import { unsupportedReasonMessages } from "./failure-messages"

const catalogDisplayName = (models: ReadonlyArray<LocalModel>, reference: DrafterReference): Option.Option<string> =>
  reference._tag === "Catalog"
    ? Option.fromNullable(models.find((model) => model.modelId === reference.modelId)?.presentation.displayName)
    : Option.none()

/** The label a drafter choice carries in the repository step, Details, and the CLI. */
export const drafterCandidateLabel = (
  candidate: DrafterCandidate,
  models: ReadonlyArray<LocalModel>,
): string => {
  switch (candidate.category) {
    case "CatalogDrafter": return Option.match(catalogDisplayName(models, candidate.reference), {
      onNone: () => `Recommended · ${candidate.label}`,
      onSome: (name) => `Recommended · Trained for ${name}`,
    })
    case "BuiltIn": return "Built into this model"
    case "NextToModel": return candidate.reference._tag === "Local" ? "In this folder" : "In this repository"
    case "OnThisComputer": return "On this computer"
    case "HuggingFace": return candidate.reference._tag === "HuggingFace"
      ? candidate.reference.repository.split("/")[0]!
      : candidate.label
    case "Other": return candidate.label
  }
}

/** A drafter can be chosen only when the engine binds it. */
export const drafterCandidateSelectable = (candidate: DrafterCandidate): boolean => candidate.verdict._tag === "Bound"

/** Why a drafter can't be chosen, or the warning a choosable one carries. */
export const drafterCandidateWarning = (candidate: DrafterCandidate): Option.Option<string> => {
  if (candidate.verdict._tag === "Rejected") {
    return Option.some(unsupportedReasonMessages(candidate.verdict.reasons).join(". "))
  }
  return candidate.lineage === "None" ? Option.some("Untested with this model") : Option.none()
}

/** The download a drafter choice adds, or nothing when its files are already here. */
export const drafterCandidateDownloadBytes = (candidate: DrafterCandidate): number =>
  candidate.onThisComputer ? 0 : candidate.sizeBytes

/** "Faster with a drafter · Add (0.9 GB)" for a custom model with no drafter and a recommended one. */
export const customModelDrafterOffer = (
  model: CustomLocalModel,
): Option.Option<{ readonly label: string; readonly candidate: DrafterCandidate }> => {
  if (Option.isSome(model.drafter)) return Option.none()
  return Option.map(model.recommendedDrafter, (candidate) => {
    const bytes = drafterCandidateDownloadBytes(candidate)
    return {
      label: bytes > 0 ? `Faster with a drafter · Add (${formatStorageSize(bytes)})` : "Faster with a drafter · Add",
      candidate,
    }
  })
}

/** "58% of drafted tokens accepted", once the model has served requests with its drafter. */
export const drafterAcceptanceLabel = (acceptance: MeasuredDrafterAcceptance): string =>
  `${Math.round(acceptance.acceptanceRate * 100)}% of drafted tokens accepted`

/** Shown with Change and Turn off when the drafter makes decoding slower than running without it. */
export const drafterSlowingWarning = (model: CustomLocalModel): Option.Option<string> =>
  Option.exists(model.measuredAcceptance, ({ slower }) => slower)
    ? Option.some("This drafter is slowing responses")
    : Option.none()
