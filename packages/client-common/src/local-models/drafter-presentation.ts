import { Data, Equal, Option } from "effect"
import type {
  CandidateFile,
  CandidateVersion,
  CustomLocalModel,
  DrafterCandidate,
} from "@magnitudedev/sdk"
import { formatStorageSize } from "../utils/format-bytes"
import { unsupportedReasonMessages } from "./failure-messages"

/**
 * The note a drafter choice carries in the repository step, Details, and the CLI: ICN's label for its
 * source ("Trained for Qwen3.5 4B", "In this repository", "Built into this model", or "Untested" for a
 * drafter the user entered whose lineage doesn't match), with "Recommended · " on the preselected
 * choice only.
 */
export const drafterCandidateLabel = (candidate: DrafterCandidate, recommended: boolean): string =>
  recommended ? `Recommended · ${candidate.label}` : candidate.label

/** A drafter can be chosen only when the engine binds it. */
export const drafterCandidateSelectable = (candidate: DrafterCandidate): boolean => candidate.verdict._tag === "Bound"

/** Why a drafter can't be chosen; an untested one says so in its label instead. */
export const drafterCandidateWarning = (candidate: DrafterCandidate): Option.Option<string> =>
  candidate.verdict._tag === "Rejected"
    ? Option.some(unsupportedReasonMessages(candidate.verdict.reasons).join(". "))
    : Option.none()

/** The download a drafter choice adds, or nothing when its files are already here. */
export const drafterCandidateDownloadBytes = (candidate: DrafterCandidate): number =>
  candidate.onThisComputer ? 0 : candidate.sizeBytes

export interface CandidateDownload {
  readonly totalBytes: number
  readonly modelBytes: number
  readonly projectorBytes: number
  readonly drafterBytes: number
}

/**
 * Everything adding a candidate version downloads: its target files and repository projector (the
 * version's `downloadBytes`, zero when already here) plus the chosen drafter unless it is here too.
 */
export const candidateDownloadBytes = (
  version: CandidateVersion,
  projector: Option.Option<CandidateFile>,
  drafter: Option.Option<DrafterCandidate>,
): CandidateDownload => {
  const drafterBytes = Option.match(drafter, { onNone: () => 0, onSome: drafterCandidateDownloadBytes })
  const projectorBytes = Option.match(projector, { onNone: () => 0, onSome: ({ sizeBytes }) => sizeBytes })
  const projectorPart = Math.min(projectorBytes, version.downloadBytes)
  return {
    totalBytes: version.downloadBytes + drafterBytes,
    modelBytes: version.downloadBytes - projectorPart,
    projectorBytes: projectorPart,
    drafterBytes,
  }
}

/**
 * "Faster with a drafter · Add (0.9 GB)" for a custom model whose recommended drafter isn't attached: none is, or,
 * under automatic choice, a lesser one on this computer serves meanwhile (its built-in heads, say). Never once its
 * drafter was turned off or chosen.
 */
export const customModelDrafterOffer = (
  model: CustomLocalModel,
): Option.Option<{ readonly label: string; readonly candidate: DrafterCandidate }> => {
  if (model.drafterChoice._tag === "None") return Option.none()
  if (Option.isSome(model.drafter) && model.drafterChoice._tag === "Reference") return Option.none()
  const offered = Option.filter(model.recommendedDrafter, (candidate) => Option.match(model.drafter, {
    onNone: () => true,
    onSome: (attached) => !candidate.onThisComputer && !Equal.equals(Data.struct(attached.reference), Data.struct(candidate.reference)),
  }))
  return Option.map(offered, (candidate) => {
    const bytes = drafterCandidateDownloadBytes(candidate)
    return {
      label: bytes > 0 ? `Faster with a drafter · Add (${formatStorageSize(bytes)})` : "Faster with a drafter · Add",
      candidate,
    }
  })
}
