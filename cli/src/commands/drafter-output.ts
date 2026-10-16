import { Option } from "effect"
import {
  drafterCandidateLabel,
  formatSpeculativeMethod,
  formatStorageSize,
  unsupportedReasonMessages,
} from "@magnitudedev/client-common"
import type { DrafterCandidate, DrafterReference, LocalModel } from "@magnitudedev/sdk"
import { drafterReferenceArgument } from "./model-references"
import { renderTable } from "./output"

interface DrafterRow {
  readonly selected: boolean
  readonly name: string
  readonly size: string
  readonly verdict: string
  readonly accepted: string
  readonly note: string
  readonly use: string
}

const drafterSize = (candidate: DrafterCandidate): string => candidate.reference._tag === "Embedded"
  ? "Included"
  : candidate.onThisComputer
    ? `${formatStorageSize(candidate.sizeBytes)}, no download`
    : formatStorageSize(candidate.sizeBytes)

export const drafterVerdictLabel = (candidate: DrafterCandidate): string => {
  switch (candidate.verdict._tag) {
    case "Bound": return `Works · ${formatSpeculativeMethod(candidate.verdict.method)}`
    case "Rejected": return `Can't be used: ${unsupportedReasonMessages(candidate.verdict.reasons).join(". ")}`
    case "None": return "Not checked"
  }
}

/** The measured acceptance rate once the model has served requests with this candidate. */
export const drafterAcceptedLabel = (candidate: DrafterCandidate): string => Option.match(candidate.measuredAcceptance, {
  onNone: () => "—",
  onSome: ({ acceptanceRate }) => `${Math.round(acceptanceRate * 100)}%`,
})

const drafterNote = (candidate: DrafterCandidate, models: readonly LocalModel[]): string => {
  const label = drafterCandidateLabel(candidate, models)
  const notes = [
    label,
    ...(candidate.lineage === "None" && candidate.verdict._tag === "Bound" ? ["untested with this model"] : []),
    ...(Option.exists(candidate.measuredAcceptance, ({ slower }) => slower) ? ["slower than no drafter here"] : []),
  ]
  return notes.join(" · ")
}

/**
 * The drafter choices in selection order, with `*` on the chosen one and a final "None" row. `chosen`
 * is the reference in use, or none when the model runs without a drafter.
 */
export const renderDrafterChoices = (
  candidates: readonly DrafterCandidate[],
  models: readonly LocalModel[],
  chosen: Option.Option<DrafterReference>,
): string => {
  const chosenArgument = Option.map(chosen, drafterReferenceArgument)
  const rows: DrafterRow[] = [
    ...candidates.map((candidate) => ({
      selected: Option.contains(chosenArgument, drafterReferenceArgument(candidate.reference)),
      name: candidate.label,
      size: drafterSize(candidate),
      verdict: drafterVerdictLabel(candidate),
      accepted: drafterAcceptedLabel(candidate),
      note: drafterNote(candidate, models),
      use: drafterReferenceArgument(candidate.reference),
    })),
    { selected: Option.isNone(chosen), name: "None", size: "—", verdict: "—", accepted: "—", note: "Run without a drafter", use: "none" },
  ]
  return renderTable(rows, [
    { heading: "DRAFTER", value: ({ selected, name }) => `${selected ? "*" : " "} ${name}` },
    { heading: "SIZE", value: ({ size }) => size },
    { heading: "VERDICT", value: ({ verdict }) => verdict },
    { heading: "ACCEPTED", value: ({ accepted }) => accepted },
    { heading: "NOTE", value: ({ note }) => note },
    { heading: "USE", value: ({ use }) => use },
  ]).trimEnd()
}
