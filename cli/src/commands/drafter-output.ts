import { Option } from "effect"
import {
  drafterCandidateLabel,
  formatSpeculativeMethod,
  formatStorageSize,
  preselectedDrafterCandidate,
  unsupportedReasonMessages,
} from "@magnitudedev/client-common"
import type { DrafterCandidate, DrafterReference } from "@magnitudedev/sdk"
import { drafterReferenceArgument } from "./model-references"
import { renderTable } from "./output"

interface DrafterRow {
  readonly selected: boolean
  readonly name: string
  readonly size: string
  readonly verdict: string
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

/**
 * The drafter choices in selection order, with `*` on the chosen one and a final "None" row. `chosen`
 * is the reference in use, or none when the model runs without a drafter; the preselected choice's
 * note reads "Recommended".
 */
export const renderDrafterChoices = (
  candidates: readonly DrafterCandidate[],
  chosen: Option.Option<DrafterReference>,
): string => {
  const chosenArgument = Option.map(chosen, drafterReferenceArgument)
  const recommended = Option.map(preselectedDrafterCandidate(candidates), ({ reference }) => drafterReferenceArgument(reference))
  const rows: DrafterRow[] = [
    ...candidates.map((candidate) => ({
      selected: Option.contains(chosenArgument, drafterReferenceArgument(candidate.reference)),
      name: candidate.name,
      size: drafterSize(candidate),
      verdict: drafterVerdictLabel(candidate),
      note: drafterCandidateLabel(candidate, Option.contains(recommended, drafterReferenceArgument(candidate.reference))),
      use: drafterReferenceArgument(candidate.reference),
    })),
    { selected: Option.isNone(chosen), name: "None", size: "—", verdict: "—", note: "Run without a drafter", use: "none" },
  ]
  return renderTable(rows, [
    { heading: "DRAFTER", value: ({ selected, name }) => `${selected ? "*" : " "} ${name}` },
    { heading: "SIZE", value: ({ size }) => size },
    { heading: "VERDICT", value: ({ verdict }) => verdict },
    { heading: "NOTE", value: ({ note }) => note },
    { heading: "USE", value: ({ use }) => use },
  ]).trimEnd()
}
