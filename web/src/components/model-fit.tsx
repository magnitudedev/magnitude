import { formatContext, formatMemorySize, unsupportedReasonMessages } from "@magnitudedev/client-common"
import type { LocalModel } from "@magnitudedev/sdk"
import { PlayIcon } from "@phosphor-icons/react"
import { Button } from "./ui/button"

type AssessedModel = Pick<LocalModel, "servingState">

/** Memory fit is advice, separate from an attempted operation's failure. */
export const fitNotice = (model: AssessedModel): string | null => {
  const serving = model.servingState
  if (serving._tag === "Assessing") return "Assessing memory and speed…"
  if (serving._tag === "Failed") return "Assessment failed."
  const assessment = serving.assessment
  if (assessment._tag === "DoesNotFit") return `This model needs ${formatMemorySize(assessment.deficitBytes, { rounding: "up" })} more memory than this computer can provide, even for a ${formatContext(assessment.shortfall.contextLength)} context. Choose a smaller model.`
  if (assessment._tag === "Unsupported") return `Magnitude can’t run this model. ${unsupportedReasonMessages(assessment.reasons).join(". ")}.`
  return null
}

/**
 * The context a fitting model is served at, when this computer has memory for less than the
 * model's own: "64K of 256K context".
 */
export const servedContextNote = (model: AssessedModel): string | null => {
  const serving = model.servingState
  if (serving._tag !== "Assessed" || serving.assessment._tag !== "Fits") return null
  const { served, profile } = serving.assessment
  return served.contextLength < profile.contextLength
    ? `${formatContext(served.contextLength)} of ${formatContext(profile.contextLength)} context`
    : null
}

/** Why the model cannot be loaded at all: an assessment that it does not fit this computer, or can't run. */
export const loadRefusal = (model: AssessedModel): string | null =>
  model.servingState._tag === "Assessed" && model.servingState.assessment._tag !== "Fits" ? fitNotice(model) : null

export function LoadModelButton({ model, pending, onLoad }: { model: AssessedModel; pending: boolean; onLoad: () => void }) {
  const refusal = loadRefusal(model)
  return <Button className="min-w-28" disabled={pending || refusal !== null} title={refusal ?? undefined} onClick={onLoad}><PlayIcon />Load model</Button>
}
