import { formatMemorySize, unsupportedReasonMessages } from "@magnitudedev/client-common"
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
  if (assessment._tag === "DoesNotFit") return `This model needs ${formatMemorySize(assessment.deficitBytes, { rounding: "up" })} more memory than this computer can provide. Choose a smaller model.`
  if (assessment._tag === "Unsupported") return `Magnitude can’t run this model. ${unsupportedReasonMessages(assessment.reasons).join(". ")}.`
  return null
}

/** Why the model cannot be loaded at all: an assessment that it does not fit this computer, or can't run. */
export const loadRefusal = (model: AssessedModel): string | null =>
  model.servingState._tag === "Assessed" && model.servingState.assessment._tag !== "Fits" ? fitNotice(model) : null

export function LoadModelButton({ model, pending, onLoad }: { model: AssessedModel; pending: boolean; onLoad: () => void }) {
  const refusal = loadRefusal(model)
  return <Button className="min-w-28" disabled={pending || refusal !== null} title={refusal ?? undefined} onClick={onLoad}><PlayIcon />Load model</Button>
}
