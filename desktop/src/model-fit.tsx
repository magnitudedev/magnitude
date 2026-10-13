import { formatMemorySize } from "@magnitudedev/client-common"
import type { CatalogLocalModel } from "@magnitudedev/sdk"
import { PlayIcon } from "@phosphor-icons/react"
import { Button } from "../../web/src/components/ui/button"

type AssessedModel = Pick<CatalogLocalModel, "servingState">

/** Memory fit is advice, separate from an attempted operation's failure. */
export const fitNotice = (model: AssessedModel): string | null => {
  const serving = model.servingState
  if (serving._tag === "Assessing") return "Assessing memory and speed…"
  if (serving._tag === "Failed") return "Assessment failed."
  const assessment = serving.assessment
  if (assessment._tag === "DoesNotFit") return `This model needs ${formatMemorySize(assessment.deficitBytes, { rounding: "up" })} more memory than this computer can provide. Choose a smaller model.`
  return null
}

/** Why the model cannot be loaded at all: an assessment that it does not fit this computer. */
export const loadRefusal = (model: AssessedModel): string | null =>
  model.servingState._tag === "Assessed" && model.servingState.assessment._tag === "DoesNotFit" ? fitNotice(model) : null

export function LoadModelButton({ model, pending, onLoad }: { model: AssessedModel; pending: boolean; onLoad: () => void }) {
  const refusal = loadRefusal(model)
  return <Button className="min-w-28" disabled={pending || refusal !== null} title={refusal ?? undefined} onClick={onLoad}><PlayIcon />Load model</Button>
}
