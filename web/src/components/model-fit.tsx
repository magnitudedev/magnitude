import { formatMemorySize } from "@magnitudedev/client-common"
import type { CatalogLocalModel } from "@magnitudedev/sdk"
import { PlayIcon } from "@phosphor-icons/react"
import { Button } from "./ui/button"
import { readLanguage, translate } from "../i18n"

const tr = (key: Parameters<typeof translate>[1]) => translate(readLanguage(), key)

type AssessedModel = Pick<CatalogLocalModel, "servingState">

/** Memory fit is advice, separate from an attempted operation's failure. */
export const fitNotice = (model: AssessedModel): string | null => {
  const serving = model.servingState
  if (serving._tag === "Assessing") return tr("assessingMemorySpeed")
  if (serving._tag === "Failed") return tr("assessmentFailed")
  const assessment = serving.assessment
  if (assessment._tag === "DoesNotFit") return tr("modelNeedsMoreMemory").replace("{size}", formatMemorySize(assessment.deficitBytes, { rounding: "up" }))
  return null
}

/** Why the model cannot be loaded at all: an assessment that it does not fit this computer. */
export const loadRefusal = (model: AssessedModel): string | null =>
  model.servingState._tag === "Assessed" && model.servingState.assessment._tag === "DoesNotFit" ? fitNotice(model) : null

export function LoadModelButton({ model, pending, onLoad }: { model: AssessedModel; pending: boolean; onLoad: () => void }) {
  const refusal = loadRefusal(model)
  return <Button className="min-w-28" disabled={pending || refusal !== null} title={refusal ?? undefined} onClick={onLoad}><PlayIcon /><span>{tr("loadModel")}</span></Button>
}
