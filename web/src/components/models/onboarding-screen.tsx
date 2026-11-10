import { useState } from "react"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import type { CatalogFormModelId, CustomLocalModel } from "@magnitudedev/sdk"
import { useAgentClient, useLocalInferenceHardware, useLocalModelMutations, useLocalModels } from "@magnitudedev/client-common"
import { useSession } from "../../session"
import { discoveredModel, unrunnableFile } from "./data"
import { machineView } from "./machine"
import { Onboarding, type Recommendation } from "./onboarding"
import { useRecommendations } from "./recommendations"

/** First run on the service's data: the machine, models already found, and the top recommendations that run well here. */
export function OnboardingScreen() {
  const client = useAgentClient()
  const session = useSession()
  const complete = useAtomSet(session.completeOnboarding)
  const hardware = useLocalInferenceHardware()
  const machine = useAtomValue(client.Configuration.GetServerMachine({})).result
  const models = useLocalModels()
  const { install } = useLocalModelMutations()
  const [usOnly, setUsOnly] = useState(false)
  const unified = Result.isSuccess(hardware) && hardware.value.memoryDomains.some(domain => domain.kind === "UnifiedMemory")
  const recommendations = useRecommendations(usOnly, unified)
  const local = Result.isSuccess(models) ? models.value : null
  const found = (local?.models ?? []).filter((model): model is CustomLocalModel => model._tag === "Custom" && model.origin._tag === "Found")
  const owned = new Set((local?.models ?? []).filter(model => model._tag === "Custom" || model.acquisitionState._tag !== "NotInstalled").map(model => String(model.modelId)))
  const measuring = !recommendations.ready || local === null || !local.preparation.assessment.complete
  const eligible = recommendations.offered.filter(entry => entry.recommendation.meetsOnboardingFloor).slice(0, 3)
  const fastest = eligible.reduce<typeof eligible[number] | undefined>((best, entry) => {
    const speed = (value: typeof entry) => value.model.quants.find(quant => quant.key === value.model.recommended)?.speed ?? 0
    return best && speed(best) >= speed(entry) ? best : entry
  }, undefined)
  const offered: readonly Recommendation[] = eligible.map((entry, index) => {
    const have = entry.model.quants.find(quant => owned.has(quant.key))
    return { model: entry.model, note: have ? `You have the ${have.bits}-bit quant` : index === 0 ? "Smartest that runs well here" : entry === fastest ? "Fastest" : undefined }
  })
  return <Onboarding
    machine={Result.isSuccess(hardware) ? machineView(hardware.value, Result.isSuccess(machine) ? machine.value.identity : null) : null}
    measuring={measuring}
    discovered={found.flatMap(model => { const view = discoveredModel(model); return view ? [view] : [] })}
    unrunnable={(local?.unsupported ?? []).map(unrunnableFile)}
    recommendations={offered}
    usOnly={usOnly}
    setUsOnly={setUsOnly}
    owned={owned}
    onDownload={(_, quant) => install(quant.key as CatalogFormModelId)}
    onFinish={() => complete()} />
}
