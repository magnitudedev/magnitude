import { useState } from "react"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import type { CatalogFormModelId, CustomLocalModel } from "@magnitudedev/sdk"
import { deriveHardwareRoles, formatStorageSize, useAgentClient, useLocalInferenceHardware, useLocalModelMutations, useLocalModels } from "@magnitudedev/client-common"
import { useSession } from "../../session"
import { useServiceObservation } from "../../service-view"
import { discoveredModel, onThisComputer, unrunnableFile } from "./data"
import { machineView } from "./machine"
import { Onboarding, preparationLabels, type PreparationStep } from "./onboarding"
import { recordOnboardingCompleted } from "./onboarding-record"
import { recommendationsSettled, recommendedForYourMachine, useRecommendations } from "./recommendations"

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
  const memory = Result.isSuccess(hardware) ? deriveHardwareRoles(hardware.value).servingMemory : "Device"
  const recommendations = useRecommendations(usOnly, memory)
  const local = Result.isSuccess(models) ? models.value : null
  const found = (local?.models ?? []).filter((model): model is CustomLocalModel => model._tag === "Custom" && model.origin._tag === "Found")
  const owned = new Set((local?.models ?? []).filter(onThisComputer).map(model => String(model.modelId)))
  const offered = recommendedForYourMachine(recommendations.offered, owned)
  const observation = useServiceObservation()
  const preparation = local?.preparation
  const plural = (count: number, noun: string) => `${count} ${noun}${count === 1 ? "" : "s"}`
  const progress: readonly { label: string; done: boolean; detail: string | null; fraction: number | null }[] = [
    { label: preparationLabels[0], done: Result.isSuccess(observation.service) && observation.service.value._tag === "Ready", detail: null, fraction: null },
    { label: preparationLabels[1], done: Result.isSuccess(hardware), detail: null, fraction: null },
    { label: preparationLabels[2], done: preparation?.discovery.complete ?? false,
      detail: preparation && (preparation.discovery.complete || found.length > 0) ? `${plural(found.length, "model")} found` : null, fraction: null },
    // The models are offered once every quant has been measured, so the list never changes under the reader.
    { label: preparationLabels[3], done: recommendationsSettled(recommendations),
      detail: recommendations.measuring && recommendations.measuring.total > 0 ? `${recommendations.measuring.settled} of ${plural(recommendations.measuring.total, "quant")}` : null,
      fraction: recommendations.measuring && recommendations.measuring.total > 0 ? recommendations.measuring.settled / recommendations.measuring.total : null },
  ]
  const active = progress.findIndex(step => !step.done)
  const steps: readonly PreparationStep[] | null = active === -1 ? null
    : progress.map((step, index) => ({ label: step.label, detail: step.detail, fraction: step.fraction, state: step.done ? "done" : index === active ? "active" : "pending" }))
  const measuring = steps !== null
  return <Onboarding
    machine={Result.isSuccess(hardware) ? machineView(hardware.value, Result.isSuccess(machine) ? machine.value.identity : null) : null}
    preparation={steps}
    discovered={found.flatMap(model => { const view = discoveredModel(model); return view ? [view] : [] })}
    unrunnable={(local?.unsupported ?? []).map(unrunnableFile)}
    recommendations={offered}
    roomNote={recommendations.roomShortfall && !measuring
      ? `${recommendations.roomShortfall.name} would be the top pick here with ${formatStorageSize(recommendations.roomShortfall.missingBytes)} more free disk space.`
      : null}
    usOnly={usOnly}
    setUsOnly={setUsOnly}
    owned={owned}
    onDownload={(_, quant) => install(quant.key as CatalogFormModelId)}
    onFinish={() => { recordOnboardingCompleted(true); complete() }} />
}
