import { useState } from "react"
import { Option } from "effect"
import { Result, useAtomValue } from "@effect-atom/atom-react"
import { MagnifyingGlassIcon, PlayIcon, StopIcon } from "@phosphor-icons/react"
import type { CustomFormModelId, LocalModel } from "@magnitudedev/sdk"
import { useAgentClient, useLocalInferenceHardware, useLocalModelMutations, useLocalModels } from "@magnitudedev/client-common"
import { Button } from "../ui/button"
import { ErrorNotice } from "../error-notice"
import { ModelsSkeleton } from "./skeletons"
import { pageLayout } from "../page-layout"
import { cn } from "@/lib/utils"
import { MachineCard } from "./parts"
import { machineView } from "./machine"
import { ownedModel, unrunnableFile } from "./data"
import { YourModels, type OwnedModel } from "./your-models"

const residencyOf = (model: LocalModel) => "residencyState" in model.acquisitionState ? model.acquisitionState.residencyState : undefined

const running = (model: LocalModel) => {
  const residency = residencyOf(model)?._tag
  return residency === "Ready" || residency === "Loading" || residency === "Requested" || residency === "Stopping"
}

/** Models: the machine, then the user's models, with Explore models for finding more. */
export function ModelsScreen({ onExplore }: { onExplore: (query: string) => void }) {
  const client = useAgentClient()
  const models = useLocalModels()
  const hardware = useLocalInferenceHardware()
  const machine = useAtomValue(client.Configuration.GetServerMachine({})).result
  const mutations = useLocalModelMutations()
  const [removed, setRemoved] = useState<ReadonlySet<string>>(new Set())
  if (Result.isFailure(models)) return <><h1 className={pageLayout.pageTitle}>Models</h1><ErrorNotice title="Couldn’t load your models" description="Model information is unavailable. Check the service in Settings." className="mt-5" /></>
  if (!Result.isSuccess(models) || !Result.isSuccess(hardware)) return <ModelsSkeleton />
  const unified = hardware.value.memoryDomains.some(domain => domain.kind === "UnifiedMemory")
  const onComputer = models.value.models.filter(model => !removed.has(model.modelId) && (model._tag === "Custom" || model.acquisitionState._tag !== "NotInstalled"))
  const byId = new Map<string, LocalModel>(onComputer.map(model => [model.modelId, model]))
  const owned = onComputer.map(model => ownedModel(model, unified))
  const action = (view: OwnedModel) => {
    const model = byId.get(view.modelId)
    if (!model) return null
    const state = model.acquisitionState
    const progress = state._tag === "Installing" || state._tag === "Updating" ? state.progress : undefined
    if (progress) return <span className="flex w-28 flex-col items-end gap-1"><span className="text-xs tabular-nums text-slate-500">Downloading {progress.totalBytes > 0 ? Math.round(progress.completedBytes / progress.totalBytes * 100) : 0}%</span>
      <span className="h-1 w-full overflow-hidden rounded-full bg-slate-150 dark:bg-slate-750"><span className="block h-full bg-blue-600 dark:bg-blue-400" style={{ width: `${progress.totalBytes > 0 ? progress.completedBytes / progress.totalBytes * 100 : 0}%` }} /></span></span>
    return running(model)
      ? <Button size="sm" variant="outline" onClick={() => mutations.stop()}><StopIcon />Stop</Button>
      : <Button size="sm" variant="outline" onClick={() => mutations.load(model.modelId)}><PlayIcon />Load</Button>
  }
  const remove = (view: OwnedModel) => {
    const model = byId.get(view.modelId)
    if (!model) return
    if (model._tag === "Custom" && model.origin._tag === "Found") mutations.hide(model.modelId)
    else mutations.remove(model.modelId)
    setRemoved(current => new Set([...current, view.modelId]))
  }
  const addDrafter = (view: OwnedModel) => {
    const model = byId.get(view.modelId)
    if (model?._tag !== "Custom") return
    Option.match(model.recommendedDrafter, {
      onNone: () => mutations.setDrafter(model.modelId as CustomFormModelId, { _tag: "Auto" }),
      onSome: drafter => mutations.setDrafter(model.modelId as CustomFormModelId, { _tag: "Reference", reference: drafter.reference }),
    })
  }
  return <div>
    <div className="mb-5 flex items-end justify-between gap-4"><h1 className={pageLayout.pageTitle}>Models</h1><Button onClick={() => onExplore("")}><MagnifyingGlassIcon />Explore models</Button></div>
    <MachineCard machine={machineView(hardware.value, Result.isSuccess(machine) ? machine.value.identity : null)} />
    {owned.length === 0 && models.value.unsupported.length === 0
      ? <div className={cn("mt-8 rounded-xl border border-slate-200 bg-white px-5 py-6 text-[13px] text-slate-500 dark:border-slate-750 dark:bg-slate-850")}>No models yet. <button type="button" onClick={() => onExplore("")} className="text-blue-700 hover:underline dark:text-blue-400">Explore models</button> to download one.</div>
      : <YourModels models={owned} unrunnable={models.value.unsupported.map(unrunnableFile)} action={action}
        onExploreQuants={view => onExplore(view.name)} onRemove={remove} onAddDrafter={addDrafter} />}
  </div>
}
