import { useState, type ReactNode } from "react"
import { Option } from "effect"
import { Result, useAtomValue } from "@effect-atom/atom-react"
import { CircleNotchIcon, MagnifyingGlassIcon, PlayIcon, StopIcon } from "@phosphor-icons/react"
import type { CustomFormModelId, LocalModel } from "@magnitudedev/sdk"
import { deriveHardwareRoles, modelDownloadFailureMessage, useAgentClient, useLocalInferenceHardware, useLocalModelMutations, useLocalModels } from "@magnitudedev/client-common"
import { Button } from "../ui/button"
import { ErrorNotice } from "../error-notice"
import { ModelsSkeleton } from "./skeletons"
import { pageLayout } from "../page-layout"
import { cn } from "@/lib/utils"
import { fitLabel, MachineCard } from "./parts"
import { machineView } from "./machine"
import { ownedModel, unrunnableFile } from "./data"
import { YourModels, type OwnedModel } from "./your-models"

const residencyOf = (model: LocalModel) => "residencyState" in model.acquisitionState ? model.acquisitionState.residencyState : undefined

const transferStage = { queued: "Queued", resolving: "Preparing", checking_space: "Checking space", downloading: "Downloading", verifying: "Verifying", publishing: "Finishing" } as const

function Status({ tone = "muted", busy = false, title, children }: { tone?: "muted" | "running" | "failed"; busy?: boolean; title?: string; children: ReactNode }) {
  return <span title={title} className={cn("inline-flex items-center gap-1.5 whitespace-nowrap text-xs", tone === "failed" ? "text-red-600 dark:text-red-400" : tone === "running" ? "text-green-700 dark:text-green-400" : "text-slate-500")}>
    {busy ? <CircleNotchIcon className="size-3.5 motion-safe:animate-spin" /> : tone === "running" && <span className="size-1.5 rounded-full bg-green-600 dark:bg-green-400" />}
    {children}
  </span>
}

/** What finds a model's other quants on Hugging Face: its repository's name without owner or `-GGUF`, else its name. */
const quantSearch = (view: OwnedModel) => {
  const repository = view.quant.repository?.split("/").pop()?.replace(/[-_.]?gguf$/i, "")
  return repository && repository.length > 0 ? repository : view.name
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
  const memory = deriveHardwareRoles(hardware.value).servingMemory
  const onComputer = models.value.models.filter(model => !removed.has(model.modelId) && (model._tag === "Custom" || model.acquisitionState._tag !== "NotInstalled"))
  const byId = new Map<string, LocalModel>(onComputer.map(model => [model.modelId, model]))
  const owned = onComputer.map(model => ownedModel(model, memory))
  const hidden = models.value.hidden.map(model => ownedModel(model, memory))
  const action = (view: OwnedModel) => {
    const model = byId.get(view.modelId)
    if (!model) return null
    const state = model.acquisitionState
    const stop = <Button size="sm" variant="outline" onClick={() => mutations.stop()}><StopIcon />Stop</Button>
    const load = <Button size="sm" variant="outline" onClick={() => mutations.load(model.modelId)}><PlayIcon />Load</Button>
    if (state._tag === "Installing" || state._tag === "Updating") {
      const { progress } = state
      const fraction = progress.totalBytes > 0 ? progress.completedBytes / progress.totalBytes : 0
      return <span className="flex items-center gap-2">
        <span className="flex w-28 flex-col items-end gap-1">
          <span className="text-xs tabular-nums text-slate-500">{transferStage[progress.stage]}{progress.stage === "downloading" ? ` ${Math.round(fraction * 100)}%` : "…"}</span>
          <span className="h-1 w-full overflow-hidden rounded-full bg-slate-150 dark:bg-slate-750"><span className="block h-full bg-blue-600 dark:bg-blue-400" style={{ width: `${fraction * 100}%` }} /></span>
        </span>
        <Button size="sm" variant="ghost" onClick={() => mutations.cancel(model.modelId)}>Cancel</Button>
      </span>
    }
    if (state._tag === "Optimizing") return <Status busy>Optimizing…</Status>
    if (state._tag === "InstallFailed" || state._tag === "UpdateFailed") {
      return <span className="flex items-center gap-2">
        <Status tone="failed" title={modelDownloadFailureMessage(state.failure)}>Download failed</Status>
        <Button size="sm" variant="outline" onClick={() => mutations.install(model.modelId)}>Retry</Button>
      </span>
    }
    if (state._tag === "Unavailable") return <Status tone="failed" title={state.failure.message}>Files unavailable</Status>
    if (!view.quant.fits) return <Status busy={view.quant.measuring}>{fitLabel(view.quant)}</Status>
    const residency = residencyOf(model)
    switch (residency?._tag) {
      case "Requested": return <span className="flex items-center gap-2"><Status busy>Loading…</Status>{stop}</span>
      case "Loading": return <span className="flex items-center gap-2"><Status busy>Loading {Math.round(residency.fraction * 100)}%</Status>{stop}</span>
      case "Ready": return <span className="flex items-center gap-2"><Status tone="running">Running</Status>{stop}</span>
      case "Stopping": return <Status busy>Stopping…</Status>
      case "Failed": return <span className="flex items-center gap-2"><Status tone="failed" title={residency.failure.message}>Couldn’t load</Status>{load}</span>
      default: return load
    }
  }
  const remove = (view: OwnedModel) => {
    const model = byId.get(view.modelId)
    if (!model) return
    if (model._tag === "Custom" && model.origin._tag === "Found") mutations.hide(model.modelId)
    else mutations.remove(model.modelId)
    setRemoved(current => new Set([...current, view.modelId]))
  }
  const unhide = (view: OwnedModel) => {
    const model = models.value.hidden.find(hiddenModel => hiddenModel.modelId === view.modelId)
    if (!model) return
    mutations.unhide(model.modelId)
    setRemoved(current => new Set([...current].filter(modelId => modelId !== view.modelId)))
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
    {owned.length === 0 && hidden.length === 0 && models.value.unsupported.length === 0
      ? <div className={cn("mt-8 rounded-xl border border-slate-200 bg-white px-5 py-6 text-[13px] text-slate-500 dark:border-slate-750 dark:bg-slate-850")}>No models yet. <button type="button" onClick={() => onExplore("")} className="text-blue-700 hover:underline dark:text-blue-400">Explore models</button> to download one.</div>
      : <YourModels models={owned} hidden={hidden} unrunnable={models.value.unsupported.map(unrunnableFile)} action={action}
        onExploreQuants={view => onExplore(quantSearch(view))} onRemove={remove} onUnhide={unhide} onAddDrafter={addDrafter} />}
  </div>
}
