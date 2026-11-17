import { useState, type ReactNode } from "react"
import { Result } from "@effect-atom/atom-react"
import { CircleNotchIcon, PlayIcon, StopIcon } from "@phosphor-icons/react"
import type { CustomFormModelId, LocalModel } from "@magnitudedev/sdk"
import { deriveHardwareRoles, modelDownloadFailureMessage, modelFailureLabel, useLocalInferenceHardware, useLocalModelMutations, useLocalModels } from "@magnitudedev/client-common"
import { Button } from "../ui/button"
import { ErrorNotice } from "../error-notice"
import { ModelsSkeleton } from "./skeletons"
import { cn } from "@/lib/utils"
import { fitLabel } from "./parts"
import { ownedModel, unrunnableFile } from "./data"
import { Hidden, Unrunnable, YourModels, type OwnedModel, type UnrunnableFile } from "./your-models"
import { DrafterDialog } from "./drafter-picker"

const residencyOf = (model: LocalModel) => "residencyState" in model.acquisitionState ? model.acquisitionState.residencyState : undefined

const transferStage = { queued: "Queued", resolving: "Preparing", checking_space: "Checking space", downloading: "Downloading", verifying: "Verifying", publishing: "Finishing" } as const

function Status({ tone = "muted", busy = false, title, children }: { tone?: "muted" | "running" | "failed"; busy?: boolean; title?: string; children: ReactNode }) {
  return <span title={title} className={cn("inline-flex items-center gap-1.5 whitespace-nowrap text-xs", tone === "failed" ? "text-red-600 dark:text-red-400" : tone === "running" ? "text-green-700 dark:text-green-400" : "text-slate-500")}>
    {busy ? <CircleNotchIcon className="size-3.5 motion-safe:animate-spin" /> : tone === "running" && <span className="size-1.5 rounded-full bg-green-600 dark:bg-green-400" />}
    {children}
  </span>
}

/**
 * What finds a model's other quants on Hugging Face: its repository's name without owner or `-GGUF`, else its name,
 * led by the repository it came from.
 */
const quantSearch = (view: OwnedModel): { readonly query: string; readonly first: string | null } => {
  const repository = view.quant.repository?.split("/").pop()?.replace(/[-_.]?gguf$/i, "")
  return { query: repository && repository.length > 0 ? repository : view.name, first: view.quant.repository }
}

/**
 * The user's visible models (`list`), or `empty` while none is visible, then those hidden and the files that can't run
 * yet, so unhiding a model brings the list back in the empty state's place.
 */
export function OwnedModels({ list, empty, hidden, unrunnable, onUnhide }: {
  list: ReactNode | null; empty: ReactNode; hidden: readonly OwnedModel[]; unrunnable: readonly UnrunnableFile[]; onUnhide: (model: OwnedModel) => void
}) {
  return <>
    {list ?? empty}
    <Hidden models={hidden} onUnhide={onUnhide} />
    <Unrunnable files={unrunnable} />
  </>
}

/** Models: the user's models, or `empty` while none is visible. */
export function ModelsScreen({ onExplore, empty }: { onExplore: (search: { readonly query: string; readonly first: string | null }) => void; empty: ReactNode }) {
  const models = useLocalModels()
  const hardware = useLocalInferenceHardware()
  const mutations = useLocalModelMutations()
  const [removed, setRemoved] = useState<ReadonlySet<string>>(new Set())
  const [picking, setPicking] = useState<OwnedModel | null>(null)
  if (Result.isFailure(models)) return <ErrorNotice title="Couldn’t load your models" description="Model information is unavailable. Check the service in Settings." />
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
      case "Failed": return <span className="flex items-center gap-2"><Status tone="failed" title={view.failure?.summary}>{view.failure?.label ?? modelFailureLabel(residency.failure)}</Status>{load}</span>
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
  const choosing = picking === null ? undefined : byId.get(picking.modelId)
  return <div>
    <OwnedModels list={owned.length === 0 ? null : <YourModels models={owned} action={action}
        onExploreQuants={view => onExplore(quantSearch(view))} onRemove={remove} onDrafter={setPicking} />}
      empty={empty} hidden={hidden} unrunnable={models.value.unsupported.map(unrunnableFile)} onUnhide={unhide} />
    {picking && choosing?._tag === "Custom" && <DrafterDialog model={choosing} bits={picking.quant.bits} onClose={() => setPicking(null)}
      onChoose={choice => mutations.setDrafter(choosing.modelId as CustomFormModelId, choice)} />}
  </div>
}
