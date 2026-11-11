import { useRef, useState } from "react"
import { CaretDownIcon, CaretRightIcon, CheckIcon, CircleNotchIcon, DownloadSimpleIcon, FolderSimpleIcon, WarningCircleIcon } from "@phosphor-icons/react"
import { formatStorageSize } from "@magnitudedev/client-common"
import { Button } from "../ui/button"
import { HuggingFaceLogo, type ModelLab } from "../model-logo"
import { Dialog, DialogContent, DialogTitle } from "../ui/dialog"
import { cn } from "@/lib/utils"
import { cardClass, LabMark, MachineCard, Section } from "./parts"
import { MachineCardSkeleton, OfferedModelsSkeleton } from "./skeletons"
import { ExpandingModel, type OfferedModel } from "./expanding-model"
import { downloadContents } from "./data"
import type { UnrunnableFile } from "./your-models"
import type { MachineView, QuantView } from "./view"

/** A model found on this computer, as onboarding lists it: what it is and where it came from. */
export interface DiscoveredModel {
  readonly key: string
  readonly name: string
  readonly lab: ModelLab | undefined
  readonly bits: number
  readonly source: string
  readonly fromHuggingFaceCache: boolean
  readonly warning: string | null
}

/** One offered model with what onboarding says beside its name. */
export interface Recommendation {
  readonly model: OfferedModel
  readonly note: string | undefined
}

/** One thing the service does before it can recommend: done, under way (with how far, when it knows), or still to come. */
export interface PreparationStep {
  readonly label: string
  readonly detail: string | null
  readonly state: "done" | "active" | "pending"
  readonly fraction: number | null
}

/** What the service does before it can recommend, in order. */
export const preparationLabels = [
  "Starting Magnitude",
  "Reading your hardware",
  "Looking for models already on this computer",
  "Measuring how models run on your machine",
] as const

function PreparationProgress({ steps }: { steps: readonly PreparationStep[] }) {
  return <div role="status" aria-live="polite" className={cn(cardClass, "mt-3 px-5 py-4")}>
    <ol className="space-y-2">
      {steps.map(step => <li key={step.label}>
        <div className="flex items-center gap-2.5 text-[13px]">
          <span className="flex size-4 shrink-0 items-center justify-center">
            {step.state === "done" ? <CheckIcon weight="bold" className="size-3.5 text-green-600 dark:text-green-400" />
              : step.state === "active" ? <CircleNotchIcon className="size-4 text-blue-600 motion-safe:animate-spin dark:text-blue-400" />
              : <span className="size-1.5 rounded-full bg-slate-300 dark:bg-slate-600" />}
          </span>
          <span className={cn(step.state === "pending" ? "text-slate-500" : step.state === "active" ? "font-medium" : "text-slate-700 dark:text-slate-300")}>{step.label}</span>
          {step.detail && <span className="ml-auto text-xs tabular-nums text-slate-500">{step.detail}</span>}
        </div>
        {step.state === "active" && step.fraction !== null && <div className="ml-[26px] mt-1.5 h-1 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-750">
          <div className="h-full rounded-full bg-blue-600 transition-[width] duration-500 dark:bg-blue-400" style={{ width: `${Math.max(0.04, step.fraction) * 100}%` }} />
        </div>}
      </li>)}
    </ol>
  </div>
}

/** Onboarding before the app has reached the service: the first step under way, the rest to come. */
export function OnboardingStarting() {
  return <Onboarding machine={null} discovered={[]} unrunnable={[]} recommendations={[]} usOnly={false} setUsOnly={() => {}} owned={new Set()}
    preparation={preparationLabels.map((label, index) => ({ label, detail: null, fraction: null, state: index === 0 ? "active" : "pending" }))}
    onDownload={() => {}} onFinish={() => {}} />
}

/** First run, in a modal over the app: what's already here, and one recommended model to download. */
export function Onboarding({ machine, preparation, discovered, unrunnable, recommendations, usOnly, setUsOnly, owned, onDownload, onFinish }: {
  machine: MachineView | null; preparation: readonly PreparationStep[] | null; discovered: readonly DiscoveredModel[]; unrunnable: readonly UnrunnableFile[]
  recommendations: readonly Recommendation[]; usOnly: boolean; setUsOnly: (value: boolean) => void; owned: ReadonlySet<string>
  onDownload: (model: OfferedModel, quant: QuantView) => void; onFinish: () => void
}) {
  const measuring = preparation !== null
  const body = useRef<HTMLDivElement>(null)
  const [selection, setSelection] = useState<string | null | undefined>(undefined)
  const [quantChoice, setQuantChoice] = useState<string | null>(null)
  const [showUnrunnable, setShowUnrunnable] = useState(false)
  const selected = selection === undefined ? recommendations[0]?.model.key ?? null : selection
  const picked = recommendations.find(entry => entry.model.key === selected)?.model
  const pickedQuant = picked ? picked.quants.find(quant => quant.key === quantChoice) ?? picked.quants.find(quant => quant.key === picked.recommended) : undefined
  return <Dialog open onOpenChange={open => { if (!open) onFinish() }} disablePointerDismissal>
    <DialogContent showCloseButton={false} initialFocus={body} className="flex h-[88vh] flex-col gap-0 overflow-hidden p-0 sm:max-w-4xl">
    <div className="flex items-center justify-between gap-4 border-b border-slate-200 px-6 py-4 dark:border-slate-750">
      <DialogTitle className="font-heading text-lg font-semibold">Choose a model</DialogTitle>
      <Button variant="ghost" size="sm" onClick={onFinish}>Skip</Button>
    </div>
    <div ref={body} tabIndex={-1} className="min-h-0 flex-1 overflow-y-auto bg-slate-50 px-6 pb-6 pt-5 outline-none dark:bg-slate-900/40">
      <div>{machine ? <MachineCard machine={machine} /> : <MachineCardSkeleton />}</div>
      {preparation && <PreparationProgress steps={preparation} />}

      {discovered.length > 0 && <div className={`${cardClass} mt-4 px-4 py-3`}>
        <p className="text-[13px] font-medium">Discovered models ({discovered.length})</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {discovered.map(entry => <span key={entry.key} className="inline-flex max-w-full items-center gap-2 rounded-lg bg-slate-100 py-1.5 pl-2 pr-2.5 text-[13px] dark:bg-slate-800">
            <LabMark lab={entry.lab} className="size-4" />
            <span className="truncate">{entry.name}</span>
            <span className="text-slate-500">{entry.bits}-bit</span>
            <span className="h-3 w-px bg-slate-300 dark:bg-slate-600" />
            <span className="inline-flex shrink-0 items-center gap-1 text-xs text-slate-500">
              {entry.fromHuggingFaceCache ? <HuggingFaceLogo className="size-3" /> : <FolderSimpleIcon className="size-3" />}{entry.source}
              {entry.warning && <span title={entry.warning} className="text-orange-600 dark:text-orange-300"><WarningCircleIcon className="size-3.5" /></span>}
            </span>
          </span>)}
        </div>
        {unrunnable.length > 0 && <div className="mt-2">
          <button type="button" onClick={() => setShowUnrunnable(value => !value)} className="inline-flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-800 dark:hover:text-slate-200">
            {showUnrunnable ? <CaretDownIcon className="size-3" /> : <CaretRightIcon className="size-3" />}Can’t run yet ({unrunnable.length})
          </button>
          {showUnrunnable && <ul className="mt-1.5 space-y-1 pl-4 text-xs text-slate-500">{unrunnable.map(file => <li key={file.name}>{file.name} · {file.reason}</li>)}</ul>}
        </div>}
      </div>}

      <Section title="Recommended for your machine" aside={<label className="flex cursor-pointer items-center gap-2 text-xs text-slate-600 dark:text-slate-400"><input type="checkbox" checked={usOnly} onChange={event => { setUsOnly(event.target.checked); setSelection(undefined); setQuantChoice(null) }} className="accent-blue-600" />Only models from US developers</label>}>
        {measuring ? <OfferedModelsSkeleton firstOpened /> : <div className="space-y-2">
            {recommendations.map(({ model, note }) => <ExpandingModel key={model.key} model={model} note={note} expanded={selected === model.key} owned={owned}
              onToggle={() => { setSelection(selected === model.key ? null : model.key); setQuantChoice(null) }}
              quantChoice={selected === model.key ? quantChoice : null} onQuantChoice={setQuantChoice} />)}
            {recommendations.length === 0 && <p className="text-[13px] text-slate-500">No model runs well on this computer yet.</p>}
          </div>}
      </Section>

    </div>
    <div className="flex items-center justify-end gap-2 border-t border-slate-200 px-6 py-4 dark:border-slate-750">
      {measuring ? <Button disabled><DownloadSimpleIcon />Download</Button> : <>
        {picked && pickedQuant && (discovered.length > 0 || owned.size > 0) && <Button variant="ghost" onClick={onFinish}>Continue without downloading</Button>}
        {picked && pickedQuant
          ? <>
            {downloadContents(picked, pickedQuant) && <span className="mr-1 text-xs text-slate-500">{downloadContents(picked, pickedQuant)}</span>}
            <Button disabled={!pickedQuant.fits} onClick={() => { onDownload(picked, pickedQuant); onFinish() }}><DownloadSimpleIcon />Download {formatStorageSize(pickedQuant.downloadBytes)}</Button>
          </>
          : <Button onClick={onFinish}>Continue</Button>}
      </>}
    </div>
    </DialogContent>
  </Dialog>
}
