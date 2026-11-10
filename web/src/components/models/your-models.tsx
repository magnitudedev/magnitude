import { Fragment, useEffect, useRef, useState, type ReactNode } from "react"
import { CaretDownIcon, CaretRightIcon, CheckIcon, DotsThreeIcon, FunnelSimpleIcon, MagnifyingGlassIcon, PlusIcon, TrashIcon } from "@phosphor-icons/react"
import { formatStorageSize } from "@magnitudedev/client-common"
import { Button } from "../ui/button"
import { Input } from "../ui/input"
import { HuggingFaceLogo, LabLogo, modelLabs, type ModelLab } from "../model-logo"
import { cn } from "@/lib/utils"
import { ContextText, LabMark, PlacementBar, RepositoryLink, Section, Speed } from "./parts"
import type { QuantView } from "./view"

/** A model on this computer, with where it lives and what Magnitude could tell about its source. */
export interface OwnedModel {
  readonly modelId: string
  readonly name: string
  readonly lab: ModelLab | undefined
  readonly country: string | null
  readonly developer: string | null
  readonly license: string | null
  readonly quant: QuantView
  readonly contextLimit: number | null
  readonly location: string | null
  readonly drafter: { readonly method: string; readonly sizeBytes: number | null; readonly attached: boolean } | null
  readonly searchText: string
}

export interface UnrunnableFile {
  readonly name: string
  readonly reason: string
}

export interface ModelFilters {
  readonly lab: ModelLab | null
  readonly usOnly: boolean
}

const passes = (filters: ModelFilters, model: OwnedModel) =>
  (filters.lab === null || model.lab?.name === filters.lab.name) && (!filters.usOnly || model.country === "US")

function useDismiss(open: boolean, close: () => void) {
  const ref = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    if (!open) return
    const listener = (event: MouseEvent) => { if (!ref.current?.contains(event.target as Node)) close() }
    document.addEventListener("mousedown", listener)
    return () => document.removeEventListener("mousedown", listener)
  }, [open, close])
  return ref
}

/** One small menu for both filters; what's chosen stays shown in the menu and as a count on the button. */
function FilterMenu({ filters, setFilters, labs }: { filters: ModelFilters; setFilters: (filters: ModelFilters) => void; labs: readonly ModelLab[] }) {
  const [open, setOpen] = useState(false)
  const ref = useDismiss(open, () => setOpen(false))
  const active = (filters.lab ? 1 : 0) + (filters.usOnly ? 1 : 0)
  return <span ref={ref} className="relative">
    <Button size="sm" variant={active > 0 ? "secondary" : "ghost"} onClick={() => setOpen(value => !value)}><FunnelSimpleIcon />Filter{active > 0 ? ` (${active})` : ""}</Button>
    {open && <div className="absolute right-0 top-8 z-20 w-56 rounded-lg border border-slate-200 bg-white p-1.5 text-[13px] text-slate-800 shadow-lg dark:border-slate-750 dark:bg-slate-850 dark:text-slate-200">
      <p className="px-2 pb-1 pt-1.5 text-[11px] font-medium text-slate-500">Lab</p>
      {[null, ...labs].map(lab => <button key={lab?.name ?? "any"} type="button" onClick={() => setFilters({ ...filters, lab })} className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left hover:bg-slate-100 dark:hover:bg-slate-800">
        {lab ? <LabLogo lab={lab} className="size-4" /> : <span className="size-4" />}{lab?.name ?? "Any lab"}{(filters.lab?.name ?? null) === (lab?.name ?? null) && <CheckIcon className="ml-auto size-3.5 text-blue-600" />}
      </button>)}
      <div className="my-1 border-t border-slate-200 dark:border-slate-750" />
      <label className="flex cursor-pointer items-center gap-2 rounded px-2 py-1.5 hover:bg-slate-100 dark:hover:bg-slate-800"><input type="checkbox" checked={filters.usOnly} onChange={event => setFilters({ ...filters, usOnly: event.target.checked })} className="accent-blue-600" />Only US developers</label>
    </div>}
  </span>
}

/** A model's ⋯ menu: other quants of it on Hugging Face, and removing this one. */
function ModelMenu({ onExploreQuants, onRemove }: { onExploreQuants: () => void; onRemove: () => void }) {
  const [open, setOpen] = useState(false)
  const ref = useDismiss(open, () => setOpen(false))
  return <span ref={ref} className="relative" onClick={event => event.stopPropagation()}>
    <Button size="icon-sm" variant="ghost" aria-label="More" onClick={() => setOpen(value => !value)}><DotsThreeIcon className="size-4" /></Button>
    {open && <div className="absolute right-0 top-8 z-20 w-48 rounded-lg border border-slate-200 bg-white p-1 text-[13px] text-slate-800 shadow-lg dark:border-slate-750 dark:bg-slate-850 dark:text-slate-200">
      <button type="button" onClick={() => { setOpen(false); onExploreQuants() }} className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left hover:bg-slate-100 dark:hover:bg-slate-800"><MagnifyingGlassIcon className="size-3.5" />Explore other quants</button>
      <button type="button" onClick={() => { setOpen(false); onRemove() }} className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-red-600 hover:bg-slate-100 dark:text-red-400 dark:hover:bg-slate-800"><TrashIcon className="size-3.5" />Remove</button>
    </div>}
  </span>
}

function QuantLine({ model }: { model: OwnedModel }) {
  return <span className="block truncate text-xs text-slate-500"><span className="font-medium text-slate-700 dark:text-slate-300">{model.quant.bits}-bit</span> · {model.quant.publisher ? `${model.quant.publisher} ` : ""}{model.quant.name}
    {model.quant.servedContext !== null && <> · <ContextText quant={model.quant} limit={model.contextLimit} modelName={model.name} /></>}</span>
}

/** What opening a model shows: how it runs here, then where it came from and what goes with it. */
function Details({ model, onAddDrafter }: { model: OwnedModel; onAddDrafter: () => void }) {
  const rows: readonly (readonly [string, ReactNode])[] = [
    ["Source", model.quant.repository ? <span className="inline-flex min-w-0 items-center gap-1 text-slate-700 dark:text-slate-300"><HuggingFaceLogo className="size-3.5" /><RepositoryLink repository={model.quant.repository} /></span> : <span className="text-slate-500">Not matched to a Hugging Face repo</span>],
    ...(model.location ? [["Location", <span className="truncate font-mono text-[11px] text-slate-600 dark:text-slate-400">{model.location}</span>] as const] : []),
    ["Drafter", model.drafter === null ? <span className="text-slate-500">None</span>
      : model.drafter.attached ? <span className="rounded border border-slate-300 px-1.5 text-[11px] font-medium text-slate-600 dark:border-slate-600 dark:text-slate-300">{model.drafter.method}{model.drafter.sizeBytes !== null && ` · ${formatStorageSize(model.drafter.sizeBytes)}`}</span>
      : <Button size="xs" variant="outline" onClick={event => { event.stopPropagation(); onAddDrafter() }}><PlusIcon />Add {model.drafter.method} drafter{model.drafter.sizeBytes !== null && ` · ${formatStorageSize(model.drafter.sizeBytes)}`}</Button>],
    ...(model.developer ? [["Developer", <span className="text-slate-700 dark:text-slate-300">{model.developer}{model.license && <span className="text-slate-500"> · {model.license}</span>}</span>] as const] : []),
  ]
  return <div className="space-y-4">
    {model.quant.fits ? <PlacementBar quant={model.quant} /> : <p className="text-xs text-slate-500">Doesn’t fit on this computer</p>}
    <dl className="grid grid-cols-[80px_minmax(0,1fr)] items-center gap-x-4 gap-y-2 text-xs">
      {rows.map(([label, value]) => <Fragment key={label}><dt className="text-slate-500">{label}</dt><dd className="flex min-w-0 items-center">{value}</dd></Fragment>)}
    </dl>
  </div>
}

export function Unrunnable({ files }: { files: readonly UnrunnableFile[] }) {
  const [open, setOpen] = useState(false)
  if (files.length === 0) return null
  return <div className="mt-3">
    <button type="button" onClick={() => setOpen(value => !value)} className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-800 dark:hover:text-slate-200">
      {open ? <CaretDownIcon className="size-3" /> : <CaretRightIcon className="size-3" />}Doesn’t run in Magnitude yet ({files.length})
    </button>
    {open && <div className="mt-2 divide-y divide-slate-200 overflow-hidden rounded-xl border border-slate-200 bg-white dark:divide-slate-750 dark:border-slate-750 dark:bg-slate-850">
      {files.map(file => <div key={file.name} className="flex items-center justify-between gap-3 px-4 py-2.5 text-[13px]">
        <span className="truncate text-slate-600 dark:text-slate-400">{file.name}</span>
        <span className="shrink-0 text-xs text-slate-500">{file.reason}</span>
      </div>)}
    </div>}
  </div>
}

/** The user's models: one row each, opened in place, with search and filters for this list only. */
export function YourModels({ models, unrunnable, action, onExploreQuants, onRemove, onAddDrafter }: {
  models: readonly OwnedModel[]; unrunnable: readonly UnrunnableFile[]
  action: (model: OwnedModel) => ReactNode; onExploreQuants: (model: OwnedModel) => void; onRemove: (model: OwnedModel) => void; onAddDrafter: (model: OwnedModel) => void
}) {
  const [open, setOpen] = useState<string | null>(null)
  const [find, setFind] = useState("")
  const [filters, setFilters] = useState<ModelFilters>({ lab: null, usOnly: false })
  const labs = modelLabs.filter(lab => models.some(model => model.lab?.name === lab.name))
  const needle = find.trim().toLowerCase()
  const shown = models.filter(model => passes(filters, model) && model.searchText.toLowerCase().includes(needle))
  return <Section title={`Your models (${models.length})`} aside={<>
    <span className="relative">
      <MagnifyingGlassIcon className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-slate-400" />
      <Input value={find} onChange={event => setFind(event.target.value)} placeholder="Find in your models" className="h-7 w-52 pl-8 text-xs" />
    </span>
    <FilterMenu filters={filters} setFilters={setFilters} labs={labs} />
  </>}>
    {shown.length === 0 ? <p className="text-[13px] text-slate-500">No models match.</p> : <div className="divide-y divide-slate-200 overflow-hidden rounded-xl border border-slate-200 bg-white dark:divide-slate-750 dark:border-slate-750 dark:bg-slate-850">
      {shown.map(model => {
        const isOpen = open === model.modelId
        return <div key={model.modelId} className={cn(isOpen && "bg-slate-50 dark:bg-slate-800/40")}>
          <div role="button" tabIndex={0} aria-expanded={isOpen} onClick={() => setOpen(isOpen ? null : model.modelId)} onKeyDown={event => { if (event.key === "Enter") setOpen(isOpen ? null : model.modelId) }}
            className="grid cursor-pointer grid-cols-[28px_minmax(0,1fr)_64px_72px_auto_28px] items-center gap-4 px-4 py-3 hover:bg-slate-50 dark:hover:bg-slate-800/40 max-lg:grid-cols-[28px_minmax(0,1fr)_64px_auto_28px]">
            <LabMark lab={model.lab} className="size-7" />
            <span className="min-w-0"><span className="block truncate text-[13px] font-medium">{model.name}</span><QuantLine model={model} /></span>
            <span className="text-right text-[13px] tabular-nums text-slate-700 dark:text-slate-300">{formatStorageSize(model.quant.sizeBytes)}</span>
            <span className="text-right max-lg:hidden">{model.quant.fits && <Speed quant={model.quant} />}</span>
            <span className="flex justify-end" onClick={event => event.stopPropagation()}>{model.quant.fits ? action(model) : <span className="text-xs text-slate-500">Doesn’t fit</span>}</span>
            <ModelMenu onExploreQuants={() => onExploreQuants(model)} onRemove={() => onRemove(model)} />
          </div>
          {isOpen && <div className="px-4 pb-5 pl-[60px] pr-16"><Details model={model} onAddDrafter={() => onAddDrafter(model)} /></div>}
        </div>
      })}
    </div>}
    <Unrunnable files={unrunnable} />
  </Section>
}
