import { useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { ArrowLeftIcon, ArrowSquareOutIcon, ArrowUpRightIcon, BookOpenIcon, CaretDownIcon, CaretRightIcon, ChartBarIcon, ChatCircleIcon, CheckIcon, CubeIcon, DownloadSimpleIcon, DotsThreeIcon, FolderSimpleIcon, FunnelSimpleIcon, GearIcon, InfoIcon, GraphicsCardIcon, HardDrivesIcon, MagnifyingGlassIcon, MemoryIcon, PlayIcon, PlugsIcon, PlusIcon, StopIcon, TrashIcon, WarningCircleIcon, XIcon } from "@phosphor-icons/react"
import { Button, buttonVariants } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog"
import { MagnitudeMark } from "@/components/magnitude-mark"
import { HuggingFaceLogo, LabLogo, labForName, modelLabs, type ModelLab } from "@/components/model-logo"
import { hardwarePhotos } from "@/components/hardware-photos"
import { pageLayout } from "@/components/page-layout"
import { cn } from "@/lib/utils"
import { fineTuneModels, foundModels, machines, pool, searchIndex, type FoundModel, type Machine, type Model, type Quant } from "./data"
import { ONBOARDING_FLOOR, allQuants, assess, chooseQuant, formatContext, formatSize, formatSpeed, measuredQuality, placementText, rankForMachine, type Assessment, type Ranked } from "./logic"

type Countries = "Any" | "US"
type Download = { readonly key: string; readonly modelId: string; readonly progress: number }
type Action = (model: Model, assessment: Assessment | undefined, primary?: boolean) => ReactNode
type Filters = { readonly lab: ModelLab | null; readonly countries: Countries }

/** A model on this computer, where it came from, and how much of its source is known. */
interface Owned {
  readonly model: Model
  readonly assessment: Assessment
  readonly origin: string
  readonly name: string
  readonly match: FoundModel["match"]
}

const quantKey = (quant: Quant) => `${quant.repo || quant.name}:${quant.name}`
const allModels = [...pool, ...fineTuneModels]
const modelById = (id: string) => allModels.find(model => model.id === id)!
const modelSummary = (model: Model) => model.fineTuneOf
  ? `Fine-tune of ${modelById(model.fineTuneOf).name} · ${model.id.split("/")[0]}`
  : `${model.developer}${model.activeB ? ` · ${model.paramsB}B MoE, ${model.activeB}B active` : ` · ${model.paramsB}B dense`}`
const quantSource = (quant: Quant) => quant.publisher ? `${quant.publisher} ${quant.name}` : `${quant.name} · source unknown`
const listClass = "divide-y divide-slate-200 overflow-hidden rounded-xl border border-slate-200 bg-white dark:divide-slate-750 dark:border-slate-750 dark:bg-slate-850"
const cardClass = "rounded-xl border border-slate-200 bg-white dark:border-slate-750 dark:bg-slate-850"
const passes = (filters: Filters, model: Model) => (filters.lab === null || model.lab === filters.lab.name) && (filters.countries === "Any" || model.country === filters.countries)

const ownedFromFound = (machine: Machine, entry: FoundModel): Owned | undefined => {
  if (!entry.modelId || !entry.quant) return undefined
  const model = modelById(entry.modelId)
  return {
    model,
    assessment: assess(machine, model, entry.quant),
    origin: entry.source,
    name: entry.match === "none" ? entry.fileName : model.name,
    match: entry.match,
  }
}

/** Every quant shown for a model on this machine, tested first, with the one chosen for this computer. */
const quantsFor = (machine: Machine, model: Model) => {
  const listed = allQuants(model, searchIndex).map(entry => ({ ...entry, assessment: assess(machine, model, entry.quant) }))
  const tested = listed.filter(entry => entry.tested)
  const candidates = (tested.length > 0 ? tested : listed.filter(entry => entry.runnable)).map(entry => entry.assessment)
  return { listed, tested, more: listed.filter(entry => !entry.tested), recommended: chooseQuant(candidates) }
}

function Logo({ lab, className = "size-8" }: { lab: string; className?: string }) {
  const found = modelLabs.find(entry => entry.name === lab)
  return <span className={cn("flex shrink-0 items-center justify-center", className)}>
    {found ? <img src={found.src} alt="" className={cn("size-full object-contain", found.theme)} /> : <CubeIcon className="size-full text-slate-400" />}
  </span>
}

function Badge({ children, tone = "slate" }: { children: ReactNode; tone?: "slate" | "blue" | "amber" }) {
  const tones = {
    slate: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300",
    blue: "bg-blue-50 text-blue-700 dark:bg-slate-800 dark:text-blue-400",
    amber: "bg-orange-50 text-orange-700 dark:bg-slate-800 dark:text-orange-300",
  }
  return <span className={cn("inline-flex items-center gap-1 whitespace-nowrap rounded px-1.5 py-0.5 text-[11px] font-medium leading-4", tones[tone])}>{children}</span>
}

const foundPaths: Record<string, string> = { "Hugging Face cache": "~/.cache/huggingface/hub", "LM Studio": "~/.lmstudio/models", "Ollama": "~/.ollama/models", "llama.cpp": "~/Library/Caches/llama.cpp", "Downloads folder": "~/Downloads" }
const outlineBadge = "inline-flex shrink-0 items-center gap-1 whitespace-nowrap rounded border px-1.5 py-px text-[11px] font-medium leading-4"

const sourceName: Record<string, string> = { "Hugging Face cache": "HF cache", "Downloads folder": "Downloads" }


/** The source as quiet inline text, for places already inside a chip. */
function SourceText({ owned }: { owned: Owned }) {
  const warning = owned.match === "model" ? "Source unknown: Magnitude knows which model this is but not which Hugging Face repo it came from."
    : owned.match === "none" ? `Unknown model: Magnitude can’t tell which model this file is. It runs because it’s built on ${owned.model.family}.` : null
  return <span className="inline-flex shrink-0 items-center gap-1 text-xs text-slate-500">
    {owned.origin === "Hugging Face cache" ? <HuggingFaceLogo className="size-3" /> : <FolderSimpleIcon className="size-3" />}
    {sourceName[owned.origin] ?? owned.origin}
    {warning && <span title={warning} className="text-orange-600 dark:text-orange-300"><WarningCircleIcon className="size-3.5" /></span>}
  </span>
}

function QuantLine({ owned }: { owned: Owned }) {
  const quant = owned.assessment.quant
  const served = owned.assessment.servedK
  return <span className="block truncate text-xs text-slate-500"><span className="font-medium text-slate-700 dark:text-slate-300">{owned.assessment.label}</span> · {quant.publisher ? `${quant.publisher} ` : ""}{quant.name}
    {served > 0 && <> · {served < owned.model.contextK ? <ContextNote model={owned.model} assessment={owned.assessment} /> : <span>{formatContext(served)} context</span>}</>}</span>
}


function RepoLink({ repo }: { repo: string }) {
  return <a href={`https://huggingface.co/${repo}`} target="_blank" rel="noreferrer" onClick={event => event.stopPropagation()} className="inline-flex min-w-0 items-center gap-1 text-slate-700 hover:text-blue-700 hover:underline dark:text-slate-300 dark:hover:text-blue-400">
    <HuggingFaceLogo className="size-3.5" /><span className="truncate">{repo}</span><ArrowSquareOutIcon className="size-3 shrink-0" />
  </a>
}

function Section({ title, aside, children, className = "mt-8" }: { title: ReactNode; aside?: ReactNode; children: ReactNode; className?: string }) {
  return <section className={className}>
    <div className="mb-3 flex min-h-7 items-center justify-between gap-3">
      <h2 className="font-heading text-sm font-semibold text-slate-900 dark:text-slate-100">{title}</h2>
      {aside && <span className="flex items-center gap-3 text-xs text-slate-500">{aside}</span>}
    </div>
    {children}
  </section>
}

/** Context is shown only when this machine cuts it short of what the model supports. */
function ContextNote({ model, assessment }: { model: Model; assessment: Assessment }) {
  if (assessment.servedK === 0 || assessment.servedK >= model.contextK) return null
  return <span title={`${model.name} supports ${formatContext(model.contextK)}; ${formatContext(assessment.servedK)} is what fits on your machine.`} className="whitespace-nowrap text-orange-700 dark:text-orange-300">{formatContext(assessment.servedK)} context</span>
}


/** An estimate, so it stays small and muted; where the model sits is in its tooltip. */
function Speed({ assessment }: { assessment: Assessment | undefined }) {
  if (!assessment || assessment.placement._tag === "DoesNotFit") return <span className="text-xs text-slate-500">Doesn’t fit</span>
  return <span title={`Estimated decode speed in a long agent conversation. ${placementText(assessment.placement)}.`} className="whitespace-nowrap text-xs tabular-nums text-slate-500">~{formatSpeed(assessment.speed)}</span>
}

const downloadSize = (model: Model, quant: Quant) => quant.sizeGB + (model.drafter?.sizeGB ?? 0)

function ModelAction({ model, assessment, downloads, installed, loaded, onDownload, onLoad, primary = false }: {
  model: Model; assessment: Assessment | undefined; downloads: readonly Download[]; installed: ReadonlySet<string>; loaded: string | null; primary?: boolean
  onDownload: (model: Model, quant: Quant) => void; onLoad: (key: string | null) => void
}) {
  if (model.comingSoon) return <Badge tone="amber">Coming soon</Badge>
  if (!assessment || assessment.placement._tag === "DoesNotFit") return <span />
  const key = quantKey(assessment.quant)
  const download = downloads.find(entry => entry.key === key)
  if (download) return <span className="flex w-28 flex-col items-end gap-1"><span className="text-xs tabular-nums text-slate-500">Downloading {Math.round(download.progress * 100)}%</span><span className="h-1 w-full overflow-hidden rounded-full bg-slate-150 dark:bg-slate-750"><span className="block h-full bg-blue-600 dark:bg-blue-400" style={{ width: `${download.progress * 100}%` }} /></span></span>
  if (installed.has(key)) return loaded === key
    ? <Button size="sm" variant="outline" onClick={event => { event.stopPropagation(); onLoad(null) }}><StopIcon />Stop</Button>
    : <Button size="sm" variant="outline" onClick={event => { event.stopPropagation(); onLoad(key) }}><PlayIcon />Load</Button>
  return <Button size={primary ? "default" : "sm"} variant={primary ? "default" : "outline"} onClick={event => { event.stopPropagation(); onDownload(model, assessment.quant) }}><DownloadSimpleIcon />Download {formatSize(downloadSize(model, assessment.quant))}</Button>
}

const tierColor: Record<string, string> = { GPU: "bg-blue-600 dark:bg-blue-400", Memory: "bg-blue-600 dark:bg-blue-400", RAM: "bg-blue-300 dark:bg-blue-700", SSD: "bg-slate-300 dark:bg-slate-600" }

const tierPhrase = (name: string) => name === "SSD" ? "read from SSD as needed" : name === "RAM" ? "held in RAM" : name === "GPU" ? "held in GPU memory" : "held in memory"

/** How the chosen quant runs: what stays in fast memory, and what is read from the next tier while it runs. */
function StackedBar({ assessment }: { assessment: Assessment }) {
  const total = assessment.tiers.reduce((sum, tier) => sum + tier.usedGB, 0)
  return <div>
    <p className="mb-2 text-xs font-medium text-slate-700 dark:text-slate-300">How {assessment.label} runs on your machine</p>
    <div className="flex gap-0.5">
      {assessment.tiers.map(tier => <div key={tier.name} className={cn("h-2 min-w-1 rounded-full", tierColor[tier.name])} style={{ width: `${tier.usedGB / total * 100}%` }} />)}
    </div>
    <p className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-slate-500">
      {assessment.tiers.map(tier => <span key={tier.name} className="inline-flex items-center gap-1.5 whitespace-nowrap"><span className={cn("size-2 rounded-full", tierColor[tier.name])} /><span className="font-medium text-slate-700 dark:text-slate-300">{formatSize(tier.usedGB)}</span>{tierPhrase(tier.name)}</span>)}
    </p>
  </div>
}



/** Rungs side by side, smallest to largest: bits, source, then the two numbers that decide it. */
function QuantLadder({ entries, selected, recommended, onSelect, foundKeys, tested = false, limitK }: { entries: readonly Assessment[]; selected: string | null; recommended: Assessment | undefined; onSelect: (key: string) => void; foundKeys?: ReadonlySet<string>; tested?: boolean; limitK?: number }) {
  return <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${entries.length}, minmax(0, 1fr))` }}>
    {entries.map(a => {
      const key = quantKey(a.quant)
      const active = selected === key
      const fits = a.placement._tag !== "DoesNotFit"
      return <button key={key} type="button" disabled={!fits} onClick={() => onSelect(key)} title={tested ? measuredQuality(a.quant) : undefined}
        className={cn("flex flex-col items-start rounded-lg border px-3 py-2.5 text-left disabled:opacity-40", active ? "border-blue-600 bg-blue-50/60 ring-1 ring-blue-600/30 dark:border-blue-400 dark:bg-slate-800" : "border-slate-200 bg-white hover:border-slate-300 dark:border-slate-750 dark:bg-slate-850")}>
        <span className="flex w-full items-center justify-between gap-1 text-[13px] font-medium">{a.label}{foundKeys?.has(key) ? <Badge>Downloaded</Badge> : recommended && quantKey(recommended.quant) === key && <Badge tone="blue">Best</Badge>}</span>
        <span className="w-full truncate text-[11px] text-slate-500">{quantSource(a.quant)}</span>
        {limitK !== undefined && fits && a.servedK < limitK && <span className="text-[11px] text-orange-700 dark:text-orange-300" title={`Fits ${formatContext(a.servedK)} of context on your machine`}>{formatContext(a.servedK)} context</span>}
        <span className="mt-2.5 flex w-full items-baseline justify-between gap-2 tabular-nums">
          <span className="text-[13px] font-medium">{fits ? `~${formatSpeed(a.speed)}` : "Doesn’t fit"}</span>
          <span className="text-xs text-slate-500">{formatSize(a.quant.sizeGB)}</span>
        </span>
      </button>
    })}
  </div>
}

/** One small menu for both filters; what's chosen stays shown in the menu and as a count on the button. */
function FilterMenu({ filters, setFilters }: { filters: Filters; setFilters: (filters: Filters) => void }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    if (!open) return
    const close = (event: MouseEvent) => { if (!ref.current?.contains(event.target as Node)) setOpen(false) }
    document.addEventListener("mousedown", close)
    return () => document.removeEventListener("mousedown", close)
  }, [open])
  const labs = modelLabs.filter(entry => allModels.some(model => model.lab === entry.name))
  const active = (filters.lab ? 1 : 0) + (filters.countries === "US" ? 1 : 0)
  return <span ref={ref} className="relative flex items-center gap-1.5">
    <Button size="sm" variant={active > 0 ? "secondary" : "ghost"} onClick={() => setOpen(value => !value)}><FunnelSimpleIcon />Filter{active > 0 ? ` (${active})` : ""}</Button>
    {open && <div className="absolute right-0 top-8 z-20 w-56 rounded-lg border border-slate-200 bg-white p-1.5 text-[13px] text-slate-800 shadow-lg dark:border-slate-750 dark:bg-slate-850 dark:text-slate-200">
      <p className="px-2 pb-1 pt-1.5 text-[11px] font-medium text-slate-500">Lab</p>
      {[null, ...labs].map(lab => <button key={lab?.name ?? "any"} type="button" onClick={() => setFilters({ ...filters, lab })} className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left hover:bg-slate-100 dark:hover:bg-slate-800">
        {lab ? <LabLogo lab={lab} className="size-4" /> : <span className="size-4" />}{lab?.name ?? "Any lab"}{(filters.lab?.name ?? null) === (lab?.name ?? null) && <CheckIcon className="ml-auto size-3.5 text-blue-600" />}
      </button>)}
      <div className="my-1 border-t border-slate-200 dark:border-slate-750" />
      <label className="flex cursor-pointer items-center gap-2 rounded px-2 py-1.5 hover:bg-slate-100 dark:hover:bg-slate-800"><input type="checkbox" checked={filters.countries === "US"} onChange={event => setFilters({ ...filters, countries: event.target.checked ? "US" : "Any" })} className="accent-blue-600" />Only US developers</label>
    </div>}
  </span>
}

function Fragment({ children }: { children: ReactNode }) { return <>{children}</> }

function Shell({ children }: { children: ReactNode }) {
  const nav = [["Chat", ChatCircleIcon], ["Models", CubeIcon], ["Connections", PlugsIcon], ["Usage", ChartBarIcon], ["Settings", GearIcon]] as const
  return <div className="relative flex h-screen bg-slate-50 font-sans text-slate-900 dark:bg-slate-925 dark:text-slate-200">
    <aside className="w-56 shrink-0 border-r border-slate-200 pt-10 pb-8 dark:border-slate-750 max-md:hidden">
      <div className="flex h-full flex-col">
        <div className="mb-10 mt-4 flex h-8 items-center gap-3 px-7 font-heading text-base font-semibold"><MagnitudeMark className="h-8 w-8 shrink-0" />Magnitude</div>
        <nav className="flex min-h-0 flex-1 flex-col gap-2 px-4">
          {nav.map(([name, Icon]) => <Fragment key={name}>
            {name === "Settings" && <span className={`${buttonVariants({ variant: "ghost" })} mt-auto h-10 justify-start gap-3 rounded-lg px-3 text-sm font-medium`}><BookOpenIcon className="size-4" />Docs<ArrowUpRightIcon className="ml-auto size-3.5 text-slate-400" /></span>}
            <span className={cn("flex h-10 items-center gap-3 rounded-lg px-3 text-sm font-medium", name === "Models" ? "bg-blue-50 text-blue-700 dark:bg-slate-800 dark:text-blue-400" : "text-slate-600 dark:text-slate-400")}><Icon className="size-4" />{name}</span>
          </Fragment>)}
        </nav>
      </div>
    </aside>
    <main className="min-w-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-[min(100%,72rem)] px-10 pb-40 pt-9 max-md:px-4">{children}</div>
    </main>
  </div>
}

// ── Models page ─────────────────────────────────────────────────────────────

function Unsupported({ items }: { items: readonly { name: string; reason: string }[] }) {
  const [open, setOpen] = useState(false)
  if (items.length === 0) return null
  return <div className="mt-3">
    <button type="button" onClick={() => setOpen(value => !value)} className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-800 dark:hover:text-slate-200">
      {open ? <CaretDownIcon className="size-3" /> : <CaretRightIcon className="size-3" />}Doesn’t run in Magnitude yet ({items.length})
    </button>
    {open && <div className={cn(listClass, "mt-2")}>
      {items.map(entry => <div key={entry.name} className="flex items-center justify-between gap-3 px-4 py-2.5 text-[13px]">
        <span className="truncate text-slate-600 dark:text-slate-400">{entry.name}</span>
        <span className="shrink-0 text-xs text-slate-500">{entry.reason}</span>
      </div>)}
    </div>}
  </div>
}

const supported: readonly (readonly [string, string, boolean])[] = [
  ["Qwen", "Qwen 3.5, 3.6, 3.8", false], ["Gemma", "Gemma 4", false], ["Liquid AI", "LFM 2.5, LFM 3", false], ["NVIDIA", "Nemotron 3, 3.5", false],
  ["OpenBMB", "MiniCPM 5, MiniCPM-V 4.7", false], ["Meta", "Muse Glimmer", false], ["DeepSeek", "DeepSeek V4.1", true],
]

/** The one place that says which models Magnitude runs: by lab and generation, fine-tunes included. */
function WhatRuns() {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const close = (event: MouseEvent) => { if (!ref.current?.contains(event.target as Node)) setOpen(false) }
    document.addEventListener("mousedown", close)
    return () => document.removeEventListener("mousedown", close)
  }, [open])
  return <div ref={ref} className="relative">
    <Button size="sm" variant="ghost" onClick={() => setOpen(value => !value)}><InfoIcon />What runs in Magnitude</Button>
    {open && <div className="absolute right-0 top-9 z-20 w-80 rounded-lg border border-slate-200 bg-white p-3 shadow-lg dark:border-slate-750 dark:bg-slate-850">
      <ul className="space-y-2">{supported.map(([lab, models, soon]) => {
        const entry = modelLabs.find(item => item.name === lab)
        return <li key={lab} className="flex items-center gap-2.5 text-[13px]">{entry && <LabLogo lab={entry} className="size-4" />}<span className="flex-1">{models}</span>{soon && <Badge tone="amber">Coming soon</Badge>}</li>
      })}</ul>
      <p className="mt-3 border-t border-slate-200 pt-3 text-xs text-slate-500 dark:border-slate-750">Any quant or fine-tune of these from Hugging Face runs too.</p>
    </div>}
  </div>
}

const supportLine = "Magnitude runs models built on Qwen 3.5–3.8, Gemma 4, LFM 2.5 and 3, Nemotron 3 and 3.5, MiniCPM and Muse, including their fine-tunes."

/** Exploring Hugging Face in a modal: search and the best models for this machine, each opening in place to its quants. */
function ExploreModal({ machine, open, setOpen, query, setQuery, action }: {
  machine: Machine; open: boolean; setOpen: (open: boolean) => void; query: string; setQuery: (value: string) => void; action: Action
}) {
  const [chosen, setExpanded] = useState<string | null | undefined>(undefined)
  const [quantChoices, setQuantChoices] = useState<Readonly<Record<string, string>>>({})
  const recommended = useMemo(() => rankForMachine(machine, "Any").filter(r => r.chosen && !r.model.comingSoon).slice(0, 3), [machine])
  const needle = query.trim().toLowerCase()
  const results = useMemo(() => {
    if (needle === "") return null
    const matches = searchIndex.filter(entry => `${entry.repo} ${allModels.find(m => m.id === entry.base)?.name ?? entry.base}`.toLowerCase().includes(needle))
    const runnable = [...new Set(matches.filter(entry => !entry.unsupportedFamily).map(entry => entry.base))].map(modelById)
      .sort((a, b) => Number(a.comingSoon ?? false) - Number(b.comingSoon ?? false) || Number(a.fineTuneOf !== undefined) - Number(b.fineTuneOf !== undefined))
    const unsupported = [...new Map(matches.filter(entry => entry.unsupportedFamily).map(entry => [entry.base, { name: entry.base, reason: `${entry.unsupportedFamily} architecture` }])).values()]
    return { runnable, unsupported }
  }, [needle])
  const expanded = chosen === undefined ? (results ? null : recommended[0]?.model.id ?? null) : chosen
  const card = (model: Model, note?: string) => <ExpandingModel key={model.id} machine={machine} model={model} note={note} expanded={expanded === model.id}
    onToggle={() => setExpanded(expanded === model.id ? null : model.id)} quantChoice={quantChoices[model.id] ?? null}
    onQuantChoice={key => setQuantChoices(choices => ({ ...choices, [model.id]: key }))} footer={current => action(model, current, true)} />
  return <Dialog open={open} onOpenChange={value => { setOpen(value); if (!value) setExpanded(undefined) }}>
    <DialogContent className="flex h-[85vh] flex-col gap-0 overflow-hidden p-0 sm:max-w-4xl">
      <DialogTitle className="sr-only">Explore models</DialogTitle>
      <div className="border-b border-slate-200 p-5 dark:border-slate-750">
        <div className="mb-4 flex items-center justify-between gap-4 pr-8"><h2 className="font-heading text-lg font-semibold">Explore models</h2><WhatRuns /></div>
        <div className="flex items-center gap-3">
          <div className="relative flex-1">
            <MagnifyingGlassIcon className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-slate-400" />
            <Input autoFocus value={query} onChange={event => setQuery(event.target.value)} placeholder="Search Hugging Face, or paste a link" className="h-10 pl-9 text-sm" />
            {query && <button type="button" aria-label="Clear search" onClick={() => setQuery("")} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-700"><XIcon className="size-4" /></button>}
          </div>
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto bg-slate-50 p-5 dark:bg-slate-900/40">
        {results ? <>
          <p className="mb-3 text-xs font-medium text-slate-500">Results on Hugging Face ({results.runnable.length})</p>
          {results.runnable.length > 0 ? <div className="space-y-2">{results.runnable.map(model => card(model))}</div> : <p className="max-w-lg text-[13px] text-slate-500">Nothing here runs in Magnitude. {supportLine}</p>}
          <Unsupported items={results.unsupported} />
        </> : <>
          <p className="mb-3 flex items-baseline justify-between text-xs font-medium text-slate-500"><span>Best for your machine</span><span className="font-normal">{machine.detail.split(" · ").slice(0, 2).join(" · ")}</span></p>
          <div className="space-y-2">{recommended.map((r, index) => card(r.model, index === 0 ? "Smartest that runs well here" : undefined))}</div>
        </>}
      </div>
    </DialogContent>
  </Dialog>
}

/** A model's ⋯ menu: other quants of it on Hugging Face, and removing this one. */
function ModelMenu({ onExploreQuants }: { onExploreQuants: () => void }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    if (!open) return
    const close = (event: MouseEvent) => { if (!ref.current?.contains(event.target as Node)) setOpen(false) }
    document.addEventListener("mousedown", close)
    return () => document.removeEventListener("mousedown", close)
  }, [open])
  return <span ref={ref} className="relative" onClick={event => event.stopPropagation()}>
    <Button size="icon-sm" variant="ghost" aria-label="More" onClick={() => setOpen(value => !value)}><DotsThreeIcon className="size-4" /></Button>
    {open && <div className="absolute right-0 top-8 z-20 w-48 rounded-lg border border-slate-200 bg-white p-1 text-[13px] text-slate-800 shadow-lg dark:border-slate-750 dark:bg-slate-850 dark:text-slate-200">
      <button type="button" onClick={() => { setOpen(false); onExploreQuants() }} className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left hover:bg-slate-100 dark:hover:bg-slate-800"><MagnifyingGlassIcon className="size-3.5" />Explore other quants</button>
      <button type="button" onClick={() => setOpen(false)} className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-red-600 hover:bg-slate-100 dark:text-red-400 dark:hover:bg-slate-800"><TrashIcon className="size-3.5" />Remove</button>
    </div>}
  </span>
}

function DrafterLine({ owned }: { owned: Owned }) {
  const drafter = owned.model.drafter
  if (!drafter || owned.match === "none") return <span className="text-slate-500">None</span>
  return owned.origin === "Magnitude"
    ? <span className={cn(outlineBadge, "border-slate-300 text-slate-600 dark:border-slate-600 dark:text-slate-300")}>{drafter.method} · {formatSize(drafter.sizeGB)}</span>
    : <Button size="xs" variant="outline" onClick={event => event.stopPropagation()}><PlusIcon />Add {drafter.method} drafter · {formatSize(drafter.sizeGB)}</Button>
}

/** What opening a model shows: how it runs here, then where it came from and what goes with it. */
function OwnedDetails({ owned }: { owned: Owned }) {
  const quant = owned.assessment.quant
  const rows: readonly (readonly [string, ReactNode])[] = [
    ["Source", quant.repo ? <RepoLink repo={quant.repo} /> : <span className="text-slate-500">Not matched to a Hugging Face repo</span>],
    ["Location", <span className="font-mono text-[11px] text-slate-600 dark:text-slate-400">{owned.origin === "Magnitude" ? foundPaths["Hugging Face cache"] : foundPaths[owned.origin]}</span>],
    ["Drafter", <DrafterLine owned={owned} />],
    ["Developer", <span className="text-slate-700 dark:text-slate-300">{owned.model.developer} <span className="text-slate-500">· {owned.model.license}</span></span>],
  ]
  return <div className="space-y-4">
    {owned.assessment.placement._tag !== "DoesNotFit" ? <StackedBar assessment={owned.assessment} /> : <p className="text-xs text-slate-500">Doesn’t fit on this computer</p>}
    <dl className="grid grid-cols-[80px_minmax(0,1fr)] items-center gap-x-4 gap-y-2 text-xs">
      {rows.map(([label, value]) => <Fragment key={label}><dt className="text-slate-500">{label}</dt><dd className="flex min-w-0 items-center">{value}</dd></Fragment>)}
    </dl>
  </div>
}

type OwnedView = "list" | "cards"

function ModelsPage({ machine, view, filters, setFilters, owned, unsupportedFound, onExplore, onExploreQuants, action }: {
  machine: Machine; view: OwnedView; filters: Filters; setFilters: (filters: Filters) => void; owned: readonly Owned[]; unsupportedFound: readonly FoundModel[]
  onExplore: () => void; onExploreQuants: (model: Model) => void; action: Action
}) {
  const [open, setOpen] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<string | null | undefined>(undefined)
  const [quantChoices, setQuantChoices] = useState<Readonly<Record<string, string>>>({})
  const recommended = useMemo(() => rankForMachine(machine, "Any").filter(r => r.chosen && !r.model.comingSoon).slice(0, 3), [machine])
  const unsupported = <Unsupported items={unsupportedFound.map(entry => ({ name: `${entry.fileName} · ${entry.source}`, reason: `${entry.unsupportedFamily} architecture` }))} />
  const started = expanded === undefined ? recommended[0]?.model.id ?? null : expanded
  const [find, setFind] = useState("")
  const shown = owned.filter(entry => passes(filters, entry.model) && `${entry.name} ${entry.assessment.quant.repo} ${entry.origin}`.toLowerCase().includes(find.trim().toLowerCase()))
  return <div>
    <div className="mb-5 flex items-center justify-between gap-4"><h1 className={pageLayout.pageTitle}>Models</h1><Button onClick={onExplore}><MagnifyingGlassIcon />Explore models</Button></div>
    <MachineCard machine={machine} />
    {owned.length === 0 && unsupportedFound.length === 0 ? <Section title="Get started">
      <div className="space-y-2">{recommended.map((r, index) => <ExpandingModel key={r.model.id} machine={machine} model={r.model} note={index === 0 ? "Smartest that runs well here" : undefined}
        expanded={started === r.model.id} onToggle={() => setExpanded(started === r.model.id ? null : r.model.id)}
        quantChoice={quantChoices[r.model.id] ?? null} onQuantChoice={key => setQuantChoices(choices => ({ ...choices, [r.model.id]: key }))} footer={current => action(r.model, current, true)} />)}</div>
    </Section> : <Section title={`Your models (${owned.length})`} aside={<>
      <span className="relative">
        <MagnifyingGlassIcon className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-slate-400" />
        <Input value={find} onChange={event => setFind(event.target.value)} placeholder="Find in your models" className="h-7 w-52 pl-8 text-xs" />
      </span>
      <FilterMenu filters={filters} setFilters={setFilters} />
    </>}>
      {view === "cards"
        ? <div className="grid grid-cols-2 gap-3 max-lg:grid-cols-1">{shown.map(entry => <div key={quantKey(entry.assessment.quant)} className={cn(cardClass, "flex flex-col p-5")}>
          <div className="flex items-start gap-3">
            <Logo lab={entry.model.lab} className="size-8" />
            <div className="min-w-0 flex-1">
              <span className="block truncate text-[14px] font-medium">{entry.name}</span>
              <QuantLine owned={entry} />
            </div>
            <ModelMenu onExploreQuants={() => onExploreQuants(entry.model)} />
          </div>
          <div className="mt-4 flex-1"><OwnedDetails owned={entry} /></div>
          <div className="mt-4 flex items-center justify-between gap-3 border-t border-slate-200 pt-3 dark:border-slate-750">
            <span className="text-xs text-slate-500"><span className="font-medium text-slate-700 dark:text-slate-300">{formatSize(entry.assessment.quant.sizeGB)}</span>{entry.assessment.placement._tag !== "DoesNotFit" && <> · <Speed assessment={entry.assessment} /></>}</span>
            {entry.assessment.placement._tag === "DoesNotFit" ? <span className="text-xs text-slate-500">Doesn’t fit</span> : action(entry.model, entry.assessment)}
          </div>
        </div>)}</div>
        : shown.length === 0 ? <p className="text-[13px] text-slate-500">No models match.</p> : <div className={listClass}>{shown.map(entry => {
          const key = quantKey(entry.assessment.quant)
          const isOpen = open === key
          return <div key={key} className={cn(isOpen && "bg-slate-50 dark:bg-slate-800/40")}>
            <div role="button" tabIndex={0} aria-expanded={isOpen} onClick={() => setOpen(isOpen ? null : key)} onKeyDown={event => { if (event.key === "Enter") setOpen(isOpen ? null : key) }}
              className="grid cursor-pointer grid-cols-[28px_minmax(0,1fr)_64px_72px_88px_28px] items-center gap-4 px-4 py-3 hover:bg-slate-50 dark:hover:bg-slate-800/40 max-lg:grid-cols-[28px_minmax(0,1fr)_64px_88px_28px]">
              <Logo lab={entry.model.lab} className="size-7" />
              <span className="min-w-0">
                <span className="block truncate text-[13px] font-medium">{entry.name}</span>
                <QuantLine owned={entry} />
              </span>
              <span className="text-right text-[13px] tabular-nums text-slate-700 dark:text-slate-300">{formatSize(entry.assessment.quant.sizeGB)}</span>
              <span className="text-right max-lg:hidden">{entry.assessment.placement._tag !== "DoesNotFit" && <Speed assessment={entry.assessment} />}</span>
              <span className="flex justify-end" onClick={event => event.stopPropagation()}>{entry.assessment.placement._tag === "DoesNotFit" ? <span className="text-xs text-slate-500">Doesn’t fit</span> : action(entry.model, entry.assessment)}</span>
              <ModelMenu onExploreQuants={() => onExploreQuants(entry.model)} />
            </div>
            {isOpen && <div className="px-4 pb-5 pl-[60px] pr-16"><OwnedDetails owned={entry} /></div>}
          </div>
        })}</div>}
      {unsupported}
    </Section>}
  </div>
}

// ── Onboarding ──────────────────────────────────────────────────────────────

function HardwarePhoto({ machine }: { machine: Machine }) {
  const photo = hardwarePhotos[machine.photo]
  if (!photo) return null
  const frame = photo.framing
  return <svg role="img" aria-label={photo.subject} viewBox={`${frame.x} ${frame.y} ${frame.width} ${frame.height}`} className="h-full w-full" preserveAspectRatio="xMidYMid meet"><image href={photo.src} width={frame.sourceWidth} height={frame.sourceHeight} /></svg>
}

/** The machine everything is measured against: what it is and its numbers. With several GPUs, the one models run on is chosen here. */
function MachineCard({ machine }: { machine: Machine }) {
  const gpus = machine.gpus ?? []
  const gpu = 0
  const chip = gpus.length > 1 ? `${gpus.length} GPUs` : machine.detail.split(" · ")[0]!.replace(/ \d+ GB$/, "")
  const disk = machine.freeDiskGB >= 1000 ? `${(machine.freeDiskGB / 1000).toFixed(1)} TB` : `${machine.freeDiskGB} GB`
  const gpuTile = gpus.length > 1
    ? <div>
      <p className="flex items-center gap-1.5 text-xs text-slate-500"><GraphicsCardIcon className="size-3.5" />GPU memory</p>
      <p className="mt-0.5 font-heading text-xl font-semibold tabular-nums">{gpus[gpu]!.vramGB} GB</p>
      <p className="mt-0.5 text-[11px] text-slate-500">GPU {gpu + 1} · {gpus[gpu]!.name}</p>
    </div>
    : null
  const parts = machine.kind === "unified"
    ? [{ icon: MemoryIcon, name: "Memory", value: `${machine.memoryGB} GB` }, { icon: HardDrivesIcon, name: "Free disk", value: disk }]
    : [{ icon: GraphicsCardIcon, name: "GPU memory", value: `${machine.vramGB} GB` }, { icon: MemoryIcon, name: "RAM", value: `${machine.memoryGB} GB` }, { icon: HardDrivesIcon, name: "Free disk", value: disk }]
  return <div className={cn(cardClass, "flex flex-wrap items-center gap-x-4 gap-y-4 px-5 py-4")}>
    <div className="aspect-[4/3] w-16 shrink-0"><HardwarePhoto machine={machine} /></div>
    <div className="min-w-56 flex-1">
      <h2 className="font-heading text-base font-semibold">Your machine</h2>
      <p className="truncate text-[14px] text-slate-700 dark:text-slate-300">{machine.name} <span className="text-slate-500">· {chip}</span></p>
      {gpus.length > 1 && <p className="mt-0.5 text-xs text-slate-500">{gpus.map(entry => entry.name).join(" + ")} · models run on GPU {gpu + 1}</p>}
    </div>
    <div className={cn("grid shrink-0 items-start gap-8", parts.length === 3 ? "grid-cols-3" : "grid-cols-2")}>
      {parts.map(part => part.name === "GPU memory" && gpuTile ? <Fragment key={part.name}>{gpuTile}</Fragment> : <div key={part.name}>
        <p className="flex items-center gap-1.5 text-xs text-slate-500"><part.icon className="size-3.5" />{part.name}</p>
        <p className="mt-0.5 font-heading text-xl font-semibold tabular-nums">{part.value}</p>
      </div>)}
    </div>
  </div>
}

function PickCard({ model, title, assessment, detail, note, active, onSelect, children }: { model: Model; title?: string; assessment: Assessment; detail: ReactNode; note?: string; active: boolean; onSelect: () => void; children?: ReactNode }) {
  return <div className={cn("overflow-hidden rounded-xl border bg-white dark:bg-slate-850", active ? "border-blue-600 ring-1 ring-blue-600/30 dark:border-blue-400" : "border-slate-200 hover:border-slate-300 dark:border-slate-750 dark:hover:border-slate-700")}>
    <div role="button" tabIndex={0} aria-expanded={active} onClick={onSelect} onKeyDown={event => { if (event.key === "Enter") onSelect() }} className="grid w-full cursor-pointer grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-4 px-4 py-3 text-left">
      <Logo lab={model.lab} className="size-10" />
      <span className="min-w-0">
        <span className="flex items-center gap-2 text-[14px] font-medium"><span className="truncate">{title ?? model.name}</span>{note && <span className="shrink-0 text-xs font-normal text-slate-500">{note}</span>}</span>
        <span className="mt-0.5 block min-w-0 truncate text-xs text-slate-500">{detail}</span>
      </span>
      <span className="flex items-center gap-4">
        <span className="grid grid-cols-[84px_64px] gap-4 text-right">
          <span><span className="block text-[15px] font-medium tabular-nums text-slate-900 dark:text-slate-100">{assessment.placement._tag === "DoesNotFit" ? "—" : `~${formatSpeed(assessment.speed)}`}</span><span className="block whitespace-nowrap text-[11px] text-slate-500">{placementText(assessment.placement)}</span></span>
          <span><span className="block text-[15px] font-medium tabular-nums text-slate-900 dark:text-slate-100">{formatSize(assessment.quant.sizeGB)}</span><span className="block text-[11px] text-slate-500">on disk</span></span>
        </span>
        <span className={cn("flex size-5 items-center justify-center rounded-full border", active ? "border-blue-600 bg-blue-600 text-white dark:border-blue-400 dark:bg-blue-400 dark:text-slate-925" : "border-slate-300 dark:border-slate-700")}>{active && <CheckIcon className="size-3" />}</span>
      </span>
    </div>
    {children}
  </div>
}

/** A model that opens in place to its quant ladder and how the chosen quant runs here. */
function ExpandingModel({ machine, model, note, expanded, onToggle, quantChoice, onQuantChoice, footer, foundKeys }: {
  machine: Machine; model: Model; note?: string | undefined; expanded: boolean; onToggle: () => void
  quantChoice: string | null; onQuantChoice: (key: string) => void; footer?: (current: Assessment | undefined) => ReactNode; foundKeys?: ReadonlySet<string>
}) {
  const quants = quantsFor(machine, model)
  const ladder = quants.tested.length > 0 ? quants.tested.map(entry => entry.assessment) : quants.more.filter(entry => entry.runnable && entry.quant.repo === quants.more[0]?.quant.repo).map(entry => entry.assessment)
  const current = ladder.find(a => quantKey(a.quant) === quantChoice) ?? quants.recommended
  if (!current) return null
  return <PickCard model={model} assessment={current} active={expanded} onSelect={onToggle} note={note}
    detail={<>
      <span className="font-medium text-slate-700 dark:text-slate-300">{current.label}</span>
      {" · "}<a href={`https://huggingface.co/${current.quant.repo}`} target="_blank" rel="noreferrer" title={current.quant.repo} onClick={event => event.stopPropagation()} className="inline-flex items-center gap-0.5 hover:text-blue-700 hover:underline dark:hover:text-blue-400">{quantSource(current.quant)}<ArrowSquareOutIcon className="size-3" /></a>
      {model.drafter ? ` · ${model.drafter.method} drafter` : ""}
      {" · "}{current.servedK < model.contextK ? <ContextNote model={model} assessment={current} /> : <span className="text-slate-700 dark:text-slate-300">{formatContext(current.servedK)} context</span>}
      {model.country === "US" ? " · 🇺🇸" : ""}
    </>}>
    {expanded && <div className="border-t border-slate-200 bg-slate-50/60 px-4 py-4 dark:border-slate-750 dark:bg-slate-900/40">
      <QuantLadder entries={ladder} selected={quantKey(current.quant)} recommended={quants.recommended} onSelect={onQuantChoice} tested={quants.tested.length > 0} limitK={model.contextK} {...(foundKeys ? { foundKeys } : {})} />
      {current.placement._tag !== "DoesNotFit" && <div className="mt-4"><StackedBar assessment={current} /></div>}
      {footer && <div className="mt-4 flex justify-end">{footer(current)}</div>}
    </div>}
  </PickCard>
}

function Onboarding({ machine, found, countries, setCountries, onFinish, download }: {
  machine: Machine; found: readonly FoundModel[]; countries: Countries; setCountries: (value: Countries) => void
  onFinish: () => void; download: (model: Model, quant: Quant) => void
}) {
  const [progress, setProgress] = useState(0)
  const [selection, setSelection] = useState<string | null | undefined>(undefined)
  const [quantChoice, setQuantChoice] = useState<string | null>(null)
  const [showUnsupported, setShowUnsupported] = useState(false)
  const measured = progress >= 1
  const owned = found.flatMap(entry => { const o = ownedFromFound(machine, entry); return o ? [o] : [] })
  const unsupportedFound = found.filter(entry => entry.match === "unsupported")
  const foundKeys = new Set(owned.map(entry => quantKey(entry.assessment.quant)))
  const candidates = useMemo(() => rankForMachine(machine, countries).filter(r => r.chosen && !r.model.comingSoon && r.chosen.speed >= ONBOARDING_FLOOR).slice(0, 3), [machine, countries])
  // The best one starts open, so its quants are visible; someone who already has models can still just continue.
  const selected = selection === undefined ? candidates[0]?.model.id ?? null : selection
  const picked = candidates.find(r => r.model.id === selected)
  const pickedQuant = picked ? picked.assessments.find(a => quantKey(a.quant) === quantChoice) ?? picked.chosen : undefined
  useEffect(() => {
    const timer = setInterval(() => setProgress(value => Math.min(1, value + 0.04)), 90)
    return () => clearInterval(timer)
  }, [machine.id])
  return <div className="h-screen overflow-y-auto bg-slate-50 font-sans text-slate-900 dark:bg-slate-925 dark:text-slate-200">
    <header className="flex items-center justify-between px-8 pt-8">
      <span className="flex items-center gap-3 font-heading text-base font-semibold"><MagnitudeMark className="h-8 w-8" />Magnitude</span>
      <Button variant="ghost" size="sm" onClick={onFinish}>Skip</Button>
    </header>
    <div className="mx-auto w-full max-w-3xl px-8 pb-40 pt-10">
      <h1 className="font-heading text-[28px] font-semibold tracking-tight">Choose a model</h1>
      <div className="mt-6"><MachineCard machine={machine} /></div>
      {!measured && <div className="mt-3 h-1 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-750"><div className="h-full bg-blue-600 dark:bg-blue-400" style={{ width: `${progress * 100}%` }} /></div>}

      {owned.length > 0 && <div className={cn(cardClass, "mt-4 px-4 py-3")}>
        <p className="text-[13px]"><span className="font-medium">Discovered models ({owned.length})</span></p>
        <div className="mt-3 flex flex-wrap gap-2">
          {owned.map(entry => <span key={quantKey(entry.assessment.quant)} className="inline-flex max-w-full items-center gap-2 rounded-lg bg-slate-100 py-1.5 pl-2 pr-2.5 text-[13px] dark:bg-slate-800">
            <Logo lab={entry.model.lab} className="size-4" />
            <span className="truncate">{entry.name}</span>
            <span className="text-slate-500">{entry.assessment.label}</span>
            <span className="h-3 w-px bg-slate-300 dark:bg-slate-600" />
            <SourceText owned={entry} />
          </span>)}
        </div>
        {unsupportedFound.length > 0 && <div className="mt-2">
          <button type="button" onClick={() => setShowUnsupported(value => !value)} className="inline-flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-800 dark:hover:text-slate-200">
            {showUnsupported ? <CaretDownIcon className="size-3" /> : <CaretRightIcon className="size-3" />}Can’t run yet ({unsupportedFound.length})
          </button>
          {showUnsupported && <ul className="mt-1.5 space-y-1 pl-4 text-xs text-slate-500">{unsupportedFound.map(entry => <li key={entry.fileName}>{entry.fileName} · {entry.source} · {entry.unsupportedFamily} architecture</li>)}</ul>}
        </div>}
      </div>}

      {!measured && <div className="mt-8 space-y-2">{[0, 1, 2].map(index => <div key={index} className="h-[66px] animate-pulse rounded-xl border border-slate-200 bg-white dark:border-slate-750 dark:bg-slate-850" />)}</div>}
      {measured && <Section title="Recommended for your machine" aside={<label className="flex cursor-pointer items-center gap-2 text-xs text-slate-600 dark:text-slate-400"><input type="checkbox" checked={countries === "US"} onChange={event => { setCountries(event.target.checked ? "US" : "Any"); setSelection(undefined); setQuantChoice(null) }} className="accent-blue-600" />Only models from US developers</label>}>
        <div className="space-y-2">
          {candidates.map((r, index) => {
            const have = r.assessments.find(a => foundKeys.has(quantKey(a.quant)))
            return <ExpandingModel key={r.model.id} machine={machine} model={r.model} expanded={selected === r.model.id}
              note={have ? `You have the ${have.label} quant` : index === 0 ? "Smartest that runs well here" : undefined}
              onToggle={() => { setSelection(selected === r.model.id ? null : r.model.id); setQuantChoice(null) }}
              quantChoice={selected === r.model.id ? quantChoice : null} onQuantChoice={setQuantChoice} foundKeys={foundKeys} />
          })}
          {candidates.length === 0 && <p className="text-[13px] text-slate-500">No model reaches {ONBOARDING_FLOOR} tok/s on this computer.</p>}
        </div>
      </Section>}

      <div className="mt-8 flex items-center justify-between gap-4">
        <span />
        <span className="flex items-center gap-2">
          {picked && pickedQuant && owned.length > 0 && <Button size="lg" variant="ghost" onClick={onFinish}>Continue without downloading</Button>}
          {picked && pickedQuant
            ? <Button size="lg" disabled={!measured} onClick={() => { download(picked.model, pickedQuant.quant); onFinish() }}><DownloadSimpleIcon />Download {formatSize(downloadSize(picked.model, pickedQuant.quant))}</Button>
            : <Button size="lg" disabled={!measured} onClick={onFinish}>Continue</Button>}
        </span>
      </div>
    </div>
  </div>
}

// ── Prototype frame ─────────────────────────────────────────────────────────

type PageState = "empty" | "found" | "downloading" | "search" | "nothing"
const pageStates: readonly (readonly [PageState, string])[] = [
  ["empty", "Nothing on this computer"],
  ["found", "Found in other apps"],
  ["downloading", "Downloading one"],
  ["search", "Exploring, searching “qwen3”"],
  ["nothing", "Exploring, searching “gpt-oss”"],
]

function Choice<T extends string>({ label, value, options, onChange }: { label: string; value: T; options: readonly (readonly [T, string])[]; onChange: (value: T) => void }) {
  return <label className="flex items-center gap-1.5"><span className="text-slate-500">{label}</span>
    <select value={value} onChange={event => onChange(event.target.value as T)} className="h-7 rounded-md border border-slate-300 bg-white px-1.5 text-xs dark:border-slate-700 dark:bg-slate-900">
      {options.map(([key, name]) => <option key={key} value={key}>{name}</option>)}
    </select>
  </label>
}

export function Prototype() {
  const [machineId, setMachineId] = useState("m4max64")
  const machine = machines.find(m => m.id === machineId)!
  const [countries, setCountries] = useState<Countries>("Any")
  const [filters, setFilters] = useState<Filters>({ lab: null, countries: "Any" })
  const [screen, setScreen] = useState<"onboarding" | "models">("onboarding")
  const [hasFound, setHasFound] = useState(true)
  const [pageState, setPageState] = useState<PageState>("found")
  const [query, setQuery] = useState("")
  const [exploring, setExploring] = useState(false)
  const [onboardingKey, setOnboardingKey] = useState(0)
  const [ownedView, setOwnedView] = useState<OwnedView>("list")
  const [downloads, setDownloads] = useState<readonly Download[]>([])
  const [installed, setInstalled] = useState<ReadonlySet<string>>(new Set())
  const [mine, setMine] = useState<ReadonlyMap<string, { modelId: string; quant: Quant }>>(new Map())
  const [loaded, setLoaded] = useState<string | null>(null)
  const [dark, setDark] = useState(document.documentElement.dataset.theme === "dark")

  useEffect(() => {
    if (downloads.length === 0) return
    const timer = setInterval(() => {
      setDownloads(current => {
        const done = current.filter(d => d.progress + 0.01 >= 1)
        if (done.length > 0) setInstalled(set => new Set([...set, ...done.map(d => d.key)]))
        return current.filter(d => d.progress + 0.01 < 1).map(d => ({ ...d, progress: d.progress + 0.01 }))
      })
    }, 200)
    return () => clearInterval(timer)
  }, [downloads.length > 0])

  const download = (model: Model, quant: Quant) => {
    const key = quantKey(quant)
    setMine(map => new Map(map).set(key, { modelId: model.id, quant }))
    setDownloads(current => current.some(d => d.key === key) ? current : [...current, { key, modelId: model.id, progress: 0 }])
  }
  const applyState = (state: PageState) => {
    setPageState(state)
    setHasFound(state !== "empty")
    setQuery(state === "search" ? "qwen3" : state === "nothing" ? "gpt-oss" : "")
    setExploring(state === "search" || state === "nothing")
    if (state === "downloading") { const flash = pool[0]!; download(flash, flash.quants[2]!) }
    if (state === "empty") { setMine(new Map()); setDownloads([]); setInstalled(new Set()) }
  }
  const found = hasFound ? foundModels : []
  const owned: readonly Owned[] = [
    ...[...mine.values()].map(({ modelId, quant }) => {
      const model = modelById(modelId)
      return { model, assessment: assess(machine, model, quant), origin: "Magnitude", name: model.name, match: "repo" as const }
    }),
    ...found.flatMap(entry => { const o = ownedFromFound(machine, entry); return o ? [o] : [] }),
  ]
  const present = new Set([...installed, ...found.flatMap(entry => entry.quant ? [quantKey(entry.quant)] : [])])
  const action: Action = (model, assessment, primary) => <ModelAction model={model} assessment={assessment} downloads={downloads} installed={present} loaded={loaded} onDownload={download} onLoad={setLoaded} primary={primary ?? false} />

  return <>
    {screen === "onboarding"
      ? <Onboarding key={`${onboardingKey}-${machineId}-${hasFound}`} machine={machine} found={found} countries={countries} setCountries={setCountries} download={download}
        onFinish={() => { setScreen("models") }} />
      : <Shell>
        <ModelsPage machine={machine} view={ownedView} filters={filters} setFilters={setFilters} owned={owned} unsupportedFound={found.filter(entry => entry.match === "unsupported")}
          onExplore={() => setExploring(true)} onExploreQuants={model => { setQuery(modelById(model.fineTuneOf ?? model.id).name); setExploring(true) }} action={action} />
        <ExploreModal machine={machine} open={exploring} setOpen={setExploring} query={query} setQuery={setQuery} action={action} />
      </Shell>}
    <div className="fixed bottom-4 left-1/2 z-50 flex -translate-x-1/2 flex-wrap items-center justify-center gap-3 rounded-2xl border border-slate-300 bg-white/95 px-4 py-2 text-xs shadow-lg backdrop-blur dark:border-slate-700 dark:bg-slate-850/95">
      <span className="font-semibold text-orange-700 dark:text-orange-300">Prototype</span>
      <Choice label="Computer" value={machineId} options={machines.map(m => [m.id, `${m.detail.split(" · ").slice(0, 2).join(" · ")}${m.measured ? " (measured)" : ""}`] as const)} onChange={setMachineId} />
      {screen === "onboarding"
        ? <Choice label="Found" value={hasFound ? "some" : "none"} options={[["some", "Models in other apps"], ["none", "Nothing"]]} onChange={value => setHasFound(value === "some")} />
        : <>
          <Choice label="Your models" value={ownedView} options={[["list", "List"], ["cards", "Cards"]]} onChange={setOwnedView} />
          <Choice label="State" value={pageState} options={pageStates} onChange={applyState} />
        </>}
      <Button size="xs" variant="outline" onClick={() => { if (screen === "onboarding") setOnboardingKey(k => k + 1); else setScreen("onboarding") }}>{screen === "onboarding" ? "Restart" : "Onboarding"}</Button>
      {screen === "onboarding" && <Button size="xs" variant="outline" onClick={() => setScreen("models")}>Models page</Button>}
      <Button size="xs" variant="ghost" onClick={() => { const next = !dark; document.documentElement.dataset.theme = next ? "dark" : "light"; document.documentElement.style.colorScheme = next ? "dark" : "light"; setDark(next) }}>{dark ? "Light" : "Dark"}</Button>
    </div>
  </>
}
