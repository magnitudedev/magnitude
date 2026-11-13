import { useEffect, useRef, useState, type ReactNode } from "react"
import { CircleNotchIcon, InfoIcon, MagnifyingGlassIcon, XIcon } from "@phosphor-icons/react"
import { cn } from "@/lib/utils"
import { Button } from "../ui/button"
import { Input } from "../ui/input"
import { Dialog, DialogContent, DialogTitle } from "../ui/dialog"
import { LabLogo, type ModelLab } from "../model-logo"
import { Badge, LabMark, cardClass } from "./parts"
import { SkeletonLine } from "../page-skeletons"
import { MeasuringRecommendations, OfferedModelsSkeleton } from "./skeletons"
import { ExpandingModel, type OfferedModel } from "./expanding-model"
import type { QuantView } from "./view"

/** A family Magnitude runs, as the "What runs in Magnitude" list names it. */
export interface RunnableFamily {
  readonly lab: ModelLab | undefined
  readonly models: string
  readonly comingSoon: boolean
}

export const runnableSummary = (families: readonly RunnableFamily[]) =>
  `Magnitude runs models built on ${families.filter(family => !family.comingSoon).map(family => family.models).join(", ")}, including their fine-tunes.`

/** The one place that says which models Magnitude runs: by lab and generation, fine-tunes included. */
export function WhatRuns({ families }: { families: readonly RunnableFamily[] }) {
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
      <ul className="space-y-2">{families.map(family => <li key={family.models} className="flex items-center gap-2.5 text-[13px]">
        {family.lab && <LabLogo lab={family.lab} className="size-4" />}<span className="flex-1">{family.models}</span>{family.comingSoon && <Badge tone="amber">Coming soon</Badge>}
      </li>)}</ul>
      <p className="mt-3 border-t border-slate-200 pt-3 text-xs text-slate-500 dark:border-slate-750">Any quant or fine-tune of these from Hugging Face runs too.</p>
    </div>}
  </div>
}

/** What search is doing now: a spinner and a status line read out as it changes. */
function SearchStatus({ children, className }: { children: ReactNode; className?: string }) {
  return <p role="status" aria-live="polite" className={cn("mb-3 flex items-center gap-1.5 text-xs font-medium text-slate-500", className)}>
    <CircleNotchIcon aria-hidden="true" className="size-3.5 shrink-0 motion-safe:animate-spin" />{children}
  </p>
}

/** A search result as its row shows it: while its check runs, its name alone; then the model it offers. */
export type ExploreRow =
  | { readonly _tag: "Checking"; readonly key: string; readonly name: string; readonly lab: ModelLab | undefined }
  | { readonly _tag: "Unchecked"; readonly key: string; readonly name: string; readonly lab: ModelLab | undefined; readonly gated: boolean; readonly retry: () => void }
  | { readonly _tag: "Offered"; readonly model: OfferedModel; readonly prefetched: boolean }

export type ExploreResults =
  | { readonly _tag: "Recommended"; readonly models: readonly OfferedModel[]; readonly machine: string; readonly measuring: { readonly settled: number; readonly total: number } | null }
  | { readonly _tag: "Searching" }
  | { readonly _tag: "Results"; readonly rows: readonly ExploreRow[]; readonly more: "Loading" | (() => void) | null }
  | { readonly _tag: "Failed"; readonly message: string }

/** The end of a page of results: reaching it asks for the next page. */
function NextPage({ onReach }: { onReach: () => void }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const target = ref.current
    if (!target) return
    const observer = new IntersectionObserver(entries => { if (entries.some(entry => entry.isIntersecting)) onReach() })
    observer.observe(target)
    return () => observer.disconnect()
  }, [onReach])
  return <div ref={ref} className="mt-2 flex justify-center"><Button variant="ghost" size="sm" onClick={onReach}>More results</Button></div>
}

/**
 * An offered model as opened: what opening it brings (a search result's every quant) is the model it shows. A
 * `prefetched` model is opened before it is shown open.
 */
function OfferedCard({ model, expanded, prefetched, download, drafters, useOpened, ...rest }: Omit<Parameters<typeof ExpandingModel>[0], "footer" | "drafter"> & {
  prefetched: boolean; download: (model: OfferedModel, quant: QuantView) => ReactNode; useOpened: (model: OfferedModel, open: boolean) => OfferedModel
  drafters: ((model: OfferedModel, quant: QuantView) => ReactNode) | undefined
}) {
  const opened = useOpened(model, expanded || prefetched)
  return <ExpandingModel {...rest} model={opened} expanded={expanded} drafter={drafters && (quant => drafters(opened, quant))} footer={quant => download(opened, quant)} />
}

/** Reports whether its row is on screen, so the rows in view are checked first. */
function Seen({ id, onSeen, children }: { id: string; onSeen: (update: (seen: ReadonlySet<string>) => ReadonlySet<string>) => void; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const target = ref.current
    if (!target) return
    const change = (shown: boolean) => onSeen(seen => {
      if (seen.has(id) === shown) return seen
      const next = new Set(seen)
      if (shown) next.add(id)
      else next.delete(id)
      return next
    })
    const observer = new IntersectionObserver(entries => change(entries.some(entry => entry.isIntersecting)))
    observer.observe(target)
    return () => { observer.disconnect(); change(false) }
  }, [id, onSeen])
  return <div ref={ref}>{children}</div>
}

/** A search result whose check failed or ran out of time: its name, why when it is gated, and a way to check it again. */
function UncheckedCard({ name, lab, gated, retry }: { name: string; lab: ModelLab | undefined; gated: boolean; retry: () => void }) {
  return <div className={cn(cardClass, "grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-4 px-4 py-3")}>
    <LabMark lab={lab} className="size-10" />
    <span className="min-w-0">
      <span className="block truncate text-[14px] font-medium">{name}</span>
      <span className="mt-0.5 block text-xs text-slate-500">{gated
        ? "Gated on Hugging Face. Accept its terms there, signed in to Hugging Face in Magnitude, to check how it runs here."
        : "Couldn't check how this runs here."}</span>
    </span>
    <Button size="sm" variant="outline" onClick={retry}>Try again</Button>
  </div>
}

/** A search result whose check is still running, laid out as the card it becomes: its name, and what is being checked. */
function CheckingCard({ name, lab }: { name: string; lab: ModelLab | undefined }) {
  return <div className={cn(cardClass, "grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-4 px-4 py-3")}>
    <LabMark lab={lab} className="size-10" />
    <span className="min-w-0">
      <span className="block truncate text-[14px] font-medium">{name}</span>
      <span role="status" className="mt-0.5 flex items-center gap-1.5 text-xs text-slate-500"><CircleNotchIcon aria-hidden="true" className="size-3.5 shrink-0 motion-safe:animate-spin" />Checking how it runs here…</span>
    </span>
    <span className="grid grid-cols-[84px_64px] gap-4 pr-9">
      <span><SkeletonLine className="ml-auto h-5 text-[15px]" width="3.5rem" /><SkeletonLine className="ml-auto h-4 text-[11px]" width="4rem" /></span>
      <span><SkeletonLine className="ml-auto h-5 text-[15px]" width="3rem" /><SkeletonLine className="ml-auto h-4 text-[11px]" width="2.5rem" /></span>
    </span>
  </div>
}

/** Exploring Hugging Face in a modal: search and the best models for this machine, each opening in place to its quants. */
export function ExploreModal({ open, setOpen, query, setQuery, results, families, owned, download, drafters, useOpened, onSeen }: {
  open: boolean; setOpen: (open: boolean) => void; query: string; setQuery: (value: string) => void
  results: ExploreResults; families: readonly RunnableFamily[]; owned: ReadonlySet<string>; download: (model: OfferedModel, quant: QuantView) => ReactNode
  drafters?: (model: OfferedModel, quant: QuantView) => ReactNode
  useOpened: (model: OfferedModel, open: boolean) => OfferedModel
  onSeen: (update: (seen: ReadonlySet<string>) => ReadonlySet<string>) => void
}) {
  const [chosen, setExpanded] = useState<string | null | undefined>(undefined)
  const [quantChoices, setQuantChoices] = useState<Readonly<Record<string, string>>>({})
  const offered = results._tag === "Results" ? results.rows.flatMap(row => row._tag === "Offered" ? [row.model] : []) : []
  const checking = results._tag === "Results" && results.rows.some(row => row._tag === "Checking")
  const expanded = chosen === undefined ? (results._tag === "Recommended" && results.measuring === null ? results.models[0]?.key ?? null : null) : chosen
  const card = (model: OfferedModel, note?: string, prefetched = false) => <OfferedCard key={model.key} model={model} note={note} expanded={expanded === model.key} owned={owned}
    prefetched={prefetched} onToggle={() => setExpanded(expanded === model.key ? null : model.key)} quantChoice={quantChoices[model.key] ?? null}
    onQuantChoice={key => setQuantChoices(choices => ({ ...choices, [model.key]: key }))} download={download} drafters={drafters} useOpened={useOpened} />
  return <Dialog open={open} onOpenChange={value => { setOpen(value); if (!value) setExpanded(undefined) }}>
    <DialogContent className="flex h-[85vh] flex-col gap-0 overflow-hidden p-0 sm:max-w-4xl">
      <DialogTitle className="sr-only">Explore models</DialogTitle>
      <div className="border-b border-slate-200 p-5 dark:border-slate-750">
        <div className="mb-4 flex items-center justify-between gap-4 pr-8"><h2 className="font-heading text-lg font-semibold">Explore models</h2><WhatRuns families={families} /></div>
        <div className="relative">
          <MagnifyingGlassIcon className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-slate-400" />
          <Input autoFocus value={query} onChange={event => { setQuery(event.target.value); setExpanded(undefined) }} placeholder="Search Hugging Face, or paste a link" className="h-10 pl-9 text-sm" />
          {query && <button type="button" aria-label="Clear search" onClick={() => setQuery("")} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-700"><XIcon className="size-4" /></button>}
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto bg-slate-50 p-5 dark:bg-slate-900/40">
        {results._tag === "Recommended" && <>
          <p className="mb-3 flex items-baseline justify-between text-xs font-medium text-slate-500"><span>Best for your machine</span><span className="font-normal">{results.machine}</span></p>
          {results.measuring ? <MeasuringRecommendations {...results.measuring} /> : <div className="space-y-2">{results.models.map((model, index) => card(model, index === 0 ? "Smartest that runs well here" : undefined))}</div>}
        </>}
        {results._tag === "Searching" && <>
          <SearchStatus>Searching Hugging Face…</SearchStatus>
          <OfferedModelsSkeleton />
        </>}
        {results._tag === "Failed" && <p className="text-[13px] text-red-600 dark:text-red-400">{results.message}</p>}
        {results._tag === "Results" && <>
          <p className="mb-3 text-xs font-medium text-slate-500">On Hugging Face, running in Magnitude{checking ? "" : ` (${offered.length})`}</p>
          {results.rows.length > 0
            ? <div className="space-y-2">{results.rows.map(row => {
              const key = row._tag === "Offered" ? row.model.key : row.key
              return <Seen key={key} id={key} onSeen={onSeen}>
                {row._tag === "Checking" ? <CheckingCard name={row.name} lab={row.lab} />
                  : row._tag === "Unchecked" ? <UncheckedCard name={row.name} lab={row.lab} gated={row.gated} retry={row.retry} />
                  : card(row.model, undefined, row.prefetched)}
              </Seen>
            })}</div>
            : results.more === null && <p className="max-w-lg text-[13px] text-slate-500">
              No models on Hugging Face that run in Magnitude match “{query.trim()}”. {runnableSummary(families)}
            </p>}
          {results.more === "Loading" ? <div className="mt-2"><SearchStatus className="mt-4">Loading more results…</SearchStatus><OfferedModelsSkeleton count={2} /></div>
            : results.more && <NextPage onReach={results.more} />}
        </>}
      </div>
    </DialogContent>
  </Dialog>
}

