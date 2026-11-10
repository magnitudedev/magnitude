import { useEffect, useRef, useState, type ReactNode } from "react"
import { CircleNotchIcon, InfoIcon, MagnifyingGlassIcon, XIcon } from "@phosphor-icons/react"
import { cn } from "@/lib/utils"
import { Button } from "../ui/button"
import { Input } from "../ui/input"
import { Dialog, DialogContent, DialogTitle } from "../ui/dialog"
import { LabLogo, type ModelLab } from "../model-logo"
import { Badge } from "./parts"
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

export type ExploreResults =
  | { readonly _tag: "Recommended"; readonly models: readonly OfferedModel[]; readonly machine: string; readonly measuring: { readonly settled: number; readonly total: number } | null }
  | { readonly _tag: "Searching" }
  | { readonly _tag: "Results"; readonly models: readonly OfferedModel[]; readonly more: "Loading" | (() => void) | null }
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

/** Exploring Hugging Face in a modal: search and the best models for this machine, each opening in place to its quants. */
export function ExploreModal({ open, setOpen, query, setQuery, results, families, owned, download }: {
  open: boolean; setOpen: (open: boolean) => void; query: string; setQuery: (value: string) => void
  results: ExploreResults; families: readonly RunnableFamily[]; owned: ReadonlySet<string>; download: (model: OfferedModel, quant: QuantView) => ReactNode
}) {
  const [chosen, setExpanded] = useState<string | null | undefined>(undefined)
  const [quantChoices, setQuantChoices] = useState<Readonly<Record<string, string>>>({})
  const listed = (results._tag === "Recommended" && results.measuring === null) || results._tag === "Results" ? results.models : []
  const expanded = chosen === undefined ? (results._tag === "Recommended" ? listed[0]?.key ?? null : null) : chosen
  const card = (model: OfferedModel, note?: string) => <ExpandingModel key={model.key} model={model} note={note} expanded={expanded === model.key} owned={owned}
    onToggle={() => setExpanded(expanded === model.key ? null : model.key)} quantChoice={quantChoices[model.key] ?? null}
    onQuantChoice={key => setQuantChoices(choices => ({ ...choices, [model.key]: key }))} footer={quant => download(model, quant)} />
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
          <SearchStatus>Searching Hugging Face and checking how each result runs on your machine…</SearchStatus>
          <OfferedModelsSkeleton />
        </>}
        {results._tag === "Failed" && <p className="text-[13px] text-red-600 dark:text-red-400">{results.message}</p>}
        {results._tag === "Results" && <>
          <p className="mb-3 text-xs font-medium text-slate-500">On Hugging Face, running in Magnitude ({results.models.length})</p>
          {results.models.length > 0
            ? <div className="space-y-2">{results.models.map(model => card(model))}</div>
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

