import { useEffect, useRef, useState, type ReactNode } from "react"
import { CircleNotchIcon, InfoIcon, MagnifyingGlassIcon, XIcon } from "@phosphor-icons/react"
import { cn } from "@/lib/utils"
import { Button } from "../ui/button"
import { Input } from "../ui/input"
import { HuggingFaceLogo, LabLogo, type ModelLab } from "../model-logo"
import { Badge, LabMark, cardClass } from "./parts"
import { OfferedModelsSkeleton } from "./skeletons"
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

/** The one place that says which models Magnitude runs, beside the Explore models title: by lab and generation, fine-tunes included. */
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

/** A settled search result as its row shows it: offered with its best quant, or unchecked to retry. */
export type ExploreRow =
  | { readonly _tag: "Unchecked"; readonly key: string; readonly name: string; readonly lab: ModelLab | undefined; readonly gated: boolean; readonly retrying: boolean; readonly retry: () => void }
  | { readonly _tag: "Offered"; readonly model: OfferedModel; readonly prefetched: boolean }

/** What follows the settled rows: checking/loading more, scrolling for more (`Auto`), clicking for more (`Manual`), or the end (`null`). */
export type ExploreMore =
  | { readonly _tag: "Loading" }
  | { readonly _tag: "Checking" }
  | { readonly _tag: "Auto"; readonly onMore: () => void }
  | { readonly _tag: "Manual"; readonly onMore: () => void }
  | null

export type ExploreResults =
  | { readonly _tag: "Idle" }
  | { readonly _tag: "Searching" }
  | { readonly _tag: "Results"; readonly rows: readonly ExploreRow[]; readonly more: ExploreMore }
  | { readonly _tag: "Failed"; readonly message: string }

/** The end of a batch of results: reaching it asks for the next batch. */
function NextPage({ onReach }: { onReach: () => void }) {
  const ref = useRef<HTMLDivElement>(null)
  const reach = useRef(onReach)
  reach.current = onReach
  useEffect(() => {
    const target = ref.current
    if (!target) return
    const observer = new IntersectionObserver(entries => { if (entries.some(entry => entry.isIntersecting)) reach.current() })
    observer.observe(target)
    return () => observer.disconnect()
  }, [])
  return <div ref={ref} className="mt-2 flex justify-center"><Button variant="ghost" size="sm" onClick={() => reach.current()}>More results</Button></div>
}

/** Hover dwell before warming a collapsed repository's full quant list. */
const HOVER_PREFETCH_MS = 120

/**
 * An offered model as opened: what opening it brings (a search result's every quant) is the model it shows. A
 * `prefetched` model is opened before it is shown open. Hovering or focusing a collapsed Hugging Face card warms its
 * quant list ahead of a click.
 */
function OfferedCard({ model, expanded, prefetched, onPrefetch, download, drafters, useOpened, ...rest }: Omit<Parameters<typeof ExpandingModel>[0], "footer" | "drafter"> & {
  prefetched: boolean; onPrefetch?: ((key: string) => void) | undefined; download: (model: OfferedModel, quant: QuantView) => ReactNode; useOpened: (model: OfferedModel, open: boolean) => OfferedModel
  drafters: ((model: OfferedModel, quant: QuantView) => ReactNode) | undefined
}) {
  const opened = useOpened(model, expanded || prefetched)
  const hoverTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const cancelHover = () => {
    if (hoverTimer.current !== null) {
      clearTimeout(hoverTimer.current)
      hoverTimer.current = null
    }
  }
  useEffect(() => cancelHover, [])
  const warm = () => {
    if (model.catalog || expanded || prefetched || !onPrefetch) return
    cancelHover()
    hoverTimer.current = setTimeout(() => {
      hoverTimer.current = null
      onPrefetch(model.key)
    }, HOVER_PREFETCH_MS)
  }
  const warmNow = () => {
    if (model.catalog || expanded || prefetched || !onPrefetch) return
    cancelHover()
    onPrefetch(model.key)
  }
  return <div onPointerEnter={warm} onPointerLeave={cancelHover} onFocus={warmNow}>
    <ExpandingModel {...rest} model={opened} expanded={expanded} drafter={drafters && (quant => drafters(opened, quant))} footer={quant => download(opened, quant)} />
  </div>
}

/** A search result whose check failed or ran out of time: its name, why when it is gated, and a way to check it again. */
function UncheckedCard({ name, lab, gated, retrying, retry }: { name: string; lab: ModelLab | undefined; gated: boolean; retrying: boolean; retry: () => void }) {
  return <div className={cn(cardClass, "grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-4 px-4 py-3")}>
    <LabMark lab={lab} className="size-10" />
    <span className="min-w-0">
      <span className="block truncate text-[14px] font-medium">{name}</span>
      <span className="mt-0.5 block text-xs text-slate-500">{gated
        ? "Gated on Hugging Face. Accept its terms there, signed in to Hugging Face in Magnitude, to check how it runs here."
        : "Couldn't check how this runs here."}</span>
    </span>
    <Button size="sm" variant="outline" disabled={retrying} onClick={retry}>
      {retrying ? <><CircleNotchIcon aria-hidden="true" className="motion-safe:animate-spin" />Checking…</> : "Try again"}
    </Button>
  </div>
}

/** Before a search: what the results will be and where they come from. */
export function ExploreIdle() {
  return <div className={cn(cardClass, "flex flex-col items-center gap-3 px-6 py-12 text-center")}>
    <HuggingFaceLogo className="size-8" />
    <p className="text-[13px] text-slate-500">Results from Hugging Face show here</p>
  </div>
}

/** Explore models, a view of the Models page: a search of Hugging Face, each result opening in place to its quants. */
export function ExploreView({ query, setQuery, results, families, owned, download, drafters, useOpened, onExpand, onPrefetch }: {
  query: string; setQuery: (value: string) => void
  results: ExploreResults; families: readonly RunnableFamily[]; owned: ReadonlySet<string>; download: (model: OfferedModel, quant: QuantView) => ReactNode
  drafters?: (model: OfferedModel, quant: QuantView) => ReactNode
  useOpened: (model: OfferedModel, open: boolean) => OfferedModel
  onExpand?: (key: string | null) => void
  onPrefetch?: (key: string) => void
}) {
  const [chosen, setExpanded] = useState<string | null | undefined>(undefined)
  const [quantChoices, setQuantChoices] = useState<Readonly<Record<string, string>>>({})
  const expanded = chosen ?? null
  const toggleExpanded = (model: OfferedModel) => {
    const next = expanded === model.key ? null : model.key
    setExpanded(next)
    onExpand?.(next !== null && !model.catalog ? next : null)
  }
  const card = (model: OfferedModel, note?: string, prefetched = false) => <OfferedCard key={model.key} model={model} note={note} expanded={expanded === model.key} owned={owned}
    prefetched={prefetched} onPrefetch={onPrefetch} onToggle={() => toggleExpanded(model)} quantChoice={quantChoices[model.key] ?? null}
    onQuantChoice={key => setQuantChoices(choices => ({ ...choices, [model.key]: key }))} download={download} drafters={drafters} useOpened={useOpened} />
  const body = <>
    {results._tag === "Idle" && <ExploreIdle />}
    {results._tag === "Searching" && <>
      <SearchStatus>Searching Hugging Face…</SearchStatus>
      <OfferedModelsSkeleton />
    </>}
    {results._tag === "Failed" && <p className="text-[13px] text-red-600 dark:text-red-400">{results.message}</p>}
    {results._tag === "Results" && <>
      {results.rows.length > 0 ? <>
        <div className="space-y-2">{results.rows.map(row => row._tag === "Unchecked"
          ? <UncheckedCard key={row.key} name={row.name} lab={row.lab} gated={row.gated} retrying={row.retrying} retry={row.retry} />
          : card(row.model, undefined, row.prefetched))}</div>
        {results.more?._tag === "Loading" ? <div className="mt-2"><SearchStatus className="mt-4">Loading more results…</SearchStatus><OfferedModelsSkeleton count={2} /></div>
          : results.more?._tag === "Checking" ? <div className="mt-2"><SearchStatus className="mt-4">Checking more models…</SearchStatus><OfferedModelsSkeleton count={2} /></div>
          : results.more?._tag === "Auto" ? <NextPage onReach={results.more.onMore} />
          : results.more?._tag === "Manual" ? <div className="mt-2 flex justify-center"><Button variant="ghost" size="sm" onClick={results.more.onMore}>More results</Button></div>
          : null}
      </> : results.more?._tag === "Loading" || results.more?._tag === "Checking" ? <>
        <SearchStatus>Checking models…</SearchStatus>
        <OfferedModelsSkeleton />
      </> : results.more?._tag === "Manual" ? <div>
        <p className="max-w-lg text-[13px] text-slate-500">
          No models in the most downloaded Hugging Face results for “{query.trim()}” run in Magnitude yet. {runnableSummary(families)}
        </p>
        <div className="mt-3"><Button variant="outline" size="sm" onClick={results.more.onMore}>Check more results</Button></div>
      </div> : <p className="max-w-lg text-[13px] text-slate-500">
        No models on Hugging Face that run in Magnitude match “{query.trim()}”. {runnableSummary(families)}
      </p>}
    </>}
  </>
  return <div>
    <div className="relative mb-5">
      <MagnifyingGlassIcon className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-slate-400" />
      <Input autoFocus value={query} onChange={event => { setQuery(event.target.value); setExpanded(undefined); onExpand?.(null) }} placeholder="Search Hugging Face, or paste a link" className="h-10 bg-white pl-9 text-sm dark:bg-slate-850" />
      {query && <button type="button" aria-label="Clear search" onClick={() => { setQuery(""); setExpanded(undefined); onExpand?.(null) }} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"><XIcon className="size-4" /></button>}
    </div>
    {body}
  </div>
}
