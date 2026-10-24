import { useId, type ReactNode } from "react"
import { MagnifyingGlassIcon } from "@phosphor-icons/react"
import { Input } from "./ui/input"
import { pageLayout } from "./page-layout"

/** The two views share one catalog, search and result count. */
export function DiscoverHeader({ view, count, search, onView, onSearch, action, hardware, notice }: {
  view: "recommended" | "all"
  count: number | null
  search: string
  onView: (view: "recommended" | "all") => void
  onSearch: (search: string) => void
  action: ReactNode
  hardware: ReactNode
  notice?: ReactNode
}) {
  const id = useId()
  const views = [["recommended", "Recommended"], ["all", "All models"]] as const
  return <>
    <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
      <h1 className={pageLayout.pageTitle}>Discover</h1>
      {action}
    </div>
    {hardware}
    {notice}
    <div className="mb-6 flex flex-wrap items-center justify-between gap-x-6 gap-y-4 border-b border-slate-200 dark:border-slate-750">
      <div role="tablist" aria-label="Discover views" className="flex gap-6">
        {views.map(([key, label], index) => <button key={key} id={`${id}-${key}`} type="button" role="tab" aria-selected={view === key} tabIndex={view === key ? 0 : -1}
          onClick={() => onView(key)}
          onKeyDown={event => {
            const next = event.key === "Home" ? 0 : event.key === "End" ? 1 : event.key === "ArrowLeft" || event.key === "ArrowRight" ? 1 - index : null
            if (next === null) return
            event.preventDefault()
            onView(views[next]![0])
            event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus()
          }}
          className={`-mb-px flex h-11 items-center gap-2 border-b-2 text-sm font-medium transition-colors focus-visible:outline-2 focus-visible:outline-blue-500 ${view === key ? "border-blue-600 text-slate-900 dark:border-blue-400 dark:text-slate-100" : "border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"}`}>
          {label}
          {key === "all" && <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] tabular-nums text-slate-500 dark:bg-slate-800 dark:text-slate-400">{count ?? "…"}</span>}
        </button>)}
      </div>
      <div className="relative mb-2 w-full sm:w-64">
        <MagnifyingGlassIcon className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-slate-500" />
        <Input aria-label="Search all models" placeholder={count === null ? "Search models…" : `Search ${count} models…`} className="pl-8" value={search} onChange={event => onSearch(event.target.value)} />
      </div>
    </div>
  </>
}
