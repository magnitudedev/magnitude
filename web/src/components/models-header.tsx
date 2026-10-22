import type { ReactNode } from "react"
import { MagnifyingGlassIcon } from "@phosphor-icons/react"
import { Input } from "./ui/input"
import { pageLayout } from "./page-layout"
import { translateKnownMessage, useI18n } from "../i18n"

/** "Search 39 models…": the count of models the selects leave, before any typed search. */
export const searchPlaceholder = (count: number) => `Search ${count} ${count === 1 ? "model" : "models"}…`

/** The spoken match count; it follows the typed search too. */
export const matchStatus = (count: number) => `${count} ${count === 1 ? "model" : "models"}`

/**
 * Catalog and My Models share one header: the title with the page's add action on the right, then
 * search on the left and the selects on the right. The count lives in the search placeholder.
 */
export function ModelsHeader({ title, action, matchingCount, search, onSearch, controls }: {
  title: string
  action: ReactNode
  selectedCount: number
  matchingCount: number
  search: string
  onSearch: (value: string) => void
  controls: ReactNode
}) {
  const { language, t } = useI18n()
  const translatedTitle = translateKnownMessage(title, language)
  return <>
    <div className={pageLayout.modelHeader}>
      <h1 className={pageLayout.pageTitle}>{translatedTitle}</h1>
      {action}
    </div>
    <div className={pageLayout.catalogToolbar}>
      <div className={`relative ${pageLayout.modelSearch}`}>
        <MagnifyingGlassIcon aria-hidden="true" className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-slate-500" />
        <Input aria-label={t("searchModels")} placeholder={t("searchModelsPlaceholder")} className="pl-8" value={search} onChange={event => onSearch(event.target.value)} />
      </div>
      <div className="flex flex-wrap items-center gap-2">{controls}</div>
      <span className="sr-only" role="status">{matchStatus(matchingCount)}</span>
    </div>
  </>
}
