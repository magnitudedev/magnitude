import type { ReactNode } from "react"
import { Option } from "effect"
import { ArrowUpRightIcon, CaretDownIcon, EyeIcon, MagnifyingGlassIcon } from "@phosphor-icons/react"
import { localModelServingProfile, type CatalogLocalModel, type LocalModel } from "@magnitudedev/sdk"
import { ActionTooltip, TooltipProvider } from "./ui/tooltip"
import { ModelRadar } from "./discovery-visuals"

/** A catalog model's Hugging Face (or other) source links. */
export function CatalogSources({ model }: { model: CatalogLocalModel }) {
  if (model.presentation.sourceUrls.length === 0) return null
  return <div><p className="mb-2 text-xs text-slate-500">Sources</p><div className="flex flex-wrap gap-x-4 gap-y-2">{model.presentation.sourceUrls.map(url => {
    const source = new URL(url)
    const label = source.hostname === "huggingface.co" ? `Hugging Face · ${source.pathname.split("/")[1]}` : source.hostname.replace(/^www\./, "")
    return <a className="inline-flex items-center gap-1 text-sm text-slate-600 hover:underline dark:text-slate-300" key={url} href={url} title={url} target="_blank" rel="noreferrer">{label}<ArrowUpRightIcon aria-hidden="true" className="size-3.5" /></a>
  })}</div></div>
}

/**
 * One layout for every model's details, catalog or custom: the drafter, the source, then License and Context
 * window on the left, and with `radar` the capability radar on the right. `onFindOtherVersions` adds Other versions.
 */
export function ModelDetails({ model, drafter, source, radar = false, open, contentId, compact = false, onFindOtherVersions }: {
  model: LocalModel
  drafter?: ReactNode
  source?: ReactNode
  radar?: boolean
  open?: boolean
  contentId?: string
  compact?: boolean
  onFindOtherVersions?: () => void
}) {
  const serving = model.servingState
  const content = (
    <div className={compact ? "grid gap-5 text-sm" : "mt-3 grid items-start gap-8 border-t border-slate-200 pt-5 dark:border-slate-750 lg:grid-cols-2"}>
      <div className="min-w-0 space-y-5">
        {drafter}
        {source}
        <dl className="flex flex-wrap gap-x-8 gap-y-3">
          <div><dt className="text-xs text-slate-500">License</dt><dd className="mt-1">{Option.getOrElse(model.presentation.license, () => "Not specified")}</dd></div>
          {Option.match(localModelServingProfile(model), { onNone: () => null, onSome: (profile) => <div><dt className="text-xs text-slate-500">Context window</dt><dd className="mt-1">{profile.contextLength.toLocaleString()} tokens</dd></div> })}
          {serving._tag === "Assessed" && serving.capabilities.vision && <div className="self-end"><TooltipProvider><ActionTooltip label="Supports vision" trigger={<button type="button" aria-label="Supports vision" className="rounded p-1 text-slate-500 hover:text-slate-800 focus-visible:outline-2 focus-visible:outline-blue-500 dark:hover:text-slate-200"><EyeIcon aria-hidden="true" className="size-4" /></button>} /></TooltipProvider></div>}
        </dl>
        {onFindOtherVersions && <div><p className="mb-2 text-xs text-slate-500">Other versions</p>
          <button type="button" onClick={onFindOtherVersions} className="inline-flex items-center gap-1.5 rounded text-sm text-slate-600 hover:underline focus-visible:outline-2 focus-visible:outline-blue-500 dark:text-slate-300"><MagnifyingGlassIcon aria-hidden="true" className="size-3.5" />Find other versions on Hugging Face</button>
        </div>}
        {serving._tag === "Failed" && <div className="min-w-0"><p className="mb-1 text-xs text-slate-500">Assessment</p><p className="break-words text-slate-600 dark:text-slate-300">{serving.failure.message}</p></div>}
      </div>
      {radar && <ModelRadar model={model} />}
    </div>
  )
  if (open !== undefined) return open ? <div id={contentId}>{content}</div> : null
  return <details className="group mt-4 text-sm">
    <summary className="flex cursor-pointer list-none items-center justify-end gap-1 rounded py-1 text-slate-600 focus-visible:outline-2 focus-visible:outline-blue-500 dark:text-slate-300 [&::-webkit-details-marker]:hidden">Model details<CaretDownIcon aria-hidden="true" className="size-4 group-open:rotate-180" /></summary>
    {content}
  </details>
}
