import type { ReactNode } from "react"
import { Option } from "effect"
import { ArrowUpRightIcon, EyeIcon } from "@phosphor-icons/react"
import { localModelServingProfile, type CatalogLocalModel, type LocalModel } from "@magnitudedev/sdk"
import { ActionTooltip, TooltipProvider } from "./ui/tooltip"

/** A catalog model's Hugging Face (or other) source links. */
export function CatalogSources({ model }: { model: CatalogLocalModel }) {
  if (model.presentation.sourceUrls.length === 0) return null
  return <div><p className="mb-2 text-xs text-slate-500">Sources</p><div className="flex flex-wrap gap-x-4 gap-y-2">{model.presentation.sourceUrls.map(url => {
    const source = new URL(url)
    const label = source.hostname === "huggingface.co" ? `Hugging Face · ${source.pathname.split("/")[1]}` : source.hostname.replace(/^www\./, "")
    return <a className="inline-flex items-center gap-1 text-sm text-slate-600 hover:underline dark:text-slate-300" key={url} href={url} title={url} target="_blank" rel="noreferrer">{label}<ArrowUpRightIcon aria-hidden="true" className="size-3.5" /></a>
  })}</div></div>
}

/** A Discover recommendation's Details view: its sources, then License and Context window, and a vision mark. */
export function ModelDetails({ model, source }: { model: LocalModel; source?: ReactNode }) {
  const serving = model.servingState
  return <div className="grid gap-5 text-sm">
    {source}
    <dl className="flex flex-wrap gap-x-8 gap-y-3">
      <div><dt className="text-xs text-slate-500">License</dt><dd className="mt-1">{Option.getOrElse(model.presentation.license, () => "Not specified")}</dd></div>
      {Option.match(localModelServingProfile(model), { onNone: () => null, onSome: (profile) => <div><dt className="text-xs text-slate-500">Context window</dt><dd className="mt-1">{profile.contextLength.toLocaleString()} tokens</dd></div> })}
      {serving._tag === "Assessed" && serving.capabilities.vision && <div className="self-end"><TooltipProvider><ActionTooltip label="Supports vision" trigger={<button type="button" aria-label="Supports vision" className="rounded p-1 text-slate-500 hover:text-slate-800 focus-visible:outline-2 focus-visible:outline-blue-500 dark:hover:text-slate-200"><EyeIcon aria-hidden="true" className="size-4" /></button>} /></TooltipProvider></div>}
    </dl>
    {serving._tag === "Failed" && <div className="min-w-0"><p className="mb-1 text-xs text-slate-500">Assessment</p><p className="break-words text-slate-600 dark:text-slate-300">{serving.failure.message}</p></div>}
  </div>
}
