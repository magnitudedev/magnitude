import type { ReactNode } from "react"
import { CaretRightIcon, EyeIcon } from "@phosphor-icons/react"
import { formatLocalModelDisplayName, unsupportedLocalModelSourceTag, unsupportedReasonMessages } from "@magnitudedev/client-common"
import type { CustomLocalModel, UnsupportedLocalModel } from "@magnitudedev/sdk"
import { Button } from "./ui/button"
import { ModelLogo, NamedLogo, labForName } from "./model-logo"
import { pageLayout } from "./page-layout"

/** Where a model's files came from: "Hugging Face", "Ollama", "LM Studio", "Hugging Face cache", or a folder name. */
export function SourceTag({ label }: { label: string }) {
  return <span className="inline-flex shrink-0 items-center rounded border border-slate-300 px-1.5 py-px font-sans text-[11px] font-medium leading-4 text-slate-600 dark:border-slate-700 dark:text-slate-300">{label}</span>
}

export const unsupportedSummary = (count: number) =>
  `${count} other ${count === 1 ? "model" : "models"} on this computer can’t run in Magnitude`

/** One collapsed line at the bottom of My Models; it explains why an expected model is missing and offers no actions. */
export function UnsupportedModels({ models }: { models: readonly UnsupportedLocalModel[] }) {
  if (models.length === 0) return null
  return <details className="group mt-6 text-sm">
    <summary className="flex w-fit cursor-pointer list-none items-center gap-1.5 rounded py-1 text-slate-500 focus-visible:outline-2 focus-visible:outline-blue-500 [&::-webkit-details-marker]:hidden">
      <CaretRightIcon aria-hidden="true" className="size-3.5 group-open:rotate-90" />{unsupportedSummary(models.length)}
    </summary>
    <ul aria-label="Models that can’t run" className="mt-2 divide-y divide-slate-200 rounded-xl border border-slate-200 dark:divide-slate-750 dark:border-slate-750">
      {models.map(model => <li key={model.path} title={model.detail} className="flex flex-wrap items-center gap-x-4 gap-y-1 px-4 py-2.5 text-slate-500 dark:text-slate-400">
        <span className="flex min-w-0 items-center gap-3 sm:w-80"><span className="opacity-60"><NamedLogo lab={labForName(model.displayName)} name={model.displayName} className="size-5" /></span><span className="truncate" title={model.path}>{model.displayName}</span></span>
        <SourceTag label={unsupportedLocalModelSourceTag(model)} />
        <span className="min-w-0 flex-1">{unsupportedReasonMessages(model.reasons).join(". ")}</span>
      </li>)}
    </ul>
  </details>
}

/** Unhide, disabled while the command is pending so a second click can't send it twice. */
export function UnhideButton({ pending, onUnhide }: { pending: boolean; onUnhide: () => void }) {
  return <Button variant="outline" disabled={pending} onClick={onUnhide}><EyeIcon />Unhide</Button>
}

/** The Hidden filter: hidden models stay reachable, each with its Unhide action. */
export function HiddenModels({ models, unhide }: { models: readonly CustomLocalModel[]; unhide: (model: CustomLocalModel) => ReactNode }) {
  if (models.length === 0) return <p className="py-8 text-slate-500">No hidden models. Models you hide from Magnitude appear here.</p>
  return <div className="grid items-start gap-5">{models.map(model => <article key={model.modelId} className={pageLayout.modelCard}>
    <div className={pageLayout.modelRow}>
      <div className="flex min-w-0 items-center gap-4"><ModelLogo model={model} /><div className="min-w-0">
        <h2 className="flex flex-wrap items-center gap-2 text-lg font-semibold">{formatLocalModelDisplayName(model)}<SourceTag label={model.presentation.sourceLabel} /></h2>
        <p className="mt-1 text-sm text-slate-500">Hidden from Magnitude</p>
      </div></div>
      <div className="flex justify-end">{unhide(model)}</div>
    </div>
  </article>)}</div>
}
