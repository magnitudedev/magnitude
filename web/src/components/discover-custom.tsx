import { Option } from "effect"
import { ArrowRightIcon, XIcon } from "@phosphor-icons/react"
import type { CustomLocalModel } from "@magnitudedev/sdk"
import { Button } from "./ui/button"
import { HuggingFaceLogo, LabLogo, modelLogoLab, type ModelLab } from "./model-logo"

const joinWithAnd = (values: readonly string[]) =>
  values.length <= 2 ? values.join(" and ") : `${values.slice(0, -1).join(", ")}, and ${values.at(-1)}`

const foundPlace = (model: CustomLocalModel): string => {
  if (model.origin._tag !== "Found") return model.presentation.sourceLabel
  switch (model.origin.source._tag) {
    case "HuggingFaceCache": return "your Hugging Face cache"
    case "Folder": return `your ${model.presentation.sourceLabel} folder`
    default: return model.presentation.sourceLabel
  }
}

/** Models found on this computer that fit it: how many, where they were found, and their labs. */
export const foundModelsSummary = (models: readonly CustomLocalModel[]): Option.Option<{ readonly count: number; readonly places: string; readonly labs: readonly ModelLab[] }> => {
  const fitting = models.filter(model => model.origin._tag === "Found" && model.servingState._tag === "Assessed" && model.servingState.assessment._tag === "Fits")
  if (fitting.length === 0) return Option.none()
  const labs = [...new Set(fitting.flatMap(model => Option.toArray(Option.fromNullable(modelLogoLab(model)))))]
  return Option.some({ count: fitting.length, places: joinWithAnd([...new Set(fitting.map(foundPlace))]), labs })
}

/** The one-line Discover banner for found models; they are acted on in My Models, never ranked here. */
export function FoundModelsBanner({ models, onView, onDismiss, dismissing = false }: { models: readonly CustomLocalModel[]; onView: () => void; onDismiss?: () => void; dismissing?: boolean }) {
  return Option.match(foundModelsSummary(models), {
    onNone: () => null,
    onSome: summary => <section aria-label="Models already on this computer" className="mb-5 flex flex-wrap items-center gap-x-4 gap-y-1 rounded-xl border border-slate-200 bg-white px-4 py-1.5 dark:border-slate-750 dark:bg-slate-850">
      {summary.labs.length > 0 && <span className="flex shrink-0 items-center gap-1.5">{summary.labs.slice(0, 3).map(lab => <LabLogo key={lab.name} lab={lab} className="size-5" />)}</span>}
      <p className="min-w-0 flex-1 text-sm"><span className="font-medium">You already have {summary.count} {summary.count === 1 ? "model that runs" : "models that run"} here.</span> <span className="text-slate-600 dark:text-slate-300">Found in {summary.places}.</span></p>
      <a href="/models" onClick={event => { event.preventDefault(); onView() }} className="inline-flex shrink-0 items-center gap-1.5 border-b border-current pb-0.5 text-[13px] font-medium leading-none text-slate-700 outline-none hover:text-slate-900 focus-visible:outline-2 focus-visible:outline-blue-500 dark:text-slate-300 dark:hover:text-slate-100">View in My Models<ArrowRightIcon aria-hidden="true" className="size-4" /></a>
      {onDismiss && <Button variant="ghost" size="icon-xs" aria-label="Dismiss found models notice" disabled={dismissing} onClick={onDismiss}><XIcon /></Button>}
    </section>,
  })
}

/** The one line under the recommendations that opens the Add dialog over Discover. */
export function HuggingFaceLink({ onAdd }: { onAdd: () => void }) {
  return <p className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-sm text-slate-500">
    <HuggingFaceLogo className="size-4" />
    <span>Looking for a specific version, finetune, or uncensored model?</span>
    <Button variant="link" size="unstyled" className="font-medium" onClick={onAdd}>Add from Hugging Face</Button>
  </p>
}

/** Nothing fits: a smaller community quant may, so Add from Hugging Face sits beside Explore Catalog. */
export function NoRecommendations({ onExplore, onAdd }: { onExplore: () => void; onAdd: () => void }) {
  return <div className="py-8">
    <p className="text-slate-500">No fitting recommendations right now.</p>
    <div className="mt-3 flex flex-wrap items-center gap-2">
      <Button variant="outline" onClick={onExplore}>Browse all models</Button>
      <Button variant="outline" onClick={onAdd}><HuggingFaceLogo />Add from Hugging Face</Button>
    </div>
  </div>
}
