import { useState, type ReactNode } from "react"
import { CircleNotchIcon } from "@phosphor-icons/react"
import { Section } from "./parts"
import { ExpandingModel, type OfferedModel } from "./expanding-model"
import type { Recommendation, RecommendationsView } from "./recommendations"
import type { QuantView } from "./view"

/**
 * The model open: the one chosen while it is still offered, none once closed, else the first. `undefined` is no
 * choice yet.
 */
export const selectedModel = (selection: string | null | undefined, recommendations: readonly Recommendation[]): string | null =>
  selection === null ? null
    : selection !== undefined && recommendations.some(entry => entry.model.key === selection) ? selection
    : recommendations[0]?.model.key ?? null

/**
 * Which recommended model is open and which of its quants is chosen. The first model opens until one is chosen or
 * closed; a choice holds while its model is still offered, whatever else measuring or a filter adds or takes away, and
 * the first opens again only once it isn't. Choosing another model returns to the quant already on this computer, else
 * the one recommended for it.
 */
export function useRecommendedSelection(recommendations: readonly Recommendation[], owned: ReadonlySet<string> = new Set()) {
  const [selection, setSelection] = useState<string | null | undefined>(undefined)
  const [quantChoice, setQuantChoice] = useState<string | null>(null)
  const selected = selectedModel(selection, recommendations)
  const picked = recommendations.find(entry => entry.model.key === selected)?.model
  const pickedQuant = picked ? picked.quants.find(quant => quant.key === quantChoice) ?? picked.quants.find(quant => owned.has(quant.key)) ?? picked.quants.find(quant => quant.key === picked.recommended) : undefined
  return {
    selected, quantChoice, picked, pickedQuant,
    toggle: (key: string) => { setSelection(selected === key ? null : key); setQuantChoice(null) },
    chooseQuant: setQuantChoice,
  }
}

export type RecommendedSelection = ReturnType<typeof useRecommendedSelection>

/**
 * How far measuring has come: centered over the loading skeleton while recommendations are still settling, or as a
 * compact bar above the list if any remain while models are shown. Keeping it out of the section title row ensures
 * "Recommended for your machine" never wraps.
 */
function MeasuringStatus({ measuring, overlay }: { measuring: NonNullable<RecommendationsView["measuring"]>; overlay: boolean }) {
  const width = `${Math.max(0.04, measuring.settled / measuring.total) * 100}%`
  if (overlay) {
    return <div role="status" className="pointer-events-none absolute inset-0 flex items-center justify-center p-6">
      <div className="w-full max-w-sm rounded-xl border border-slate-200 bg-white/95 px-5 py-4 shadow-sm backdrop-blur-xs dark:border-slate-700 dark:bg-slate-850/95">
        <div className="flex items-center justify-between gap-3 text-[13px]">
          <span className="flex items-center gap-2 font-medium text-slate-900 dark:text-slate-100">
            <CircleNotchIcon className="size-4 shrink-0 text-blue-600 motion-safe:animate-spin dark:text-blue-400" />
            Measuring speeds on this computer
          </span>
          <span className="shrink-0 text-xs tabular-nums text-slate-500">{measuring.settled} of {measuring.total}</span>
        </div>
        <div aria-hidden="true" className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-750">
          <div className="h-full rounded-full bg-blue-600 transition-[width] duration-500 dark:bg-blue-400" style={{ width }} />
        </div>
      </div>
    </div>
  }
  return <div role="status" className="mb-3 flex items-center gap-3 text-xs text-slate-500">
    <CircleNotchIcon className="size-3.5 shrink-0 text-blue-600 motion-safe:animate-spin dark:text-blue-400" />
    <span className="tabular-nums">Measuring speeds on this computer · {measuring.settled} of {measuring.total}</span>
    <span aria-hidden="true" className="h-1 w-24 shrink-0 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-750">
      <span className="block h-full rounded-full bg-blue-600 transition-[width] duration-500 dark:bg-blue-400" style={{ width }} />
    </span>
  </div>
}

/**
 * "Recommended for your machine", as onboarding and an empty Models page show it: the models that run well here, the
 * first opened to its quant ladder and placement, with a filter to US developers' models. `placeholder` stands in for
 * the list until it can be shown, which is once every quant has been measured, so no row appears or moves as
 * measuring goes on; while measuring, progress sits directly inside the placeholder rather than crowding the heading.
 * `footer` adds an opened model's own actions, such as its Download.
 */
export function RecommendedForYourMachine({ recommendations, placeholder, measuring, usOnly, setUsOnly, owned, selection, footer, className }: {
  recommendations: readonly Recommendation[]; placeholder: ReactNode | null; measuring: RecommendationsView["measuring"]
  usOnly: boolean; setUsOnly: (value: boolean) => void; owned: ReadonlySet<string>; selection: RecommendedSelection
  footer?: (model: OfferedModel, quant: QuantView) => ReactNode; className?: string
}) {
  const activeMeasuring = measuring !== null && measuring.total > measuring.settled ? measuring : null
  return <Section title="Recommended for your machine" {...(className === undefined ? {} : { className })}
    aside={<label className="flex shrink-0 cursor-pointer items-center gap-2 whitespace-nowrap text-xs text-slate-600 dark:text-slate-400"><input type="checkbox" checked={usOnly} onChange={event => setUsOnly(event.target.checked)} className="accent-blue-600" />Only models from US developers</label>}>
    {placeholder ? <div className="relative">
      <div className={activeMeasuring ? "opacity-45" : undefined}>{placeholder}</div>
      {activeMeasuring && <MeasuringStatus measuring={activeMeasuring} overlay />}
    </div> : <>
      {activeMeasuring && <MeasuringStatus measuring={activeMeasuring} overlay={false} />}
      <div className="space-y-2">
        {recommendations.map(({ model, note }) => <ExpandingModel key={model.key} model={model} note={note} expanded={selection.selected === model.key} owned={owned}
          onToggle={() => selection.toggle(model.key)}
          quantChoice={selection.selected === model.key ? selection.quantChoice : null} onQuantChoice={selection.chooseQuant}
          {...(footer ? { footer: (quant: QuantView) => footer(model, quant) } : {})} />)}
        {recommendations.length === 0 && <p className="text-[13px] text-slate-500">No model runs well on this computer yet.</p>}
      </div>
    </>}
  </Section>
}
