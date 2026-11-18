import { useState, type ReactNode } from "react"
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
 * the first opens again only once it isn't. Choosing another model returns to the quant recommended for it.
 */
export function useRecommendedSelection(recommendations: readonly Recommendation[]) {
  const [selection, setSelection] = useState<string | null | undefined>(undefined)
  const [quantChoice, setQuantChoice] = useState<string | null>(null)
  const selected = selectedModel(selection, recommendations)
  const picked = recommendations.find(entry => entry.model.key === selected)?.model
  const pickedQuant = picked ? picked.quants.find(quant => quant.key === quantChoice) ?? picked.quants.find(quant => quant.key === picked.recommended) : undefined
  return {
    selected, quantChoice, picked, pickedQuant,
    toggle: (key: string) => { setSelection(selected === key ? null : key); setQuantChoice(null) },
    chooseQuant: setQuantChoice,
  }
}

export type RecommendedSelection = ReturnType<typeof useRecommendedSelection>

/**
 * How far measuring has come, beside the filter: a few words with the count and a thin bar. The words give way to the
 * bar alone where the row is narrow, so the row never grows taller.
 */
function MeasuringStatus({ measuring }: { measuring: RecommendationsView["measuring"] }) {
  if (measuring === null || measuring.total <= measuring.settled) return null
  return <span role="status" className="flex min-w-0 items-center gap-2 whitespace-nowrap text-slate-500">
    <span className="truncate max-lg:sr-only">Measuring speeds on this computer · {measuring.settled} of {measuring.total}</span>
    <span aria-hidden="true" className="h-1 w-16 shrink-0 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-750">
      <span className="block h-full rounded-full bg-blue-600 transition-[width] duration-500 dark:bg-blue-400" style={{ width: `${Math.max(0.04, measuring.settled / measuring.total) * 100}%` }} />
    </span>
  </span>
}

/**
 * "Recommended for your machine", as onboarding and an empty Models page show it: the models that run well here, the
 * first opened to its quant ladder and placement, with a filter to US developers' models. `placeholder` stands in for
 * the list until it can be shown, which is once every quant has been measured, so no row appears or moves as
 * measuring goes on; how far it has come is said in the title's row. `footer` adds an opened model's own actions,
 * such as its Download.
 */
export function RecommendedForYourMachine({ recommendations, placeholder, measuring, usOnly, setUsOnly, owned, selection, footer, className }: {
  recommendations: readonly Recommendation[]; placeholder: ReactNode | null; measuring: RecommendationsView["measuring"]
  usOnly: boolean; setUsOnly: (value: boolean) => void; owned: ReadonlySet<string>; selection: RecommendedSelection
  footer?: (model: OfferedModel, quant: QuantView) => ReactNode; className?: string
}) {
  return <Section title="Recommended for your machine" {...(className === undefined ? {} : { className })}
    aside={<>
      <MeasuringStatus measuring={measuring} />
      <label className="flex shrink-0 cursor-pointer items-center gap-2 whitespace-nowrap text-xs text-slate-600 dark:text-slate-400"><input type="checkbox" checked={usOnly} onChange={event => setUsOnly(event.target.checked)} className="accent-blue-600" />Only models from US developers</label>
    </>}>
    {placeholder ?? <div className="space-y-2">
        {recommendations.map(({ model, note }) => <ExpandingModel key={model.key} model={model} note={note} expanded={selection.selected === model.key} owned={owned}
          onToggle={() => selection.toggle(model.key)}
          quantChoice={selection.selected === model.key ? selection.quantChoice : null} onQuantChoice={selection.chooseQuant}
          {...(footer ? { footer: (quant: QuantView) => footer(model, quant) } : {})} />)}
        {recommendations.length === 0 && <p className="text-[13px] text-slate-500">No model runs well on this computer yet.</p>}
      </div>}
  </Section>
}
