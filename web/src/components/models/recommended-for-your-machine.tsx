import { useState, type ReactNode } from "react"
import { Section } from "./parts"
import { ExpandingModel, type OfferedModel } from "./expanding-model"
import type { Recommendation } from "./recommendations"
import type { QuantView } from "./view"

/**
 * Which recommended model is open and which of its quants is chosen. The first model opens until one is chosen or
 * closed; choosing another model returns to the quant recommended for it.
 */
export function useRecommendedSelection(recommendations: readonly Recommendation[]) {
  const [selection, setSelection] = useState<string | null | undefined>(undefined)
  const [quantChoice, setQuantChoice] = useState<string | null>(null)
  const selected = selection === undefined ? recommendations[0]?.model.key ?? null : selection
  const picked = recommendations.find(entry => entry.model.key === selected)?.model
  const pickedQuant = picked ? picked.quants.find(quant => quant.key === quantChoice) ?? picked.quants.find(quant => quant.key === picked.recommended) : undefined
  return {
    selected, quantChoice, picked, pickedQuant,
    toggle: (key: string) => { setSelection(selected === key ? null : key); setQuantChoice(null) },
    chooseQuant: setQuantChoice,
    reset: () => { setSelection(undefined); setQuantChoice(null) },
  }
}

export type RecommendedSelection = ReturnType<typeof useRecommendedSelection>

/**
 * "Recommended for your machine", as onboarding and an empty Models page show it: the models that run well here, the
 * first opened to its quant ladder and placement, with a filter to US developers' models. `placeholder` stands in for
 * the list while it can't be shown yet; `footer` adds an opened model's own actions, such as its Download.
 */
export function RecommendedForYourMachine({ recommendations, placeholder, usOnly, setUsOnly, owned, selection, footer, className }: {
  recommendations: readonly Recommendation[]; placeholder: ReactNode | null
  usOnly: boolean; setUsOnly: (value: boolean) => void; owned: ReadonlySet<string>; selection: RecommendedSelection
  footer?: (model: OfferedModel, quant: QuantView) => ReactNode; className?: string
}) {
  return <Section title="Recommended for your machine" {...(className === undefined ? {} : { className })}
    aside={<label className="flex cursor-pointer items-center gap-2 text-xs text-slate-600 dark:text-slate-400"><input type="checkbox" checked={usOnly} onChange={event => { setUsOnly(event.target.checked); selection.reset() }} className="accent-blue-600" />Only models from US developers</label>}>
    {placeholder ?? <div className="space-y-2">
        {recommendations.map(({ model, note }) => <ExpandingModel key={model.key} model={model} note={note} expanded={selection.selected === model.key} owned={owned}
          onToggle={() => selection.toggle(model.key)}
          quantChoice={selection.selected === model.key ? selection.quantChoice : null} onQuantChoice={selection.chooseQuant}
          {...(footer ? { footer: (quant: QuantView) => footer(model, quant) } : {})} />)}
        {recommendations.length === 0 && <p className="text-[13px] text-slate-500">No model runs well on this computer yet.</p>}
      </div>}
  </Section>
}
