import type { ModelFact } from "@magnitudedev/client-common"

/** Labeled facts as a definition list; renders nothing without facts. */
export function ModelFacts({ facts, label, className }: { readonly facts: readonly ModelFact[]; readonly label: string; readonly className: string }) {
  if (facts.length === 0) return null
  return <dl aria-label={label} className={className}>
    {facts.map(([term, value]) => <div key={term} className="min-w-0">
      <dt className="text-xs text-slate-500">{term}</dt>
      <dd className="mt-0.5 text-[13px] tabular-nums text-slate-700 dark:text-slate-300" data-model-fact={term}>{value}</dd>
    </div>)}
  </dl>
}
