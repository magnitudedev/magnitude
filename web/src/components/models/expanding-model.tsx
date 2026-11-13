import type { ReactNode } from "react"
import { CircleNotchIcon } from "@phosphor-icons/react"
import type { ModelLab } from "../model-logo"
import { ContextText, ModelCardShell, PlacementBar, QuantLadder, QuantList, RepositoryLink, quantSource } from "./parts"
import type { QuantView } from "./view"

/** A model offered for download: its quant ladder, the rung chosen for this machine, and what it pairs with. */
export interface OfferedModel {
  readonly key: string
  readonly catalog: boolean
  readonly name: string
  readonly lab: ModelLab | undefined
  readonly country: string | null
  readonly fineTuneOf: string | null
  readonly drafter: { readonly method: string; readonly sizeBytes: number } | null
  readonly contextLimit: number | null
  readonly quants: readonly QuantView[]
  readonly recommended: string | null
  readonly checking: boolean
}

/** A model that opens in place to its quant ladder and how the chosen quant runs here. */
export function ExpandingModel({ model, note, expanded, onToggle, quantChoice, onQuantChoice, owned, drafter, footer }: {
  model: OfferedModel; note?: string | undefined; expanded: boolean; onToggle: () => void
  quantChoice: string | null; onQuantChoice: (key: string) => void; owned: ReadonlySet<string>
  drafter?: ((quant: QuantView) => ReactNode) | undefined; footer?: (quant: QuantView) => ReactNode
}) {
  const current = model.quants.find(quant => quant.key === quantChoice) ?? model.quants.find(quant => quant.key === model.recommended) ?? model.quants[0]
  if (!current) return null
  return <ModelCardShell lab={model.lab} title={model.name} note={note} quant={current} active={expanded} onSelect={onToggle}
    detail={<>
      <span className="font-medium text-slate-700 dark:text-slate-300">{current.bits}-bit</span>
      {" · "}{current.repository ? <RepositoryLink repository={current.repository}>{quantSource(current)}</RepositoryLink> : quantSource(current)}
      {model.fineTuneOf ? ` · Fine-tune of ${model.fineTuneOf}` : ""}
      {model.drafter ? ` · ${model.drafter.method}` : ""}
      {current.servedContext !== null && <>{" · "}<ContextText quant={current} limit={model.contextLimit} modelName={model.name} /></>}
      {model.country === "US" ? " · 🇺🇸" : ""}
    </>}>
    {expanded && <div className="border-t border-slate-200 bg-slate-50/60 px-4 py-4 dark:border-slate-750 dark:bg-slate-900/40">
      {model.catalog
        ? <QuantLadder quants={model.quants} selected={current.key} recommended={model.recommended} owned={owned} limit={model.contextLimit} onSelect={onQuantChoice} />
        : <QuantList quants={model.quants} selected={current.key} recommended={model.recommended} owned={owned} onSelect={onQuantChoice} />}
      {model.checking && <p role="status" className="mt-2 flex items-center gap-1.5 text-xs text-slate-500"><CircleNotchIcon aria-hidden="true" className="size-3.5 motion-safe:animate-spin" />Checking this repository's other quants…</p>}
      {current.fits && <div className="mt-4"><PlacementBar quant={current} /></div>}
      {(drafter || footer) && <div className="mt-4 flex flex-wrap items-center justify-between gap-x-4 gap-y-3">
        {drafter?.(current) ?? <span />}
        {footer?.(current)}
      </div>}
    </div>}
  </ModelCardShell>
}
