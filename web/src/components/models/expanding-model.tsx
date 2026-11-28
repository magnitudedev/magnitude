import { Fragment, type ReactNode } from "react"
import { CircleNotchIcon } from "@phosphor-icons/react"
import type { ModelLab } from "../model-logo"
import { ContextText, DownloadsCount, ModelCardShell, PlacementBar, QuantLadder, QuantList, RepositoryLink, quantSource } from "./parts"
import type { QuantView } from "./view"

/** A model offered for download: its quant ladder, the rung chosen for this machine, and what it pairs with. */
export interface OfferedModel {
  readonly key: string
  readonly commit?: string
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
  /** Hugging Face repository metadata for non-catalog search rows. */
  readonly downloads?: number | null
  readonly uploader?: string | null
  readonly quantCount?: number | null
}

/** Joins non-null detail items with middle dots so no leading or trailing separator ever appears. */
function DetailItems({ items }: { items: readonly (ReactNode | null | false | undefined)[] }) {
  const present = items.filter((item): item is ReactNode => item !== null && item !== false && item !== undefined)
  return <>{present.map((item, index) => <Fragment key={index}>{index > 0 && <span aria-hidden="true">·</span>}{item}</Fragment>)}</>
}

/** A model that opens in place to its quant ladder and how the chosen quant runs here. */
export function ExpandingModel({ model, note, expanded, onToggle, quantChoice, onQuantChoice, owned, drafter, footer }: {
  model: OfferedModel; note?: string | undefined; expanded: boolean; onToggle: () => void
  quantChoice: string | null; onQuantChoice: (key: string) => void; owned: ReadonlySet<string>
  drafter?: ((quant: QuantView) => ReactNode) | undefined; footer?: (quant: QuantView) => ReactNode
}) {
  const current = model.quants.find(quant => quant.key === quantChoice) ?? model.quants.find(quant => quant.key === model.recommended) ?? model.quants[0]
  if (!current) return null
  const totalQuants = model.quantCount ?? model.quants.length
  const multiQuantRepo = !model.catalog && totalQuants > 1
  const isRecommended = current.key === model.recommended
  const sizeLabel = model.catalog
    ? "on disk"
    : multiQuantRepo && isRecommended
      ? `Best: ${current.name}`
      : current.name
  const detail = model.catalog
    ? <DetailItems items={[
        current.bits > 0 && <span className="font-medium text-slate-700 dark:text-slate-300">{current.bits}-bit</span>,
        current.repository ? <RepositoryLink repository={current.repository}>{quantSource(current)}</RepositoryLink> : quantSource(current),
        model.fineTuneOf && `Fine-tune of ${model.fineTuneOf}`,
        model.drafter && !current.drafterWithheld && model.drafter.method,
        current.servedContext !== null && <ContextText quant={current} limit={model.contextLimit} modelName={model.name} />,
        model.country === "US" && "🇺🇸",
      ]} />
    : <DetailItems items={[
        model.downloads != null && model.downloads > 0 && <DownloadsCount count={model.downloads} />,
        current.repository
          ? <RepositoryLink repository={current.repository}>{model.uploader ?? current.publisher ?? current.repository}</RepositoryLink>
          : (model.uploader ?? current.publisher ?? null),
        multiQuantRepo && <span className="whitespace-nowrap">{totalQuants} quants</span>,
        model.fineTuneOf && <span className="truncate">Fine-tune of {model.fineTuneOf}</span>,
        model.country === "US" && "🇺🇸",
      ]} />
  return <ModelCardShell lab={model.lab} title={model.name} note={note} quant={current} sizeLabel={sizeLabel}
    disclosure={model.catalog ? "check" : "chevron"} active={expanded} onSelect={onToggle} detail={detail}>
    {expanded && <div className="border-t border-slate-200 bg-slate-50/60 px-4 py-4 dark:border-slate-750 dark:bg-slate-900/40">
      {model.catalog
        ? <QuantLadder quants={model.quants} selected={current.key} recommended={model.recommended} owned={owned} limit={model.contextLimit} onSelect={onQuantChoice} />
        : <QuantList quants={model.quants} selected={current.key} recommended={model.recommended} owned={owned} onSelect={onQuantChoice} />}
      {model.checking && <p role="status" className="mt-2 flex items-center gap-1.5 text-xs text-slate-500"><CircleNotchIcon aria-hidden="true" className="size-3.5 motion-safe:animate-spin" />Checking this repository's other quants…</p>}
      {current.fits && <div className="mt-4"><PlacementBar quant={current} /></div>}
      {/* Each slot under its own key: a drafter and a download both keyed by their quant would otherwise collide. */}
      {(drafter || footer) && <div className="mt-4 flex flex-wrap items-center justify-between gap-x-4 gap-y-3">
        <Fragment key="drafter">{drafter?.(current) ?? <span />}</Fragment>
        <Fragment key="footer">{footer?.(current)}</Fragment>
      </div>}
    </div>}
  </ModelCardShell>
}
