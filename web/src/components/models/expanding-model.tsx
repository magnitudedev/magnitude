import { Fragment, type ReactNode } from "react"
import { CircleNotchIcon, StarIcon } from "@phosphor-icons/react"
import type { ModelLab } from "../model-logo"
import { ContextText, DownloadsCount, ModelCardShell, PlacementBar, QuantLadder, QuantList, RepositoryLink, bitWidthLabel, quantSource } from "./parts"
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
export function ExpandingModel({ model, note, featured = false, expanded, onToggle, quantChoice, onQuantChoice, owned, drafter, footer }: {
  model: OfferedModel; note?: string | undefined; featured?: boolean; expanded: boolean; onToggle: () => void
  quantChoice: string | null; onQuantChoice: (key: string) => void; owned: ReadonlySet<string>
  drafter?: ((quant: QuantView) => ReactNode) | undefined; footer?: (quant: QuantView) => ReactNode
}) {
  const current = model.quants.find(quant => quant.key === quantChoice) ?? model.quants.find(quant => quant.key === model.recommended) ?? model.quants[0]
  if (!current) return null
  const totalQuants = model.quantCount ?? model.quants.length
  const multiQuantRepo = !model.catalog && totalQuants > 1
  const isRecommended = current.key === model.recommended
  const currentBitsOrName = bitWidthLabel(current.bits) ?? current.name
  const sizeLabel = model.catalog
    ? featured
      ? model.quants.length > 1 && isRecommended
        ? `Best: ${currentBitsOrName}`
        : currentBitsOrName
      : "on disk"
    : multiQuantRepo && isRecommended
      ? `Best: ${current.name}`
      : current.name
  const detail = model.catalog
    ? featured
      ? <DetailItems items={[
          model.quants.length > 1 && <span className="whitespace-nowrap">{model.quants.length} quants</span>,
          model.drafter && !current.drafterWithheld && <span className="whitespace-nowrap">{model.drafter.method} drafter</span>,
          current.repository ? <RepositoryLink repository={current.repository}>{quantSource(current)}</RepositoryLink> : quantSource(current),
          model.country === "US" && "🇺🇸",
        ]} />
      : <DetailItems items={[
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
  const badge = featured
    ? <span className="inline-flex shrink-0 items-center gap-1 whitespace-nowrap rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 text-[11px] font-medium leading-4 text-blue-700 dark:border-blue-500/35 dark:bg-blue-900/35 dark:text-blue-300"><StarIcon weight="fill" aria-hidden="true" className="size-3 shrink-0" />Magnitude Pick</span>
    : undefined
  return <ModelCardShell lab={model.lab} title={model.name} badge={badge} note={note} quant={current} sizeLabel={sizeLabel}
    disclosure={model.catalog && !featured ? "check" : "chevron"} featured={featured} active={expanded} onSelect={onToggle} detail={detail}>
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
