import { LightningIcon } from "@phosphor-icons/react"
import { catalogComponentDownloadLabel, catalogComponentName, catalogComponentOffers, formatStorageSize } from "@magnitudedev/client-common"
import type { CatalogComponent, CatalogLocalModel } from "@magnitudedev/sdk"
import { Button } from "./ui/button"

/** The card line offering the catalog drafter a found exact target lacks; one click downloads only the drafter. */
export function CatalogDrafterOffer({ model, disabled, onDownload }: { model: CatalogLocalModel; disabled: boolean; onDownload: (component: CatalogComponent) => void }) {
  const offer = catalogComponentOffers(model).find(({ component }) => component === "Drafter")
  if (offer === undefined) return null
  return <Button variant="link" size="unstyled" disabled={disabled} onClick={() => onDownload("Drafter")} className="inline-flex items-center gap-1 text-sm text-blue-700 dark:text-blue-400">
    <LightningIcon aria-hidden="true" className="size-3.5" />{offer.downloadBytes > 0 ? `Faster with a drafter · Add (${formatStorageSize(offer.downloadBytes)})` : "Faster with a drafter · Add"}
  </Button>
}

/** In Details, each catalog component a found exact target lacks ("Image input · Download (1.0 GB)"), fetched alone. */
export function CatalogComponentDetails({ model, disabled, onDownload }: { model: CatalogLocalModel; disabled: boolean; onDownload: (component: CatalogComponent) => void }) {
  const offers = catalogComponentOffers(model)
  if (offers.length === 0) return null
  return <dl className="flex flex-wrap gap-x-8 gap-y-3">
    {offers.map(offer => <div key={offer.component}><dt className="text-xs text-slate-500">{catalogComponentName(offer.component)}</dt><dd className="mt-1 flex items-center gap-2">
      <span className="text-slate-500">Not downloaded</span>
      <Button variant="outline" size="unstyled" className="h-auto px-1.5 py-0.5 text-xs" disabled={disabled} onClick={() => onDownload(offer.component)}>{catalogComponentDownloadLabel(offer)}</Button>
    </dd></div>)}
  </dl>
}
