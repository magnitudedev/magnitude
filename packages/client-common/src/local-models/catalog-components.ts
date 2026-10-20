import { Option } from "effect"
import { catalogDrafterName, type CatalogComponent, type CatalogLocalModel, type CatalogMissingComponent, type LocalModel } from "@magnitudedev/sdk"
import { formatStorageSize } from "../utils/format-bytes"

/** What a catalog component adds, as Details and the CLI name it. */
export const catalogComponentName = (component: CatalogComponent): string =>
  component === "Projector" ? "Image input" : "Drafter"

/**
 * The catalog components a model lacks that can be fetched now: those an installed catalog model's found
 * target lacks, none while any download, optimization or removal of it runs.
 */
export const catalogComponentOffers = (model: LocalModel): readonly CatalogMissingComponent[] =>
  model._tag === "Catalog" && model.acquisitionState._tag === "Installed" ? model.missingComponents : []

/** "Download (1.0 GB)", or "Download" when its bytes are already on this computer. */
export const catalogComponentDownloadLabel = ({ downloadBytes }: CatalogMissingComponent): string =>
  downloadBytes > 0 ? `Download (${formatStorageSize(downloadBytes)})` : "Download"

/**
 * Why an installed catalog model runs without its catalog drafter: it doesn't fit this computer with it. A withheld
 * model is assessed served plain, so its verdict tells whether it fits only without the drafter or not at all.
 */
export const catalogDrafterWithheldNote = (model: LocalModel): Option.Option<string> => {
  if (model._tag !== "Catalog" || !model.drafterWithheld) return Option.none()
  const serving = model.servingState
  if (serving._tag !== "Assessed") return Option.some("Off - this model doesn't fit this computer with it")
  return Option.some(serving.assessment._tag === "Fits"
    ? "Off - this model fits this computer only without it"
    : "Off - this model doesn't fit this computer")
}

/**
 * What a catalog model's Drafter row says: the download offer when its found target lacks the drafter, Off when
 * it doesn't fit with it, the drafter it serves with, None when it has none, or Unknown before it is assessed.
 */
export type CatalogDrafterState =
  | { readonly _tag: "Offer"; readonly offer: CatalogMissingComponent }
  | { readonly _tag: "Withheld"; readonly note: string }
  | { readonly _tag: "Attached"; readonly name: string }
  | { readonly _tag: "None" }
  | { readonly _tag: "Unknown" }

export const catalogDrafterState = (model: CatalogLocalModel): CatalogDrafterState => {
  const offer = catalogComponentOffers(model).find(({ component }) => component === "Drafter")
  if (offer !== undefined) return { _tag: "Offer", offer }
  const withheld = catalogDrafterWithheldNote(model)
  if (Option.isSome(withheld)) return { _tag: "Withheld", note: withheld.value }
  const serving = model.servingState
  if (serving._tag !== "Assessed") return { _tag: "Unknown" }
  return Option.match(serving.speculativeMethod, {
    onNone: () => ({ _tag: "None" }),
    onSome: (method) => ({ _tag: "Attached", name: catalogDrafterName(model.presentation.displayName, Option.some(method)) }),
  })
}
