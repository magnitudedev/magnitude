import { Option } from "effect"
import type { CatalogComponent, CatalogMissingComponent, LocalModel } from "@magnitudedev/sdk"
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

/** Why an installed catalog model runs without its catalog drafter: it fits this computer only without it. */
export const catalogDrafterWithheldNote = (model: LocalModel): Option.Option<string> =>
  model._tag === "Catalog" && model.drafterWithheld
    ? Option.some("Off - this model fits this computer only without it")
    : Option.none()
