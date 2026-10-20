import { installedAcquisition, type LocalModel } from "@magnitudedev/sdk"

/**
 * What Remove does follows who owns the files: delete Magnitude's own, forget an imported file, hide a found one (a
 * catalog model whose weights are found elsewhere included), or, for such a catalog model, delete only the parts
 * Magnitude stores for it and cancel any it is still fetching ("parts").
 */
export type RemovalKind = "delete" | "forget" | "hide" | "parts"

/** Whether Magnitude is fetching parts (a projector or drafter) of a catalog model whose weights are found elsewhere. */
export const acquiringCatalogParts = (model: LocalModel): boolean =>
  model._tag === "Catalog"
  && installedAcquisition(model.acquisitionState)?.installation.ownership === "External"
  && (model.acquisitionState._tag === "Updating" || model.acquisitionState._tag === "Optimizing")

export const removalKind = (model: LocalModel): RemovalKind => {
  if (model._tag === "Catalog") {
    if (installedAcquisition(model.acquisitionState)?.installation.ownership !== "External") return "delete"
    return model.storedComponents.length > 0 || acquiringCatalogParts(model) ? "parts" : "hide"
  }
  return model.origin._tag === "Added" ? "delete" : model.origin._tag === "Imported" ? "forget" : "hide"
}
