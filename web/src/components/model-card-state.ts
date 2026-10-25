import { Option } from "effect"
import {
  catalogDrafterState, catalogVariantOfName, customModelDrafterOffer, describeModelLoadStage, formatModelParameters, formatModelReleaseAge,
  localModelSpeculativeMethodLabel,
} from "@magnitudedev/client-common"
import type { LocalInferenceHardware, LocalModel, ModelTransferProgress } from "@magnitudedev/sdk"
import { modelLab } from "./model-logo"
import { drafterUpdateAvailable } from "./custom-model-details"

export type StatusTone = "ready" | "busy" | "absent" | "stored" | "problem"

/** Time left from the live transfer rate; preparation and verification have no download ETA. */
export const modelTransferEstimate = (progress: ModelTransferProgress): string | null => {
  if (progress.stage !== "downloading") return null
  const rate = Option.getOrNull(progress.bytesPerSecond)
  if (rate === null || rate <= 0 || progress.totalBytes <= 0) return "Estimating…"
  const seconds = (progress.totalBytes - progress.completedBytes) / rate
  return seconds < 60 ? "est. <1 min" : `est. ${Math.ceil(seconds / 60)} min`
}

/** What a card's Drafter cell shows and offers. */
export type ModelCardDrafter =
  | { readonly _tag: "Add" }
  | { readonly _tag: "Adding"; readonly percent: number | null }
  | { readonly _tag: "Attached"; readonly label: string; readonly title: string; readonly changeable: boolean; readonly update: boolean }
  | { readonly _tag: "None"; readonly changeable: boolean }
  | { readonly _tag: "Withheld"; readonly note: string }
  | { readonly _tag: "Pending" }
  | { readonly _tag: "Unknown" }

/**
 * A catalog model offers its catalog drafter or shows the one it serves with; a custom model shows its attached
 * drafter (changeable, with Update when a newer revision exists), that its automatic drafter is still being found,
 * the recommended one to add, or None to choose one.
 */
export const modelCardDrafter = (model: LocalModel): ModelCardDrafter => {
  const method = localModelSpeculativeMethodLabel(model)
  if (model._tag === "Catalog") {
    const state = catalogDrafterState(model)
    const drafterUpdate = catalogDrafterOnlyUpdate(model)
    switch (state._tag) {
      case "Offer": return { _tag: "Add" }
      case "Withheld": return { _tag: "Withheld", note: state.note }
      case "Attached": return { _tag: "Attached", label: Option.getOrElse(method, () => state.name), title: state.name, changeable: false, update: drafterUpdate }
      case "None": return drafterUpdate ? { _tag: "Add" } : { _tag: "None", changeable: false }
      case "Unknown": return { _tag: "Unknown" }
    }
  }
  const attached = Option.getOrUndefined(model.drafter)
  if (attached?.availability._tag === "Downloading") {
    const { progress } = attached.availability
    return { _tag: "Adding", percent: progress.totalBytes > 0 ? Math.floor(progress.completedBytes * 100 / progress.totalBytes) : null }
  }
  if (attached !== undefined) return { _tag: "Attached", label: Option.getOrElse(method, () => attached.name), title: attached.name, changeable: true, update: drafterUpdateAvailable(model) }
  // Drafter selection waits for the target files; that is not an active search.
  if (model.drafterPending) return { _tag: addedModelAwaitingFiles(model) ? "Unknown" : "Pending" }
  return Option.match(customModelDrafterOffer(model), {
    onNone: (): ModelCardDrafter => ({ _tag: "None", changeable: true }),
    onSome: (): ModelCardDrafter => ({ _tag: "Add" }),
  })
}

/** A catalog update that brings only a drafter: offered in the Drafter cell, never as the model's Update. */
export const catalogDrafterOnlyUpdate = (model: LocalModel): boolean =>
  model._tag === "Catalog" && model.acquisitionState._tag === "UpdateAvailable" && model.acquisitionState.parts.every(part => part === "Drafter")

/**
 * An added model whose files have not arrived: not downloaded yet, downloading, or after a cancelled or failed
 * download. Magnitude assesses a custom model only from its files, so it has no assessment until they land; the
 * repository step already checked that the version runs here. Its card claims no assessment in progress and keeps
 * Download and Remove, so a cancelled download can be started again or the model forgotten.
 */
export const addedModelAwaitingFiles = (model: LocalModel): boolean =>
  model._tag === "Custom" && ["NotInstalled", "Installing", "InstallFailed"].includes(model.acquisitionState._tag)

/** Whether Download (or Retry download) can fetch the model: one assessed to fit, or an added model awaiting its files. */
export const modelCanDownload = (model: LocalModel): boolean =>
  addedModelAwaitingFiles(model)
  || (model._tag === "Custom" || model.catalogData.support._tag === "Supported")
    && model.servingState._tag === "Assessed" && model.servingState.assessment._tag === "Fits"

/** A card's Status cell: where the model is in its lifecycle, from not downloaded to running. */
export const modelCardStatus = (model: LocalModel, hardware: Option.Option<LocalInferenceHardware>): { readonly label: string; readonly tone: StatusTone } => {
  const acquisition = model.acquisitionState
  switch (acquisition._tag) {
    case "Installing": return { label: "Downloading", tone: "busy" }
    case "Updating": return { label: "Updating", tone: "busy" }
    case "Optimizing": return { label: "Optimizing", tone: "busy" }
    case "Removing": return { label: "Removing…", tone: "busy" }
    case "RemoveFailed": return { label: "Removal failed", tone: "problem" }
    case "Unavailable": return { label: "Files unavailable", tone: "problem" }
    case "InstallFailed": return { label: "Not downloaded", tone: "problem" }
    case "UpdateFailed": return { label: "Update incomplete", tone: "problem" }
  }
  if (!("residencyState" in acquisition)) return { label: "Not downloaded", tone: "absent" }
  const residency = acquisition.residencyState
  switch (residency._tag) {
    case "Ready": return { label: "Running", tone: "ready" }
    case "Requested": return { label: "Preparing…", tone: "busy" }
    case "Loading": return { label: describeModelLoadStage(residency.stage, residency.plannedAllocation, hardware), tone: "busy" }
    case "Stopping": return { label: "Stopping…", tone: "busy" }
    case "Failed": return { label: "Not loaded", tone: "problem" }
    default: return { label: acquisition._tag === "External" ? "On this computer" : "Downloaded", tone: "stored" }
  }
}

/** Who made the model: an added repository's owner, otherwise the lab its name or catalog lineage points to. */
const modelProducer = (model: LocalModel): Option.Option<string> =>
  model._tag === "Custom" && model.origin._tag === "Added"
    ? Option.some(model.origin.repository.split("/")[0]!)
    : Option.fromNullable(modelLab(model)?.name)

/**
 * The line under a card's name, never repeating the strip: on Catalog the maker and how long ago the model was
 * released; on My Models the maker and the model's size, or what an added model is a variant of.
 */
export const modelSubtitle = (model: LocalModel, models: readonly LocalModel[], page: "catalog" | "models"): string => {
  const detail = model._tag === "Catalog"
    ? Option.some(page === "catalog" ? formatModelReleaseAge(model.catalogData.releaseDate) : formatModelParameters(model.catalogData.parameterization))
    : Option.map(catalogVariantOfName(models, model.presentation.variantOf), name => `Variant of ${name}`)
  return [modelProducer(model), detail].flatMap(part => Option.isSome(part) ? [part.value] : []).join(" · ")
}

const updatePartNames = { Model: "model", ImageInput: "image input", Drafter: "drafter" } as const

/** What a model's Update brings, for its tooltip: "A newer version of the drafter is available." */
export const modelUpdateDescription = (model: LocalModel): Option.Option<string> => {
  const acquisition = model.acquisitionState
  if (acquisition._tag !== "UpdateAvailable") return Option.none()
  if (model._tag === "Custom") return Option.some("The Hugging Face repository has a newer version of this model.")
  const names = acquisition.parts.map(part => updatePartNames[part])
  return Option.some(names.length === 1
    ? `A newer version of the ${names[0]} is available.`
    : `Newer versions of the ${names.slice(0, -1).join(", ")} and ${names.at(-1)} are available.`)
}
