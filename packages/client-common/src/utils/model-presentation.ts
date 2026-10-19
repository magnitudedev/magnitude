import { Option } from "effect"
import {
  customModelSourceLabel,
  formatModelDisplayName,
  installedAcquisition,
  localModelServingState,
  type CatalogFormModelId,
  type CatalogDeprecation,
  type CatalogLocalModel,
  type CatalogSupport,
  type LocalInferenceBackend,
  type LocalModel,
  type ModelParameterization,
  type ModelReleaseDate,
  type ModelVariantLabel,
  type SpeculativeMethod,
  type UnsupportedLocalModel,
} from "@magnitudedev/sdk"

export { formatModelDisplayName }

export const formatLocalInferenceBackend = (backend: LocalInferenceBackend): string => {
  switch (backend) {
    case "cpu": return "CPU"
    case "metal": return "Metal"
    case "cuda": return "CUDA"
    case "vulkan": return "Vulkan"
  }
}

export const formatLocalModelDisplayName = (
  model: { readonly presentation: { readonly displayName: string; readonly variantLabel: ModelVariantLabel } },
): string => formatModelDisplayName(
  model.presentation.displayName,
  Option.some(model.presentation.variantLabel),
)

/** The label a catalog model carries when its release does not promise full support. */
export const catalogSupportLabel = (support: CatalogSupport): Option.Option<string> => {
  switch (support._tag) {
    case "Supported": return Option.none()
    case "Disabled": return Option.some("Disabled")
    case "Deprecated": return Option.some("Deprecated")
  }
}

/** The catalog configuration a deprecated model names as its replacement, when it is listed. */
export const catalogModelReplacement = (
  models: readonly LocalModel[],
  deprecation: CatalogDeprecation,
): Option.Option<CatalogLocalModel> => Option.fromNullable(models.find((candidate): candidate is CatalogLocalModel =>
  candidate._tag === "Catalog" && candidate.modelId === deprecation.replacement))

/** One sentence reporting a deprecated installed model and what to use instead. */
export const describeCatalogDeprecation = (
  deprecation: CatalogDeprecation,
  replacement: Option.Option<CatalogLocalModel>,
): string => Option.match(replacement, {
  onNone: () => `No longer supported: ${deprecation.reason} Switch to ${deprecation.replacement}.`,
  onSome: (model) => `No longer supported: ${deprecation.reason} Switch to ${formatLocalModelDisplayName(model)}.`,
})

export const formatSpeculativeMethod = (method: SpeculativeMethod): string =>
  method._tag === "Mtp" ? "MTP" : method._tag

export const localModelSpeculativeMethodLabel = (model: LocalModel): Option.Option<string> =>
  Option.flatMap(localModelServingState(model), (serving) =>
    serving._tag === "Assessed"
      ? Option.map(serving.speculativeMethod, formatSpeculativeMethod)
      : Option.none())

/** The tag showing where a model's files came from: a custom model's source, or "Magnitude" for a catalog model Magnitude downloaded. */
export const localModelSourceTag = (model: LocalModel): Option.Option<string> =>
  model._tag === "Custom"
    ? Option.some(model.presentation.sourceLabel)
    : installedAcquisition(model.acquisitionState)?.installation.ownership === "Magnitude" ? Option.some("Magnitude") : Option.none()

export const unsupportedLocalModelSourceTag = (model: UnsupportedLocalModel): string =>
  customModelSourceLabel(model.origin)

const catalogBase = (modelId: string): string => modelId.slice(0, modelId.indexOf(":"))

/** The catalog model's name for "Variant of <model>", matched by base model so any listed variant names it. */
export const catalogVariantOfName = (
  models: readonly LocalModel[],
  variantOf: Option.Option<CatalogFormModelId>,
): Option.Option<string> => Option.flatMap(variantOf, (modelId) =>
  Option.fromNullable(models.find((model) =>
    model._tag === "Catalog" && catalogBase(model.modelId) === catalogBase(modelId))?.presentation.displayName))

const MILLISECONDS_PER_DAY = 24 * 60 * 60 * 1_000

/** How long ago a catalog model was released, in whole local calendar days ("93 days ago"). */
export const formatModelReleaseAge = (releaseDate: ModelReleaseDate, now = new Date()): string => {
  const [year, month, day] = releaseDate.split("-").map(Number) as [number, number, number]
  const elapsed = Math.max(0, Math.round((Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()) - Date.UTC(year, month - 1, day)) / MILLISECONDS_PER_DAY))
  return elapsed === 0 ? "Today" : elapsed === 1 ? "1 day ago" : `${elapsed} days ago`
}

const formatParameterCount = (parameters: number): string => {
  const billions = parameters / 1_000_000_000
  return `${billions >= 10 ? Math.round(billions) : Math.round(billions * 10) / 10}B`
}

/** A model's size in parameters: "27B" for a dense model, "26B total · 4B active" for a mixture of experts. */
export const formatModelParameters = (parameterization: ModelParameterization): string =>
  parameterization.architecture === "dense"
    ? formatParameterCount(parameterization.totalParameters)
    : `${formatParameterCount(parameterization.totalParameters)} total · ${formatParameterCount(parameterization.activeParameters)} active`
