import { Option } from "effect"
import {
  formatModelDisplayName,
  localModelServingState,
  type CatalogDeprecation,
  type CatalogLocalModel,
  type CatalogSupport,
  type LocalInferenceBackend,
  type LocalModel,
  type ModelVariantLabel,
  type SpeculativeMethod,
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
