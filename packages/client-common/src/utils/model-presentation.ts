import { Option } from "effect"
import {
  formatModelDisplayName,
  type LocalModel,
  type SpeculativeMethod,
  type ModelVariantLabel,
} from "@magnitudedev/sdk"

export { formatModelDisplayName }

export const formatLocalModelDisplayName = (
  model: { readonly presentation: { readonly displayName: string; readonly variantLabel: ModelVariantLabel } },
): string => formatModelDisplayName(
  model.presentation.displayName,
  Option.some(model.presentation.variantLabel),
)

export const formatSpeculativeMethod = (method: SpeculativeMethod): string =>
  method._tag === "Mtp" ? "MTP" : method._tag

export const localModelSpeculativeMethodLabel = (model: LocalModel): Option.Option<string> =>
  model.bundle._tag === "Standalone"
    ? Option.none()
    : Option.some(formatSpeculativeMethod(model.bundle.method))
