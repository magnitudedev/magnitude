import { Option } from "effect"
import {
  formatModelDisplayName,
  localModelServingState,
  type LocalModel,
  type ModelVariantLabel,
  type SpeculativeMethod,
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
  Option.flatMap(localModelServingState(model), (serving) =>
    serving._tag === "Assessed"
      ? Option.map(serving.speculativeMethod, formatSpeculativeMethod)
      : Option.none())
