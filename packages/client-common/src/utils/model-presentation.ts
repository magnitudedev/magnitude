import { Option } from "effect"
import {
  formatModelDisplayName,
  type ModelVariantLabel,
} from "@magnitudedev/sdk"

export { formatModelDisplayName }

export const formatLocalModelDisplayName = (
  model: { readonly presentation: { readonly displayName: string; readonly variantLabel: ModelVariantLabel } },
): string => formatModelDisplayName(
  model.presentation.displayName,
  Option.some(model.presentation.variantLabel),
)
