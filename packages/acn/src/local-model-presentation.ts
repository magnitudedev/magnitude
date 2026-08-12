import { Option } from "effect"
import {
  ModelVariantLabelSchema,
  type LocalModelPresentation,
  type ModelVariantLabel,
  type ServableModelBundle,
} from "@magnitudedev/acn-protocol"

interface CuratedModelPresentation {
  readonly displayName: string
  readonly variantLabel: ModelVariantLabel
  readonly description: string
  readonly license?: string
}

export const bundlePackages = (bundle: ServableModelBundle) =>
  bundle._tag === "Standalone" ? [bundle.package] : [bundle.target, bundle.draft]

const sourceName = (bundle: ServableModelBundle): string => {
  const primary = bundle._tag === "Standalone" ? bundle.package : bundle.target
  return primary.source._tag === "HuggingFace"
    ? primary.source.repository.split("/").at(-1) ?? primary.source.repository
    : primary.files[0]?.path.split("/").at(-1) ?? primary.id
}

export const resolveBundlePresentation = (
  bundle: ServableModelBundle,
  curated: CuratedModelPresentation | undefined,
): LocalModelPresentation => {
  const packages = bundlePackages(bundle)
  const quantization = packages.map(({ properties }) => properties.quantization).join(" + ")
  return {
    displayName: curated?.displayName ?? sourceName(bundle),
    variantLabel: curated?.variantLabel ?? ModelVariantLabelSchema.make(quantization),
    description: curated?.description ?? "",
    license: Option.fromNullable(curated?.license),
    quantization,
    precisionLabel: packages.map(({ properties }) => properties.quantizationName).join(" + "),
  }
}
