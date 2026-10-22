import { Option, Schema } from "effect"
import {
  catalogDrafterName,
  DrafterCandidateSchema,
  DrafterReferenceSchema,
  type DrafterCandidate,
  type DrafterReference,
  type SpeculativeMethod,
} from "@magnitudedev/acn-protocol"
import type * as Native from "@magnitudedev/icn-protocol/schemas"
import type { CatalogModelSource } from "./local-model-sources"
import { productUnsupportedReasons } from "./unsupported-reasons"

/** An encoded optional field: present with its value, or absent; never present and undefined. */
export const presentField = <K extends string, A>(key: K, value: Option.Option<A>): { readonly [P in K]?: A } =>
  Option.match(value, { onNone: () => ({}), onSome: (present) => ({ [key]: present }) as { readonly [P in K]?: A } })

/** A stable identity for one drafter, used to match candidates. */
export const drafterReferenceKey = (reference: DrafterReference): string => {
  switch (reference._tag) {
    case "Catalog": return `catalog:${reference.modelId}`
    case "Embedded": return "embedded"
    case "HuggingFace": return `hf:${reference.repository}@${reference.commit}/${reference.path}`
    case "Local": return `local:${reference.path}`
  }
}

export const nativeDrafterReference = (reference: DrafterReference): Native.DrafterReference =>
  reference as unknown as Native.DrafterReference

export const productDrafterReference = (reference: Native.DrafterReference): Option.Option<DrafterReference> =>
  Schema.decodeUnknownOption(DrafterReferenceSchema)(reference)

export interface DrafterNaming {
  readonly name: string
  readonly label: string
}

const fileName = (path: string): string => path.split(/[\\/]/).at(-1) ?? path

/**
 * The name and label of the drafter a catalog model ships, binding with `method`, such as
 * "Magnitude DFlash for Qwen3.5 4B" and "Trained for Qwen3.5 4B": the same form ICN gives a catalog
 * drafter candidate. The method is the drafter's own, never the catalog model's served one, which
 * has none when the model fits this computer only without its drafter.
 */
export const catalogDrafterNaming = (
  catalog: readonly CatalogModelSource[],
  modelId: string,
  method: Option.Option<SpeculativeMethod>,
): Option.Option<DrafterNaming> => Option.map(
  Option.fromNullable(catalog.find(({ id }) => id === modelId)),
  ({ source }) => ({
    name: catalogDrafterName(source.displayName, method),
    label: `Trained for ${source.displayName}`,
  }),
)

/**
 * How an attached drafter reads when no candidate describes it: a catalog drafter is named from the
 * catalog, and any other separate drafter, not being the recommendation, was chosen by the user.
 */
export const drafterReferenceNaming = (
  reference: DrafterReference,
  catalog: readonly CatalogModelSource[],
  method: Option.Option<SpeculativeMethod>,
): DrafterNaming => {
  switch (reference._tag) {
    case "Catalog": return Option.getOrElse(catalogDrafterNaming(catalog, reference.modelId, method), () => ({
      name: "Magnitude drafter",
      label: "Trained by Magnitude",
    }))
    case "Embedded": return { name: "MTP heads", label: "Built into this model" }
    case "HuggingFace": return { name: `${reference.repository}/${fileName(reference.path)}`, label: "Chosen by you" }
    case "Local": return { name: fileName(reference.path), label: "Chosen by you" }
  }
}

/**
 * ICN's candidate in product form; a candidate the protocol cannot represent is dropped. A catalog
 * drafter takes its name from the catalog, so it reads the same as the drafter once attached.
 */
export const productDrafterCandidate = (
  candidate: Native.DrafterCandidate,
  catalog: readonly CatalogModelSource[],
):Option.Option<DrafterCandidate> => Schema.decodeUnknownOption(DrafterCandidateSchema)({
  category: candidate.category,
  reference: candidate.reference,
  ...Option.getOrElse(
    candidate.reference._tag === "Catalog"
      ? catalogDrafterNaming(catalog, candidate.reference.modelId, candidate.method)
      : Option.none<DrafterNaming>(),
    (): DrafterNaming => ({ name: candidate.name, label: candidate.label }),
  ),
  sizeBytes: candidate.sizeBytes,
  lineage: candidate.lineage,
  automatic: candidate.automatic,
  onThisComputer: candidate.onThisComputer,
  ...presentField("method", candidate.method),
  verdict: candidate.verdict._tag === "Rejected"
    ? { ...candidate.verdict, reasons: productUnsupportedReasons(candidate.verdict.reasons, candidate.verdict.detail) }
    : candidate.verdict,
})
