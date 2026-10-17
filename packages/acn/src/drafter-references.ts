import { Context, Effect, Layer, Option, Ref, Schema } from "effect"
import {
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

/** A stable identity for one drafter, used to match candidates and remember binding methods. */
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

const methodName = (method: SpeculativeMethod): string => method._tag === "Mtp" ? "MTP" : method._tag

const fileName = (path: string): string => path.split(/[\\/]/).at(-1) ?? path

/**
 * The name and label of the drafter a catalog model ships, such as "Magnitude DFlash for Qwen3.5 4B"
 * and "Trained for Qwen3.5 4B": the same form ICN gives a catalog drafter candidate.
 */
export const catalogDrafterNaming = (
  catalog: readonly CatalogModelSource[],
  modelId: string,
): Option.Option<DrafterNaming> => Option.map(
  Option.fromNullable(catalog.find(({ id }) => id === modelId)),
  ({ source }) => ({
    name: Option.match(source.desired.speculativeMethod, {
      onNone: () => `Magnitude drafter for ${source.displayName}`,
      onSome: (method) => `Magnitude ${methodName(method)} for ${source.displayName}`,
    }),
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
): DrafterNaming => {
  switch (reference._tag) {
    case "Catalog": return Option.getOrElse(catalogDrafterNaming(catalog, reference.modelId), () => ({
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
      ? catalogDrafterNaming(catalog, candidate.reference.modelId)
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

/** The method each drafter binds with, learned from the engine; MTP heads are always MTP. */
export interface DrafterMethodsApi {
  readonly get: (reference: DrafterReference) => Effect.Effect<Option.Option<SpeculativeMethod>>
  readonly remember: (reference: DrafterReference, method: SpeculativeMethod) => Effect.Effect<void>
  readonly snapshot: Effect.Effect<(reference: DrafterReference) => Option.Option<SpeculativeMethod>>
}
export class DrafterMethods extends Context.Tag("DrafterMethods")<DrafterMethods, DrafterMethodsApi>() {}

export const DrafterMethodsLive = Layer.effect(DrafterMethods, Effect.gen(function* () {
  const known = yield* Ref.make(new Map<string, SpeculativeMethod>())
  const lookup = (methods: ReadonlyMap<string, SpeculativeMethod>) => (reference: DrafterReference) =>
    reference._tag === "Embedded"
      ? Option.some<SpeculativeMethod>({ _tag: "Mtp" })
      : Option.fromNullable(methods.get(drafterReferenceKey(reference)))
  return DrafterMethods.of({
    get: (reference) => Ref.get(known).pipe(Effect.map((methods) => lookup(methods)(reference))),
    snapshot: Ref.get(known).pipe(Effect.map(lookup)),
    remember: (reference, method) => Ref.update(known, (methods) => new Map(methods).set(drafterReferenceKey(reference), method)),
  })
}))
