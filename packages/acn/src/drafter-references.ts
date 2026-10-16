import { Context, Effect, Layer, Option, Ref, Schema } from "effect"
import {
  DrafterCandidateSchema,
  DrafterReferenceSchema,
  type DrafterCandidate,
  type DrafterLineage,
  type DrafterReference,
  type MeasuredDrafterAcceptance,
  type SpeculativeMethod,
} from "@magnitudedev/acn-protocol"
import type * as Native from "@magnitudedev/icn-protocol/schemas"

/** An encoded optional field: present with its value, or absent; never present and undefined. */
export const presentField = <K extends string, A>(key: K, value: Option.Option<A>): { readonly [P in K]?: A } =>
  Option.match(value, { onNone: () => ({}), onSome: (present) => ({ [key]: present }) as { readonly [P in K]?: A } })

/** A stable identity for one drafter, used to attribute serving usage and match candidates. */
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

/** ICN's candidate in product form; a candidate the protocol cannot represent is dropped. */
export const productDrafterCandidate = (
  candidate: Native.DrafterCandidate,
  measuredAcceptance: Option.Option<MeasuredDrafterAcceptance> = Option.none(),
): Option.Option<DrafterCandidate> => Schema.decodeUnknownOption(DrafterCandidateSchema)({
  category: candidate.category,
  reference: candidate.reference,
  label: candidate.label,
  sizeBytes: candidate.sizeBytes,
  lineage: candidate.lineage,
  onThisComputer: candidate.onThisComputer,
  ...presentField("method", candidate.method),
  verdict: candidate.verdict._tag === "Rejected" && candidate.verdict.reasons.length === 0
    ? { ...candidate.verdict, reasons: [{ _tag: "Unknown", kind: "Unknown", detail: candidate.verdict.detail }] }
    : candidate.verdict,
}).pipe(Option.map((decoded) => ({ ...decoded, measuredAcceptance })))

const categoryOrder: Record<DrafterCandidate["category"], number> = {
  CatalogDrafter: 0, BuiltIn: 1, NextToModel: 2, OnThisComputer: 3, HuggingFace: 4, Other: 5,
}

const lineageOrder: Record<DrafterLineage, number> = { Exact: 0, Lineage: 0, Fingerprint: 1, None: 2 }

/**
 * Candidates in category order; within a category a lineage match precedes a fingerprint-only
 * match, then the higher measured acceptance (a measured pair before an unmeasured one), then the
 * smaller download. Equal candidates keep ICN's order.
 */
export const orderDrafterCandidates = (candidates: readonly DrafterCandidate[]): readonly DrafterCandidate[] =>
  candidates
    .map((candidate, index) => ({ candidate, index }))
    .sort((left, right) => {
      const a = left.candidate
      const b = right.candidate
      const acceptance = (candidate: DrafterCandidate) => Option.match(candidate.measuredAcceptance, {
        onNone: () => -1,
        onSome: ({ acceptanceRate }) => acceptanceRate,
      })
      return categoryOrder[a.category] - categoryOrder[b.category]
        || lineageOrder[a.lineage] - lineageOrder[b.lineage]
        || acceptance(b) - acceptance(a)
        || a.sizeBytes - b.sizeBytes
        || left.index - right.index
    })
    .map(({ candidate }) => candidate)

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
