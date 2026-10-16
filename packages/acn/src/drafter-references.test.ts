import { Effect, Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { DrafterCandidateSchema, DrafterReferenceSchema, type DrafterCandidate } from "@magnitudedev/acn-protocol"
import {
  DrafterMethods,
  DrafterMethodsLive,
  drafterReferenceKey,
  orderDrafterCandidates,
  productDrafterCandidate,
} from "./drafter-references"

const reference = (path: string) => Schema.decodeUnknownSync(DrafterReferenceSchema)({
  _tag: "HuggingFace", repository: "z-lab/drafters", commit: "abc", path,
})

const candidate = (path: string, overrides: Partial<Schema.Schema.Encoded<typeof DrafterCandidateSchema>> = {}, rate?: number): DrafterCandidate => ({
  ...Schema.decodeUnknownSync(DrafterCandidateSchema)({
    category: "HuggingFace",
    reference: reference(path),
    label: path,
    sizeBytes: 100,
    lineage: "Lineage",
    verdict: { _tag: "Bound", method: { _tag: "DFlash" }, memoryBytes: 1 },
    onThisComputer: false,
    ...overrides,
  }),
  measuredAcceptance: rate === undefined
    ? Option.none()
    : Option.some({ acceptanceRate: rate, tokensPerSecond: 10, withoutDrafterTokensPerSecond: Option.none(), slower: false }),
})

describe("drafter candidate order", () => {
  it("keeps category order and breaks ties by lineage, measured acceptance, then size", () => {
    const ordered = orderDrafterCandidates([
      candidate("fingerprint.gguf", { lineage: "Fingerprint", sizeBytes: 1 }),
      candidate("unmeasured.gguf", { sizeBytes: 1 }),
      candidate("measured-low.gguf", {}, 0.3),
      candidate("measured-high.gguf", {}, 0.7),
      candidate("catalog.gguf", { category: "CatalogDrafter", lineage: "None", sizeBytes: 999 }),
      candidate("big.gguf", { sizeBytes: 5 }),
    ])
    expect(ordered.map(({ label }) => label)).toEqual([
      "catalog.gguf",
      "measured-high.gguf",
      "measured-low.gguf",
      "unmeasured.gguf",
      "big.gguf",
      "fingerprint.gguf",
    ])
  })

  it("is deterministic for equal candidates", () => {
    const input = [candidate("a.gguf"), candidate("b.gguf")]
    expect(orderDrafterCandidates(input).map(({ label }) => label)).toEqual(["a.gguf", "b.gguf"])
  })
})

describe("drafter identity", () => {
  it("keys each kind of reference stably", () => {
    expect(drafterReferenceKey(reference("d.gguf"))).toBe("hf:z-lab/drafters@abc/d.gguf")
    expect(drafterReferenceKey({ _tag: "Embedded" })).toBe("embedded")
    expect(drafterReferenceKey({ _tag: "Local", path: "/d.gguf" })).toBe("local:/d.gguf")
  })

  it("renders a rejected verdict without reasons as an unknown reason", () => {
    const product = productDrafterCandidate({
      category: "Other",
      reference: { _tag: "Local", path: "/d.gguf" },
      label: "d",
      lineage: "None",
      method: Option.none(),
      onThisComputer: true,
      sizeBytes: 1,
      verdict: { _tag: "Rejected", reasons: [], detail: "mismatch" },
    } as never)
    expect(Option.map(product, ({ verdict }) => verdict)).toEqual(Option.some({
      _tag: "Rejected",
      reasons: [{ _tag: "Unknown", kind: "Unknown", detail: "mismatch" }],
      detail: "mismatch",
    }))
  })

  it("remembers binding methods and knows MTP heads without asking", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const methods = yield* DrafterMethods
      const before = yield* methods.get(reference("d.gguf"))
      yield* methods.remember(reference("d.gguf"), { _tag: "DSpark" })
      const lookup = yield* methods.snapshot
      return { before, after: lookup(reference("d.gguf")), embedded: lookup({ _tag: "Embedded" }) }
    }).pipe(Effect.provide(DrafterMethodsLive)))
    expect(result).toEqual({
      before: Option.none(),
      after: Option.some({ _tag: "DSpark" }),
      embedded: Option.some({ _tag: "Mtp" }),
    })
  })
})
