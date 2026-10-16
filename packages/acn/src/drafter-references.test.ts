import { Effect, Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  CatalogFormModelIdSchema,
  DrafterReferenceSchema,
} from "@magnitudedev/acn-protocol"
import type { CatalogModelSource } from "./local-model-sources"
import {
  DrafterMethods,
  DrafterMethodsLive,
  drafterReferenceKey,
  drafterReferenceNaming,
  productDrafterCandidate,
} from "./drafter-references"

const reference = (path: string) => Schema.decodeUnknownSync(DrafterReferenceSchema)({
  _tag: "HuggingFace", repository: "z-lab/drafters", commit: "abc", path,
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
      name: "d.gguf",
      label: "d",
      lineage: "None",
      method: Option.none(),
      onThisComputer: true,
      sizeBytes: 1,
      verdict: { _tag: "Rejected", reasons: [], detail: "mismatch" },
    } as never, [])
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

describe("drafter naming", () => {
  const catalogId = CatalogFormModelIdSchema.make("qwen3.6-35b-a3b:gguf:q4")
  const catalog = [{
    id: catalogId,
    source: { displayName: "Qwen3.5 4B", desired: { speculativeMethod: Option.some({ _tag: "DFlash" }) } },
  }] as unknown as readonly CatalogModelSource[]

  it("names a catalog drafter from the catalog, as ICN does, before ICN describes it", () => {
    expect(drafterReferenceNaming({ _tag: "Catalog", modelId: catalogId }, catalog)).toEqual({
      name: "Magnitude DFlash for Qwen3.5 4B",
      label: "Trained for Qwen3.5 4B",
    })
  })

  it("names embedded heads, and Hugging Face and local files the user chose", () => {
    expect(drafterReferenceNaming({ _tag: "Embedded" }, [])).toEqual({ name: "MTP heads", label: "Built into this model" })
    expect(drafterReferenceNaming(reference("sub/dflash.gguf"), [])).toEqual({
      name: "z-lab/drafters/dflash.gguf",
      label: "Chosen by you",
    })
    expect(drafterReferenceNaming({ _tag: "Local", path: "/models/mtp.gguf" }, [])).toEqual({
      name: "mtp.gguf",
      label: "Chosen by you",
    })
  })

  it("gives a catalog candidate the catalog name whatever ICN called it", () => {
    const product = productDrafterCandidate({
      category: "CatalogDrafter",
      reference: { _tag: "Catalog", modelId: catalogId },
      name: "Something else",
      label: "Something",
      lineage: "Lineage",
      method: Option.none(),
      onThisComputer: false,
      sizeBytes: 1,
      verdict: { _tag: "Bound", method: { _tag: "DFlash" }, memoryBytes: 1 },
    } as never, catalog)
    expect(Option.map(product, ({ name, label }) => ({ name, label }))).toEqual(Option.some({
      name: "Magnitude DFlash for Qwen3.5 4B",
      label: "Trained for Qwen3.5 4B",
    }))
  })
})
