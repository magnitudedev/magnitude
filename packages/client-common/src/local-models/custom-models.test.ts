import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  CatalogFormModelIdSchema,
  ModelVersionNotRunnable,
  UnsupportedReasonSchema,
  type CandidateVersion,
  type DrafterCandidate,
  type LocalModelsState,
  type UnsupportedReason,
} from "@magnitudedev/sdk"
import { customModelId, makeFoundModel, makeSetupModel, providerModelId } from "../application/fixtures/model"
import { activeLocalModel } from "../application/session"
import { localModelsFromCatalog } from "../model-catalog/projection"
import { catalogVariantOfName, localModelSourceTag, unsupportedLocalModelSourceTag } from "../utils/model-presentation"
import {
  candidateDownloadBytes,
  customModelDrafterOffer,
  drafterCandidateLabel,
  drafterCandidateSelectable,
  drafterCandidateWarning,
} from "./drafter-presentation"
import { modelVersionNotRunnableMessage, unsupportedReasonMessage, unsupportedReasonMessages } from "./failure-messages"
import { noRunnableVersionReason } from "./candidate-versions"
import {
  installedLocalModels,
  localModelIsInstalled,
  localModelProviderModelId,
  localModelResidency,
  preselectedDrafterCandidate,
} from "./projection"
import { localModelCommandStatus } from "./service"

const reason = (value: unknown): UnsupportedReason => Schema.decodeUnknownSync(UnsupportedReasonSchema)(value)

describe("unsupported reasons in the UI", () => {
  it("explains a refused version with its reasons or the memory it lacks", () => {
    expect(modelVersionNotRunnableMessage(new ModelVersionNotRunnable({
      reference: "owner/model-GGUF:Q2_K",
      version: "Q2_K",
      verdict: { _tag: "Unsupported", reasons: [reason({ _tag: "WeightTypes", types: [{ ggmlType: "Q2_K", tensorCount: 2 }] })] },
    }))).toBe("Magnitude can't run Q2_K: Uses Q2_K weights, which Magnitude doesn't support yet.")
    expect(modelVersionNotRunnableMessage(new ModelVersionNotRunnable({
      reference: "owner/model-GGUF:Q8_0",
      version: "Q8_0",
      verdict: { _tag: "DoesNotFit", deficitBytes: 6 * 1024 ** 3 + 1 },
    }))).toBe("Q8_0 needs 6.1 GB more memory than this computer can provide.")
  })

  it("words every known case as the product specifies", () => {
    expect([
      { _tag: "Architecture", architecture: "deepseek4" },
      { _tag: "WeightTypes", types: [{ ggmlType: "IQ1_M", tensorCount: 120 }] },
      { _tag: "RopeScaling", kind: "yarn" },
      { _tag: "UnrecognizedTensor", name: "rope_freqs.weight" },
      { _tag: "UnrecognizedMetadata", key: "llama.rope.scaling.type" },
      { _tag: "Tokenizer", model: "t5" },
      { _tag: "ChatTemplate" },
      { _tag: "TiedOutputHead" },
      { _tag: "TargetMismatch", detail: "hidden size differs" },
      { _tag: "KernelDomain", backend: "metal", entry: "attention", statics: [["head_dim", 80]] },
      { _tag: "Backend", backend: "vulkan" },
      { _tag: "Unreadable" },
    ].map((value) => unsupportedReasonMessage(reason(value)))).toEqual([
      "No kernels for the deepseek4 architecture",
      "Uses IQ1_M weights, which Magnitude doesn't support yet",
      "Uses rope frequency scaling, which Magnitude doesn't support yet",
      "Has parts Magnitude doesn't recognize (rope_freqs.weight)",
      "Has parts Magnitude doesn't recognize (llama.rope.scaling.type)",
      "Uses a tokenizer Magnitude doesn't support",
      "Its chat template can't be read",
      "Uses a tied output layer, which Magnitude doesn't support yet",
      "Its drafter wasn't built for this model",
      "Its layer sizes aren't supported on this GPU",
      "Its layer sizes aren't supported on this GPU",
      "The file is incomplete or damaged",
    ])
  })

  it("names every unsupported weight type and renders an unknown case generically", () => {
    expect(unsupportedReasonMessage(reason({
      _tag: "WeightTypes",
      types: [{ ggmlType: "Q2_K", tensorCount: 2 }, { ggmlType: "IQ1_M", tensorCount: 40 }, { ggmlType: "TQ1_0", tensorCount: 1 }],
    }))).toBe("Uses Q2_K, IQ1_M, and TQ1_0 weights, which Magnitude doesn't support yet")
    expect(unsupportedReasonMessage(reason({ _tag: "SlidingWindow", detail: "window 4096" })))
      .toBe("Magnitude can't run this model")
  })

  it("collapses reasons that read the same", () => {
    expect(unsupportedReasonMessages([
      reason({ _tag: "KernelDomain", backend: "metal", entry: "a", statics: [] }),
      reason({ _tag: "Backend", backend: "metal" }),
    ])).toEqual(["Its layer sizes aren't supported on this GPU"])
  })

  it("says many unrecognized parts once, with how many there are", () => {
    const tensors = Array.from({ length: 60 }, (_, index) => reason({ _tag: "UnrecognizedTensor", name: `blk.${15 + Math.floor(index / 3)}.attn_k` }))
    expect(unsupportedReasonMessages([
      ...tensors,
      reason({ _tag: "UnrecognizedMetadata", key: "gemma4.a" }),
      reason({ _tag: "UnrecognizedMetadata", key: "gemma4.b" }),
      reason({ _tag: "Architecture", architecture: "deepseek4" }),
    ])).toEqual([
      "Has parts Magnitude doesn't recognize (60 tensors)",
      "Has parts Magnitude doesn't recognize (2 metadata keys)",
      "No kernels for the deepseek4 architecture",
    ])
    expect(unsupportedReasonMessages([reason({ _tag: "UnrecognizedTensor", name: "blk.15.attn_k" })]))
      .toEqual(["Has parts Magnitude doesn't recognize (blk.15.attn_k)"])
  })
})

const bound = { _tag: "Bound", method: { _tag: "DFlash" }, memoryBytes: 900_000_000 } as const
const candidate = (overrides: Partial<DrafterCandidate>): DrafterCandidate => ({
  category: "CatalogDrafter",
  reference: { _tag: "Catalog", modelId: providerModelId },
  name: "Magnitude DFlash for Setup Model",
  label: "Trained for Setup Model",
  sizeBytes: 900_000_000,
  lineage: "Lineage",
  verdict: bound,
  onThisComputer: false,
  method: Option.some({ _tag: "DFlash" }),
  ...overrides,
})

describe("drafter presentation", () => {
  it("shows ICN's label, marked Recommended only on the preselected choice", () => {
    expect(drafterCandidateLabel(candidate({}), true)).toBe("Recommended · Trained for Setup Model")
    expect(drafterCandidateLabel(candidate({}), false)).toBe("Trained for Setup Model")
    expect(drafterCandidateLabel(candidate({ category: "BuiltIn", reference: { _tag: "Embedded" }, label: "Built into this model" }), false))
      .toBe("Built into this model")
    expect(drafterCandidateLabel(candidate({ category: "NextToModel", label: "In this repository" }), true))
      .toBe("Recommended · In this repository")
    expect(drafterCandidateLabel(candidate({ category: "Other", lineage: "None", label: "Untested" }), false)).toBe("Untested")
  })

  it("totals a candidate's download: model, projector, and a drafter not already here", () => {
    const version = {
      label: "Q4_K_M",
      modelId: customModelId,
      files: ["model.gguf"],
      sizeBytes: 3_010_000_000,
      downloadBytes: 3_680_000_000,
      assessment: Option.none(),
      recommended: true,
      onThisComputer: Option.none(),
    } satisfies CandidateVersion
    const projector = Option.some({ path: "mmproj.gguf", sizeBytes: 670_000_000 })
    expect(candidateDownloadBytes(version, projector, Option.some(candidate({ sizeBytes: 685_000_000 })))).toEqual({
      totalBytes: 4_365_000_000,
      modelBytes: 3_010_000_000,
      projectorBytes: 670_000_000,
      drafterBytes: 685_000_000,
    })
    expect(candidateDownloadBytes({ ...version, downloadBytes: 0 }, projector, Option.some(candidate({ onThisComputer: true }))))
      .toEqual({ totalBytes: 0, modelBytes: 0, projectorBytes: 0, drafterBytes: 0 })
  })

  it("never offers a rejected drafter and explains why", () => {
    const rejected = candidate({
      category: "NextToModel",
      verdict: { _tag: "Rejected", reasons: [reason({ _tag: "TargetMismatch", detail: "" })], detail: "" },
    })
    expect(drafterCandidateSelectable(rejected)).toBe(false)
    expect(drafterCandidateWarning(rejected)).toEqual(Option.some("Its drafter wasn't built for this model"))
    expect(drafterCandidateWarning(candidate({ category: "Other", lineage: "None", label: "Untested" }))).toEqual(Option.none())
    expect(preselectedDrafterCandidate([rejected, candidate({ category: "BuiltIn" })]).pipe(Option.map(({ category }) => category)))
      .toEqual(Option.some("BuiltIn"))
  })

  it("preselects the first bound candidate in ICN's order, never one the user entered", () => {
    const repository = candidate({ category: "NextToModel", label: "In this repository" })
    expect(preselectedDrafterCandidate([repository, candidate({})])).toEqual(Option.some(repository))
    expect(preselectedDrafterCandidate([candidate({ category: "Other", lineage: "None", label: "Untested" })])).toEqual(Option.none())
  })

  it("offers the recommended drafter to a model without one", () => {
    const offer = customModelDrafterOffer(makeFoundModel({ recommendedDrafter: Option.some(candidate({})) }))
    expect(Option.map(offer, ({ label }) => label)).toEqual(Option.some("Faster with a drafter · Add (900 MB)"))
    expect(customModelDrafterOffer(makeFoundModel({
      recommendedDrafter: Option.some(candidate({ onThisComputer: true })),
    })).pipe(Option.map(({ label }) => label))).toEqual(Option.some("Faster with a drafter · Add"))
    expect(customModelDrafterOffer(makeFoundModel())).toEqual(Option.none())
  })
})

describe("custom models in client projections", () => {
  const unsupported = {
    displayName: "DeepSeek V4 Flash",
    origin: { _tag: "Found", source: { _tag: "Folder", path: "/Volumes/External/models/" } },
    path: "/Volumes/External/models/deepseek.gguf",
    reasons: [reason({ _tag: "Architecture", architecture: "deepseek4" })],
    detail: "",
  } as const

  it("separates callable rows from unsupported local models", () => {
    const state = localModelsFromCatalog({
      _tag: "Ready",
      providers: [],
      failures: [],
      models: [makeSetupModel(false), makeFoundModel()].map((product) => ({ _tag: "Local", product, offering: Option.none() })),
      localModelPreparation: {
        discovery: { complete: true, modelsFound: 1 },
        assessment: { complete: true, settledModels: 2, totalModels: 2 },
      },
      unsupportedLocalModels: [unsupported],
      hiddenLocalModels: [],
    })
    expect(state.models.map(({ _tag }) => _tag)).toEqual(["Catalog", "Custom"])
    expect(state.hidden).toEqual([])
    expect(state.unsupported).toEqual([unsupported])
    expect(installedLocalModels(state).map(({ modelId }) => modelId)).toEqual([customModelId])
    expect(unsupportedLocalModelSourceTag(unsupported)).toBe("models")
  })

  it("keeps hidden models out of the callable rows", () => {
    const hidden = makeFoundModel()
    const state = localModelsFromCatalog({
      _tag: "Ready",
      providers: [],
      failures: [],
      models: [{ _tag: "Local", product: makeSetupModel(false), offering: Option.none() }],
      localModelPreparation: {
        discovery: { complete: true, modelsFound: 0 },
        assessment: { complete: true, settledModels: 1, totalModels: 1 },
      },
      unsupportedLocalModels: [],
      hiddenLocalModels: [hidden],
    })
    expect(state.models.map(({ modelId }) => modelId)).not.toContain(hidden.modelId)
    expect(state.hidden).toEqual([hidden])
  })

  it("treats a found model as installed and offerable, but never as managed", () => {
    const found = makeFoundModel()
    expect(localModelIsInstalled(found)).toBe(true)
    expect(localModelProviderModelId(found)).toEqual(Option.some(customModelId))
    expect(localModelSourceTag(found)).toEqual(Option.some("Hugging Face cache"))
    expect(localModelSourceTag(makeSetupModel(true))).toEqual(Option.none())
    expect(catalogVariantOfName([makeSetupModel(false)], found.presentation.variantOf)).toEqual(Option.some("Setup Model"))
    expect(catalogVariantOfName([], Option.some(CatalogFormModelIdSchema.make("other:gguf:q4")))).toEqual(Option.none())
    const listed = makeSetupModel(false)
    const otherVariant = CatalogFormModelIdSchema.make(`${listed.modelId.slice(0, listed.modelId.indexOf(":"))}:gguf:q2`)
    expect(catalogVariantOfName([listed], Option.some(otherVariant))).toEqual(Option.some("Setup Model"))
    const unavailable = makeFoundModel({
      acquisitionState: {
        _tag: "Unavailable",
        installation: { _tag: "Resolved", installedBytes: 1, primaryPath: "/gone.gguf", ownership: "External" },
        failure: { code: "missing", message: "Missing", retryable: false },
      },
    })
    expect(localModelResidency(unavailable)).toEqual(Option.none())
    expect(localModelProviderModelId(unavailable)).toEqual(Option.none())
  })

  it("finds a loading found model for the tray and retires its load feedback when ready", () => {
    const loading = makeFoundModel({
      acquisitionState: {
        _tag: "External",
        installation: { _tag: "Resolved", installedBytes: 2, primaryPath: "/m.gguf", ownership: "External" },
        residencyState: { _tag: "Ready", allocation: { contextWindowTokens: 4096, memoryDomains: [] } },
      },
    })
    const state: LocalModelsState = {
      preparation: { discovery: { complete: true, modelsFound: 1 }, assessment: { complete: true, settledModels: 1, totalModels: 1 } },
      models: [makeSetupModel(true), loading],
      unsupported: [],
      hidden: [],
    }
    expect(Option.map(activeLocalModel(state), ({ model }) => model.modelId)).toEqual(Option.some(customModelId))
    const failedLoad = {
      modelId: customModelId,
      operation: "load" as const,
      pending: false,
      failure: Option.some({ operation: "load" as const, rejection: Option.none() }),
    }
    expect(localModelCommandStatus(customModelId, [[failedLoad]], Option.some(loading)).failures).toEqual([])
  })
})

describe("repositories where nothing runs", () => {
  const assessment = (value: Record<string, unknown>) => Option.some(value as never)
  const version = (label: string, verdict: Option.Option<never>) => ({
    label, modelId: `hf:o/r/${label}.gguf`, files: [`${label}.gguf`], sizeBytes: 1, downloadBytes: 1,
    assessment: verdict, recommended: false, onThisComputer: Option.none(),
  }) as never
  const tooLarge = assessment({ _tag: "DoesNotFit" })
  const unsupportedTensors = (count: number) => assessment({
    _tag: "Unsupported",
    reasons: Array.from({ length: count }, (_, index) => reason({ _tag: "UnrecognizedTensor", name: `blk.${index}.attn_k` })),
  })

  it("says why no version runs from the engine's verdicts", () => {
    expect(noRunnableVersionReason([version("Q8_0", tooLarge)])).toEqual(Option.some("Every version is too large for this machine."))
    expect(noRunnableVersionReason([version("Q4_K_M", unsupportedTensors(60)), version("Q8_0", unsupportedTensors(60))]))
      .toEqual(Option.some("Magnitude can't run any version: Has parts Magnitude doesn't recognize (120 tensors)."))
    expect(noRunnableVersionReason([version("Q8_0", tooLarge), version("Q4_K_M", unsupportedTensors(1))]))
      .toEqual(Option.some("Some versions are too large for this machine, and Magnitude can't run the others: Has parts Magnitude doesn't recognize (blk.0.attn_k)."))
  })

  it("says nothing when some version fits or hasn't been checked", () => {
    expect(noRunnableVersionReason([version("Q8_0", tooLarge), version("Q4_K_M", assessment({ _tag: "Fits" }))])).toEqual(Option.none())
    expect(noRunnableVersionReason([version("Q8_0", tooLarge), version("Q4_K_M", Option.none() as never)])).toEqual(Option.none())
    expect(noRunnableVersionReason([])).toEqual(Option.none())
  })
})
