import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import { servingHardware } from "@magnitudedev/client-common/fixtures/hardware"
import { streamedFitResidency, streamedReadyResidency } from "@magnitudedev/client-common/fixtures/model"
import {
  catalogState,
  fitsAssessment,
  makeCatalogModel,
  makeHardware,
  makeInstalledCatalogModel,
  recommendedModel,
} from "./fixtures/inference"
import {
  hardwareJson,
  renderHardware,
  renderModelDetail,
  renderRecommendations,
} from "./inference-runtime"

const streamedFit = () => {
  const model = makeCatalogModel()
  if (model.servingState._tag !== "Assessed") {
    throw new Error("the fixture model fits")
  }
  const assessment = { ...fitsAssessment(), residency: streamedFitResidency() }
  return makeCatalogModel({ servingState: { ...model.servingState, assessment } })
}

/** A model with a streamed fit, loaded with an expert bank: the combination a user sees. */
const streamedLoad = () => {
  const model = streamedFit()
  return makeInstalledCatalogModel({
    servingState: model.servingState,
    acquisitionState: {
      _tag: "Installed",
      installation: { _tag: "Resolved", installedBytes: 1, primaryPath: "/models/catalog-model.gguf", ownership: "Magnitude" },
      residencyState: streamedReadyResidency(["cuda-0", "system"]),
    },
  })
}

describe("expert streaming", () => {
  it("states a streamed fit's share, floor and floor speed in its details and recommendation", () => {
    const model = streamedFit()
    const detail = renderModelDetail({ model, models: [model], hardware: makeHardware(), drafters: { _tag: "NotApplicable" }, hidden: false })
    expect(detail).toMatch(/Experts this machine allows\s+60%, the rest read from disk/)
    expect(detail).toMatch(/Expert floor\s+1%/)
    expect(detail).toMatch(/Speed at the floor\s+~5 tok\/s/)
    const recommendation = renderRecommendations({
      catalog: catalogState([model]), models: [model], hardware: makeHardware(), developerCountry: Option.none(), recommended: [model].map(recommendedModel),
    })
    expect(recommendation).toContain(
      "   Experts this machine allows 60%, the rest read from disk - Expert floor 1% - Speed at the floor ~5 tok/s")
  })

  it("says nothing about streaming for a resident fit", () => {
    const model = makeCatalogModel()
    expect(renderModelDetail({ model, models: [model], hardware: makeHardware(), drafters: { _tag: "NotApplicable" }, hidden: false })).not.toContain("Expert")
  })

  it("states a loaded model's prompt group places and where its experts are", () => {
    const model = streamedLoad()
    const hardware = renderHardware({ hardware: servingHardware("Discrete"), models: [model] })
    expect(hardware).toContain("  Prompt group places: 1 of 8")
    expect(hardware).toContain("  Experts in memory: 40%, floor 1%, ceiling 60%")
    expect(hardware).toContain("  Experts in Test GPU · GPU 1: 300 MB")
    expect(hardware).toContain("  Experts in System memory: 400 MB")
    expect(hardware).toContain("  Cold experts in System memory: 210 MB, 50% of its most")
    expect(hardware).toContain("  Experts on disk: 300 MB")
    expect(hardware).toContain("  N-gram table in memory: 126 MB, 50% of its most, floor 25%")
    expect(hardware).toMatch(/ {2}Last resize: Experts shrank from 50% to 40% as memory grew short on /)
  })

  it("states a loaded streamed model's bank once in its detail, in place of its fit", () => {
    const model = streamedLoad()
    const detail = renderModelDetail({
      model, models: [model], hardware: servingHardware("Discrete"), drafters: { _tag: "NotApplicable" }, hidden: false,
    })
    expect(detail).toMatch(/Prompt group places\s+1 of 8/)
    expect(detail.match(/Experts in memory/g)).toHaveLength(1)
    expect(detail).toMatch(/Experts in memory\s+40%, floor 1%, ceiling 60%/)
    expect(detail).toMatch(/Experts in Test GPU · GPU 1\s+300 MB/)
    expect(detail).not.toContain("Experts this machine allows")
    const recommendation = renderRecommendations({
      catalog: catalogState([model]), models: [model], hardware: servingHardware("Discrete"), developerCountry: Option.none(), recommended: [model].map(recommendedModel),
    })
    expect(recommendation).toContain("   Experts in memory 40%, floor 1%, ceiling 60% - Experts in Test GPU · GPU 1 300 MB")
    expect(recommendation).not.toContain("Experts this machine allows")
  })

  it("encodes a loaded model's streaming status without memory-domain identities", () => {
    const json = Effect.runSync(hardwareJson.encode({ hardware: makeHardware(), models: [streamedLoad()] }))
    const text = JSON.stringify(json)
    expect(text).toContain("\"slots\":40")
    expect(text).toContain("\"promptGroupPlaces\":{\"held\":1,\"capacity\":8}")
    expect(text).toContain("\"_tag\":\"Shrank\"")
    expect(text).not.toContain("cuda-0")
  })
})
