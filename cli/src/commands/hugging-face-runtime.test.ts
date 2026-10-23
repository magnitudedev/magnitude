import { Effect, Option } from "effect"
import { describe, expect, it, vi } from "vitest"
import { CatalogFormModelIdSchema } from "@magnitudedev/sdk"
import { acquisitionHint, parseCatalogComponent, pull, pullComponent, pullJson, renderInspection, renderPull, type PullResult } from "./hugging-face-runtime"
import type { CliModelsClient } from "./inference-client"
import { parseDrafterArgument, parsePullReference } from "./model-references"
import { fitsAssessment, makeInspection, TEST_REPOSITORY } from "./fixtures/inference"

const catalogModel = CatalogFormModelIdSchema.make("qwen-test:gguf:q4")
const [tooLarge, onDisk, recommended, unsupported] = makeInspection().versions

describe("catalog inspect with a named version that doesn't run here", () => {
  it("explains a too-large version, how to force it, and the recommended version", () => {
    const inspection = makeInspection({ selectedVersion: Option.some("Q8_K_P") })
    expect(acquisitionHint(inspection)).toBe([
      `Q8_K_P needs 8.6 GB more memory than this computer can provide. Download it anyway with: magnitude catalog pull ${TEST_REPOSITORY}:Q8_K_P --force`,
      `The recommended version is Q4_K_P. Download: magnitude catalog pull ${TEST_REPOSITORY}:Q4_K_P`,
    ].join("\n"))
    expect(renderInspection({ inspection, drafters: [], models: [] })).not.toContain("No version of this model runs")
  })

  it("explains an unsupported version with its reasons and points to the recommended version", () => {
    const hint = acquisitionHint(makeInspection({ selectedVersion: Option.some("Q2_K_P") }))
    expect(hint).toBe([
      "Magnitude can't run Q2_K_P: Uses Q2_K weights, which Magnitude doesn't support yet.",
      `The recommended version is Q4_K_P. Download: magnitude catalog pull ${TEST_REPOSITORY}:Q4_K_P`,
    ].join("\n"))
  })

  it("says other versions run when none is recommended, and that nothing runs only when nothing does", () => {
    const withoutRecommendation = makeInspection({
      selectedVersion: Option.some("Q8_K_P"),
      versions: [tooLarge!, { ...onDisk!, recommended: false }],
    })
    expect(acquisitionHint(withoutRecommendation).split("\n").at(-1)).toBe("Other versions run here; see the table above.")
    const nothingRuns = makeInspection({ selectedVersion: Option.some("Q8_K_P"), versions: [tooLarge!, unsupported!] })
    expect(acquisitionHint(nothingRuns).split("\n")).toEqual([
      `Q8_K_P needs 8.6 GB more memory than this computer can provide. Download it anyway with: magnitude catalog pull ${TEST_REPOSITORY}:Q8_K_P --force`,
      "No version of this model runs on this computer.",
    ])
  })

  it("never marks a named version that can't run as selected", () => {
    for (const label of ["Q8_K_P", "Q2_K_P"]) {
      const output = renderInspection({ inspection: makeInspection({ selectedVersion: Option.some(label) }), drafters: [], models: [] })
      expect(output).not.toContain("(selected)")
      expect(output).toMatch(/^Q4_K_P .*Fits \(recommended\)$/m)
    }
    expect(renderInspection({ inspection: makeInspection({ selectedVersion: Option.some("Q6_K_P") }), drafters: [], models: [] }))
      .toMatch(/^Q6_K_P .*Fits \(selected\) · already on this computer$/m)
  })

  it("keeps the download hint for a named version that fits", () => {
    expect(acquisitionHint(makeInspection({ selectedVersion: Option.some("Q6_K_P") })))
      .toBe(`Download: magnitude catalog pull ${TEST_REPOSITORY}:Q6_K_P`)
  })
})

describe("a version that is exactly a catalog model", () => {
  it("points catalog inspect at the catalog model", () => {
    const inspection = makeInspection({
      versions: [{ ...recommended!, catalogModel: Option.some(catalogModel), assessment: Option.some(fitsAssessment()) }],
    })
    expect(acquisitionHint(inspection))
      .toBe(`This is the catalog model ${catalogModel}, which comes with its drafter. Download: magnitude catalog pull ${catalogModel}`)
  })

  const routed = (outcome: "Started" | "AlreadyOnThisComputer"): PullResult => ({
    _tag: "HuggingFace",
    repository: TEST_REPOSITORY,
    versionRequested: false,
    added: { _tag: "Catalog", modelId: catalogModel, version: "Q4_K_P", outcome },
  })

  it("says catalog pull installed the catalog model instead of adding a Hugging Face model", () => {
    expect(renderPull(routed("Started"))).toBe([
      `${TEST_REPOSITORY}:Q4_K_P (the recommended version) is the catalog model ${catalogModel}, so Magnitude installs that model with its drafter.`,
      `Model ID: ${catalogModel}`,
      `Check progress: magnitude models status ${catalogModel}`,
      "",
    ].join("\n"))
    expect(renderPull(routed("AlreadyOnThisComputer"))).toBe([
      `${TEST_REPOSITORY}:Q4_K_P (the recommended version) is the catalog model ${catalogModel}, which is already on this computer. Nothing to download.`,
      `Model ID: ${catalogModel}`,
      `Load it: magnitude models load ${catalogModel}`,
      "",
    ].join("\n"))
  })

  it("passes the service's catalog routing through pull and its JSON", async () => {
    const addHuggingFaceModel = vi.fn(() => Effect.succeed({ _tag: "Catalog", modelId: catalogModel, version: "Q4_K_P", outcome: "Started" }))
    const client = { models: { addHuggingFaceModel } } as unknown as CliModelsClient
    const reference = Effect.runSync(parsePullReference(TEST_REPOSITORY))
    const result = await Effect.runPromise(pull(client, TEST_REPOSITORY, reference, Effect.runSync(parseDrafterArgument("auto", "/work"))))
    expect(addHuggingFaceModel).toHaveBeenCalledWith(expect.objectContaining({ repository: TEST_REPOSITORY, drafter: { _tag: "Auto" } }))
    expect(result).toEqual(routed("Started"))
    const document = await Effect.runPromise(pullJson.encode(result))
    expect(document).toEqual({
      _tag: "HuggingFace",
      repository: TEST_REPOSITORY,
      versionRequested: false,
      added: { _tag: "Catalog", modelId: catalogModel, version: "Q4_K_P", outcome: "Started" },
    })
  })
})

describe("downloading one catalog component", () => {
  it("downloads only the projector of a catalog model and says so", async () => {
    const downloadCatalogComponent = vi.fn(() => Effect.succeed({ started: true }))
    const syncLocalModel = vi.fn(() => Effect.die("only the component is downloaded"))
    const client = { models: { downloadCatalogComponent, syncLocalModel } } as unknown as CliModelsClient
    const component = Effect.runSync(parseCatalogComponent("projector"))
    const result = await Effect.runPromise(pullComponent(client, catalogModel, component))
    expect(downloadCatalogComponent).toHaveBeenCalledWith({ modelId: catalogModel, component: "Projector" })
    expect(syncLocalModel).not.toHaveBeenCalled()
    expect(renderPull(result)).toBe([
      `Downloading only the image input for ${catalogModel}.`,
      `Check progress: magnitude models status ${catalogModel}`,
      "",
    ].join("\n"))
    expect(await Effect.runPromise(pullJson.encode(result)))
      .toEqual({ _tag: "Component", modelId: catalogModel, component: "Projector", started: true })
    expect(renderPull({ ...result, started: false } as PullResult)).toBe(`${catalogModel} already has its image input. Nothing to download.\n`)
  })

  it("accepts only projector or drafter", () => {
    expect(Effect.runSync(parseCatalogComponent("Drafter"))).toBe("Drafter")
    expect(Effect.runSync(Effect.flip(parseCatalogComponent("weights"))).message).toBe("--only takes projector or drafter, not weights.")
  })
})
