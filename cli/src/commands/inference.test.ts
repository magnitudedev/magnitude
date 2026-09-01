import { Command } from "@commander-js/extra-typings"
import { Option } from "effect"
import { CatalogFormModelIdSchema, type ModelCatalogState } from "@magnitudedev/sdk"
import { describe, expect, it } from "vitest"
import {
  makeAcquiringModel,
  makeCatalogModel,
  makeInstalledCatalogModel,
} from "../features/local-inference/test-fixtures"
import { registerInferenceCommands } from "./inference"
import { renderCatalog, renderModelsStatus } from "./inference-runtime"

const catalogState = (...models: ReturnType<typeof makeCatalogModel>[]): ModelCatalogState => ({
  _tag: "Ready",
  providers: [],
  models: models.map((product) => ({ _tag: "Local", product, offering: Option.none() })),
  failures: [],
  localModelPreparation: {
    discovery: { complete: true, modelsFound: models.length },
    assessment: { complete: true, settledModels: models.length, totalModels: models.length },
  },
})

describe("inference command surface", () => {
  it("exposes acquisition under catalog and residency under models", () => {
    const program = new Command().name("magnitude")
    registerInferenceCommands(program)

    expect(program.commands.map((command) => command.name())).toEqual([
      "hardware",
      "catalog",
      "models",
    ])
    expect(program.commands[1]!.commands.map((command) => command.name())).toEqual([
      "list",
      "show",
      "recommendations",
      "pull",
      "cancel",
      "remove",
    ])
    expect(program.commands[2]!.commands.map((command) => command.name())).toEqual([
      "status",
      "load",
      "stop",
    ])
    expect(program.commands[2]!.commands[2]!.registeredArguments).toHaveLength(0)
  })

  it("renders only fitting catalog evidence and exact model IDs", () => {
    const model = makeCatalogModel()
    const output = renderCatalog(catalogState(model))
    expect(output).toContain("Local model catalog - 1 compatible model")
    expect(output).toContain(model.modelId)
    expect(output).toContain("tok/s")
    expect(output).not.toContain("NotInstalled")
    expect(output).not.toContain("assessmentId")
  })

  it("summarizes outstanding assessment without emitting placeholder rows", () => {
    const ready = makeCatalogModel()
    const assessing = makeCatalogModel({
      modelId: CatalogFormModelIdSchema.make("assessing:gguf:q4"),
      servingState: { _tag: "Assessing", profile: { contextLength: 32_768 } },
    })
    const output = renderCatalog(catalogState(ready, assessing))
    expect(output).toContain("Assessing 1 additional catalog model; this list may grow.")
    expect(output).not.toContain(assessing.modelId)
  })

  it("renders explicit empty and failed-assessment catalog states", () => {
    expect(renderCatalog(catalogState())).toBe("No catalog models are compatible with this computer.\n")
    const failed = makeCatalogModel({
      servingState: {
        _tag: "Failed",
        profile: Option.none(),
        failure: { code: "assessment_failed", message: "Could not inspect this model", retryable: true },
      },
    })
    const output = renderCatalog(catalogState(failed))
    expect(output).toContain("Assessment failed for 1 catalog model.")
    expect(output).not.toContain(failed.modelId)
  })

  it("keeps acquisition and residency in model status", () => {
    const downloading = makeAcquiringModel({
      _tag: "Installing",
      progress: {
        stage: "downloading",
        completedBytes: 50,
        totalBytes: 100,
        bytesPerSecond: Option.none(),
      },
    })
    const installed = makeInstalledCatalogModel()
    const output = renderModelsStatus([downloading, installed])
    expect(output).toContain("Downloading 50%")
    expect(output).toContain("Unloaded")
    expect(output).toContain(downloading.modelId)
  })

  it("renders an explicit empty local-model state", () => {
    expect(renderModelsStatus([])).toBe("No local models are on this computer.\n")
  })
})
