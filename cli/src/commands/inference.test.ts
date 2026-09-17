import { Command } from "@commander-js/extra-typings"
import { Option } from "effect"
import { CatalogFormModelIdSchema, type ModelCatalogState } from "@magnitudedev/sdk"
import { describe, expect, it, vi } from "vitest"
import {
  makeAcquiringModel,
  makeCatalogModel,
  makeHardware,
  makeInstalledCatalogModel,
} from "./fixtures/inference"
import { registerInferenceCommands } from "./inference"
import {
  showRecommendations, showCatalogModel, pullModel, cancelDownload, removeModel, showModelsStatus, loadInstance,
  renderCatalog,
  renderCatalogStatus,
  renderModelsStatus,
  renderRecommendations,
} from "./inference-runtime"

const startupProbe = vi.hoisted(() => vi.fn(() => { throw new Error("Unexpected service startup") }))
vi.mock("../server/acn-connection", async () => {
  const { Effect } = await import("effect")
  return { headlessAcnConnection: Effect.sync(startupProbe) }
})

type CatalogSnapshotState = Exclude<ModelCatalogState, { readonly _tag: "Initializing" }>

const catalogState = (...models: ReturnType<typeof makeCatalogModel>[]): CatalogSnapshotState => ({
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
  it.each([
    [() => showRecommendations("unknown", "10"), "Preference must be one of: fastest, faster, balanced, smarter, smartest"],
    [() => showRecommendations("balanced", "0"), "Limit must be a positive integer"],
    [() => showCatalogModel(""), "Invalid catalog model ID: "],
    [() => pullModel(""), "Invalid catalog model ID: "],
    [() => cancelDownload(""), "Invalid catalog model ID: "],
    [() => removeModel(""), "Invalid catalog model ID: "],
    [() => showModelsStatus(""), "Invalid catalog model ID: "],
    [() => loadInstance(""), "Invalid catalog model ID: "],
    [() => loadInstance("hf:test/model/model.gguf"), "Invalid catalog model ID: hf:test/model/model.gguf"],
    [() => showModelsStatus("hf:test/model/model.gguf"), "Invalid catalog model ID: hf:test/model/model.gguf"],
  ])("validates model arguments before acquiring a service", async (command, message) => {
    const exitCode = process.exitCode
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true)
    startupProbe.mockClear()
    try {
      await command()
      expect(stderr).toHaveBeenCalledWith(`${message}\n`)
      expect(process.exitCode).toBe(1)
      expect(startupProbe).not.toHaveBeenCalled()
    } finally {
      stderr.mockRestore()
      process.exitCode = exitCode
    }
  })

  it("exposes acquisition under catalog and residency under models", () => {
    const program = new Command().name("magnitude")
    registerInferenceCommands(program)

    expect(program.commands.map((command) => command.name())).toEqual([
      "hardware",
      "catalog",
      "models",
    ])
    expect(program.commands[1]!.commands.map((command) => command.name())).toEqual([
      "status",
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
    for (const command of program.commands[2]!.commands) {
      expect(command.options).toHaveLength(0)
    }
    expect(program.commands[0]!.options).toHaveLength(0)
    for (const command of program.commands[1]!.commands) {
      expect(command.options.map(({ long }) => long)).not.toContain("--json")
    }
  })

  it.each([
    ["models", "status", "--json"],
    ["models", "load", "test-model", "--json"],
    ["models", "stop", "--json"],
  ])("rejects the removed JSON mode: %s %s", async (...args) => {
    const program = new Command().name("magnitude").exitOverride().configureOutput({ writeErr: () => {} })
    registerInferenceCommands(program)
    await expect(program.parseAsync(args, { from: "user" })).rejects.toMatchObject({ code: "commander.unknownOption" })
  })

  it("renders assessment progress without arbitrary-model discovery", () => {
    const active: ModelCatalogState = {
      ...catalogState(),
      localModelPreparation: {
        discovery: { complete: true, modelsFound: 4 },
        assessment: { complete: false, settledModels: 3, totalModels: 4 },
      },
    }
    expect(renderCatalogStatus(active)).toBe([
      "Model catalog preparation",
      "Assessment: In progress - 3 of 4 models assessed",
      "",
    ].join("\n"))

    expect(renderCatalogStatus({
      ...active,
      localModelPreparation: {
        discovery: { complete: true, modelsFound: 4 },
        assessment: { complete: true, settledModels: 4, totalModels: 4 },
      },
    })).toContain("Assessment: Complete - 4 of 4 models assessed")
  })

  it("reports assessment as pending while the catalog initializes", () => {
    expect(renderCatalogStatus({ _tag: "Initializing" })).toBe([
      "Model catalog preparation",
      "Assessment: In progress",
      "",
    ].join("\n"))
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

  it("links recommendation evidence to the bundled methodology topic", () => {
    const model = makeCatalogModel()
    const catalog = catalogState(model)
    const output = renderRecommendations({
      catalog,
      models: [model],
      hardware: makeHardware(),
      preference: { label: "Balanced" },
      ranked: [model],
    })

    expect(output).toContain("Local model recommendations - Balanced")
    expect(output).toContain("Learn more: magnitude docs recommendations")
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
