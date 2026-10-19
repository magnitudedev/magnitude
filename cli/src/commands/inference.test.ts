import { Command } from "@commander-js/extra-typings"
import { Effect, Option } from "effect"
import {
  CatalogFormModelIdSchema,
  CustomFormModelIdSchema,
  DirectoryNotFound,
  ModelVersionNotRunnable,
  type CustomLocalModel,
  type ModelCatalogState,
} from "@magnitudedev/sdk"
import { afterEach, describe, expect, it, vi } from "vitest"
import {
  ADDED_MODEL_ID,
  catalogState,
  doesNotFitAssessment,
  FOUND_MODEL_ID,
  makeAcquiringModel,
  makeAddedModel,
  makeCatalogModel,
  makeDrafterCandidate,
  makeDrafterCandidates,
  makeFoundModel,
  makeHardware,
  makeImportedModel,
  makeInspection,
  makeInstalledCatalogModel,
  makeLocations,
  makeSearchResponse,
  makeUnsupportedLocalModel,
  makeUnsupportedSearchResult,
  TEST_COMMIT,
  TEST_REPOSITORY,
  unsupportedAssessment,
  unreadableAssessment,
} from "./fixtures/inference"
import { registerInferenceCommands } from "./inference"
import type { CliModelsClient } from "./inference-client"
import { parseDrafterArgument, parsePullReference } from "./model-references"
import { asJson, jsonDocument, runCommand } from "./output"
import { findModel } from "./model-format"
import {
  addModelFolder,
  cancelDownload,
  catalogStatusJson,
  hardwareJson,
  hideModel,
  importModel,
  inspectCandidate,
  inspectionJson,
  loadInstance,
  modelDetailJson,
  modelsStatusJson,
  pullJson,
  pullModel,
  removeModel,
  removeModelFolder,
  renderCatalog,
  renderCatalogDetail,
  renderCatalogStatus,
  renderImport,
  renderInspection,
  downloadSizeLine,
  execute,
  renderModelDetail,
  parseAutomaticLocation,
  renderModelLocations,
  renderModelsStatus,
  renderPull,
  renderRecommendations,
  renderSearch,
  renderSetDrafter,
  renderRemoval,
  searchHuggingFace,
  searchJson,
  setModelDrafter,
  showCatalogModel,
  showModelsStatus,
  showRecommendations,
  unhideModel,
  versionVerdict,
  importFile,
  ImportResultSchema,
  ModelLocationsJsonSchema,
  modelsStatusResultJson,
  catalogDetailJson,
  catalogListJson,
  recommendationsJson,
  pull,
  readModelsStatus,
  readSearch,
  renderModelsStatusResult,
  setDrafter,
  type ImportResult,
  type ModelsStatusView,
} from "./inference-runtime"

afterEach(() => {
  vi.restoreAllMocks()
  process.exitCode = undefined
})

const capture = async (command: () => Effect.Effect<void> | Promise<void>) => {
  let stdout = ""
  let stderr = ""
  vi.spyOn(process.stdout, "write").mockImplementation((chunk) => { stdout += String(chunk); return true })
  vi.spyOn(process.stderr, "write").mockImplementation((chunk) => { stderr += String(chunk); return true })
  await execute(command())
  return { stdout, stderr, exitCode: process.exitCode }
}

const fakeClient = (handlers: Record<string, (payload: never) => Effect.Effect<unknown, unknown>>) => {
  const models = Object.fromEntries(Object.entries(handlers).map(([name, handler]) => [name, vi.fn(handler)]))
  return { client: { models } as unknown as CliModelsClient, models }
}

const run = <A>(effect: Effect.Effect<A, unknown>) => Effect.runPromise(Effect.either(effect))

const statusView = (overrides: Partial<ModelsStatusView> = {}): ModelsStatusView => ({
  models: [],
  unsupported: [],
  hidden: [],
  all: false,
  ...overrides,
})

const leafCommands = (command: Command): Command[] => command.commands.length === 0
  ? [command as Command]
  : command.commands.flatMap((child) => leafCommands(child as Command))

const program = () => {
  const root = new Command().name("magnitude").exitOverride().configureOutput({ writeErr: () => {}, writeOut: () => {} })
  registerInferenceCommands(root)
  return root
}

describe("inference argument validation", () => {
  it.each([
    [() => showRecommendations("unknown", "10"), "Preference must be one of: fastest, faster, balanced, smarter, smartest"],
    [() => showRecommendations("balanced", "0"), "Limit must be a positive integer"],
    [() => showCatalogModel(""), "Invalid catalog model ID: "],
    [() => showCatalogModel("hf:test/model/model.gguf"), "Invalid catalog model ID: hf:test/model/model.gguf"],
    [() => pullModel(""), "Expected a model ID, a Hugging Face link, or owner/repo[:QUANT]: "],
    [() => pullModel("not a model"), "Expected a model ID, a Hugging Face link, or owner/repo[:QUANT]: not a model"],
    [() => pullModel("https://example.com/owner/repo"), "Expected a model ID, a Hugging Face link, or owner/repo[:QUANT]: https://example.com/owner/repo"],
    [() => pullModel("test:gguf:q4", { drafter: "none" }), "Catalog models come with their drafter; --drafter applies only to models from Hugging Face."],
    [() => pullModel("owner/repo", { drafter: " " }), "Drafter must be auto, none, built-in, a model ID, a Hugging Face reference, or a GGUF path"],
    [() => cancelDownload(""), "Invalid model ID: "],
    [() => removeModel("nonsense"), "Invalid model ID: nonsense"],
    [() => showModelsStatus(""), "Invalid model ID: "],
    [() => loadInstance("local:not-a-digest"), "Invalid model ID: local:not-a-digest"],
    [() => searchHuggingFace("  "), "Search query must not be empty"],
    [() => searchHuggingFace("qwen", { limit: "101" }), "Limit must be an integer from 1 to 100"],
    [() => searchHuggingFace("qwen", { limit: "0" }), "Limit must be an integer from 1 to 100"],
    [() => inspectCandidate(" "), "Expected a Hugging Face link, owner/repo[:QUANT], or a GGUF path"],
    [() => importModel("/models/readme.txt"), "Expected a .gguf file: /models/readme.txt"],
    [() => importModel(""), "Expected a path to a .gguf file"],
    [() => setModelDrafter("test:gguf:q4", "auto"),
      "test:gguf:q4 is a catalog model. Catalog models keep the drafter Magnitude ships with them; it changes only when you update with `magnitude catalog pull test:gguf:q4`."],
    [() => setModelDrafter("garbage", "auto"), "Invalid model ID: garbage"],
    [() => hideModel("nope"), "Invalid model ID: nope"],
    [() => unhideModel("nope"), "Invalid model ID: nope"],
    [() => addModelFolder(""), "Expected a folder path"],
    [() => removeModelFolder("  "), "Expected a folder path"],
  ])("rejects invalid input before acquiring a service (%#)", async (command, message) => {
    const result = await capture(command)
    expect(result.stderr).toBe(`${message}\n`)
    expect(result.stdout).toBe("")
    expect(result.exitCode).toBe(1)
  })
})

describe("inference command surface", () => {
  it("exposes finding and downloading under catalog and this computer under models", () => {
    const root = program()
    expect(root.commands.map((command) => command.name())).toEqual(["hardware", "catalog", "models"])
    expect(root.commands[1]!.commands.map((command) => command.name())).toEqual([
      "status", "list", "show", "recommendations", "search", "inspect", "pull", "cancel", "remove",
    ])
    expect(root.commands[2]!.commands.map((command) => command.name())).toEqual([
      "status", "load", "stop", "import", "drafter", "hide", "unhide", "folders",
    ])
    expect(root.commands[2]!.commands[7]!.commands.map((command) => command.name())).toEqual(["list", "add", "remove", "enable", "disable"])
    expect(root.commands[2]!.commands[2]!.registeredArguments.map((argument) => argument.required)).toEqual([false])
  })

  it("accepts --json on every inference command", () => {
    const leaves = leafCommands(program())
    expect(leaves.length).toBe(22)
    for (const command of leaves) {
      expect(command.options.map(({ long }) => long), command.name()).toContain("--json")
    }
  })

  it("declares the options and arguments of each command", () => {
    const shape = Object.fromEntries(leafCommands(program()).map((command) => [
      `${command.parent!.name()} ${command.name()}`,
      {
        arguments: command.registeredArguments.map((argument) => `${argument.required ? "<" : "["}${argument.name()}${argument.required ? ">" : "]"}`),
        options: command.options.map(({ long }) => long).filter((long) => long !== "--json"),
      },
    ]))
    expect(shape).toEqual({
      "magnitude hardware": { arguments: [], options: [] },
      "catalog status": { arguments: [], options: [] },
      "catalog list": { arguments: [], options: [] },
      "catalog show": { arguments: ["<model-id>"], options: [] },
      "catalog recommendations": { arguments: [], options: ["--preference", "--limit"] },
      "catalog search": { arguments: ["<query>"], options: ["--limit"] },
      "catalog inspect": { arguments: ["<reference>"], options: [] },
      "catalog pull": { arguments: ["<model>"], options: ["--drafter", "--force", "--only"] },
      "catalog cancel": { arguments: ["<model-id>"], options: [] },
      "catalog remove": { arguments: ["<model-id>"], options: [] },
      "models status": { arguments: ["[model-id]"], options: ["--all"] },
      "models load": { arguments: ["<model-id>"], options: [] },
      "models stop": { arguments: ["[model-id]"], options: [] },
      "models import": { arguments: ["<path>"], options: ["--drafter"] },
      "models drafter": { arguments: ["<model-id>", "<drafter>"], options: [] },
      "models hide": { arguments: ["<model-id>"], options: [] },
      "models unhide": { arguments: ["<model-id>"], options: [] },
      "folders list": { arguments: [], options: [] },
      "folders add": { arguments: ["<path>"], options: [] },
      "folders remove": { arguments: ["<path>"], options: [] },
      "folders enable": { arguments: ["<location>"], options: [] },
      "folders disable": { arguments: ["<location>"], options: [] },
    })
  })

  it("gives every command and option help text", () => {
    for (const command of leafCommands(program())) {
      expect(command.description().length, command.name()).toBeGreaterThan(0)
      for (const option of command.options) expect(option.description.length).toBeGreaterThan(0)
      for (const argument of command.registeredArguments) expect(argument.description.length).toBeGreaterThan(0)
    }
  })

  it.each([
    [["catalog", "search"]],
    [["catalog", "inspect"]],
    [["catalog", "pull"]],
    [["models", "import"]],
    [["models", "drafter", "hf:a/b/c.gguf"]],
    [["models", "hide"]],
    [["models", "unhide"]],
    [["models", "folders", "add"]],
    [["models", "folders", "remove"]],
    [["models", "folders", "enable"]],
    [["models", "folders", "disable"]],
  ])("requires arguments: %j", async (args) => {
    await expect(program().parseAsync(args, { from: "user" })).rejects.toMatchObject({ code: "commander.missingArgument" })
  })

  it.each([
    [["catalog", "search", "qwen", "--quant", "Q4"]],
    [["models", "folders", "list", "--all"]],
    [["models", "stop", "--force"]],
  ])("rejects unknown options: %j", async (args) => {
    await expect(program().parseAsync(args, { from: "user" })).rejects.toMatchObject({ code: "commander.unknownOption" })
  })
})

describe("catalog renderers", () => {
  it("renders assessment progress", () => {
    const active: ModelCatalogState = {
      ...catalogState(),
      localModelPreparation: {
        discovery: { complete: true, modelsFound: 4 },
        assessment: { complete: false, settledModels: 3, totalModels: 4 },
      },
    }
    expect(renderCatalogStatus(active)).toBe("Model catalog preparation\nAssessment: In progress - 3 of 4 models assessed\n")
    expect(renderCatalogStatus({ _tag: "Initializing" })).toBe("Model catalog preparation\nAssessment: In progress\n")
  })

  it("renders only fitting catalog evidence and exact model IDs", () => {
    const model = makeCatalogModel()
    const output = renderCatalog(catalogState([model, makeFoundModel()]))
    expect(output).toContain("Local model catalog - 1 compatible model")
    expect(output).toContain(model.modelId)
    expect(output).not.toContain(FOUND_MODEL_ID)
    expect(output).not.toContain("assessmentId")
  })

  it("summarizes outstanding and failed assessment without placeholder rows", () => {
    const assessing = makeCatalogModel({
      modelId: CatalogFormModelIdSchema.make("assessing:gguf:q4"),
      servingState: { _tag: "Assessing", profile: { contextLength: 32_768 } },
    })
    expect(renderCatalog(catalogState([makeCatalogModel(), assessing]))).toContain("Assessing 1 additional catalog model; this list may grow.")
    expect(renderCatalog(catalogState())).toBe("No catalog models are compatible with this computer.\n")
  })

  it("ends recommendations with the methodology and Hugging Face hints", () => {
    const model = makeCatalogModel()
    const view = { catalog: catalogState([model]), models: [model], hardware: makeHardware(), preference: { label: "Balanced" } }
    const lines = renderRecommendations({ ...view, ranked: [model] }).trimEnd().split("\n")
    expect(lines.at(-2)).toBe("Learn more: magnitude docs recommendations")
    expect(lines.at(-1)).toBe("Looking for a specific version, finetune, or uncensored model? magnitude catalog search \"<model name>\"")
    expect(renderRecommendations({ ...view, ranked: [] }).trimEnd().split("\n").at(-1)).toContain("magnitude catalog search")
  })

  it("prints the search command for other versions in catalog details", () => {
    const model = makeCatalogModel()
    expect(renderCatalogDetail({ model, models: [model] }).trimEnd().split("\n").at(-1))
      .toBe("Other versions: magnitude catalog search \"Qwen Test\"")
  })
})

describe("Hugging Face search", () => {
  it("lists repositories with the catalog model each is a variant of", () => {
    const output = renderSearch({ query: "qwen test", response: makeSearchResponse(), models: [makeCatalogModel()] })
    expect(output).toBe([
      "Hugging Face models matching \"qwen test\"",
      "",
      "MODEL                  UPLOADER   VARIANT OF  DOWNLOADS  REPOSITORY",
      "Qwen Test Uncensored   HauhauCS   Qwen Test   48K        HauhauCS/Qwen-Test-Uncensored-GGUF",
      "Qwen Test abliterated  huihui-ai  —           912        huihui-ai/Qwen-Test-abliterated-GGUF",
      "",
      "Inspect one: magnitude catalog inspect <repository>",
      "",
    ].join("\n"))
  })

  it("labels a repository whose files need access on Hugging Face", () => {
    const [first, ...rest] = makeSearchResponse().results
    const output = renderSearch({ query: "qwen test", response: makeSearchResponse({ results: [{ ...first!, gated: true }, ...rest] }), models: [makeCatalogModel()] })
    expect(output).toContain("Qwen Test Uncensored · Requires access on Hugging Face")
  })

  it("lists results of architectures Magnitude doesn't run apart, with why", () => {
    const response = makeSearchResponse({ results: [...makeSearchResponse().results, makeUnsupportedSearchResult()] })
    const output = renderSearch({ query: "qwen test", response, models: [makeCatalogModel()] })
    expect(output).toContain([
      "Can't run in Magnitude",
      "",
      "MODEL              REASON                               REPOSITORY",
      "DeepSeek V4 Flash  Magnitude doesn't run deepseek4 yet  unsloth/DeepSeek-V4-Flash-GGUF",
    ].join("\n"))
    expect(output.split("Can't run in Magnitude")[0]).not.toContain("DeepSeek")
    const only = renderSearch({ query: "deepseek", response: makeSearchResponse({ results: [makeUnsupportedSearchResult()] }), models: [] })
    expect(only).toContain("No Hugging Face models that run in Magnitude match \"deepseek\".")
    expect(only).not.toContain("Inspect one")
  })
})

describe("candidate inspection", () => {
  it("prints the repository step with verdicts, drafters, and the download command", () => {
    const output = renderInspection({ inspection: makeInspection(), drafters: makeDrafterCandidates(), models: [makeCatalogModel()] })
    const lines = output.split("\n")
    expect(lines.slice(0, 9)).toEqual([
      "Qwen Test Uncensored",
      "Variant of Qwen Test · apache-2.0 · pinned to 4f1c2e9",
      "",
      "VERSION  SIZE     ON THIS COMPUTER",
      "Q8_K_P   43.6 GB  Too large",
      "Q6_K_P   30.6 GB  Fits · already on this computer",
      "Q4_K_P   23.4 GB  Fits (recommended)",
      "Q2_K_P   15 GB    Unsupported: Q2_K weights",
      "",
    ])
    expect(lines[9]).toMatch(/^DRAFTER\s+SIZE\s+VERDICT\s+NOTE\s+USE$/)
    expect(lines[10]).toContain("Can't be used: Its drafter wasn't built for this model")
    expect(lines[10]).toMatch(/^ {2}HauhauCS\/Qwen-Test-Uncensored-GGUF\/dflash\.gguf\s+1\.2 GB\s+Can't be used.*\s+In this repository\s+/)
    expect(lines[11]).toMatch(/^\* Magnitude DFlash for Qwen Test\s+900 MB\s+Works · DFlash\s+Recommended · Trained for Qwen Test\s+qwen-test:gguf:q4$/)
    expect(lines[12]).toMatch(/^ {2}MTP heads\s+Included\s+Works · MTP\s+Built into this model\s+built-in$/)
    expect(lines[13]).toMatch(/^ {2}None\s+—\s+—\s+Run without a drafter\s+none$/)
    expect(lines.at(-3)).toBe("Download size: 24.3 GB (model 23.4 GB + drafter 900 MB)")
    expect(lines.at(-2)).toBe(`Download: magnitude catalog pull ${TEST_REPOSITORY}:Q4_K_P`)
  })

  it("totals the download with the projector and the preselected drafter, omitting what is already here", () => {
    const [, onDisk, recommended] = makeInspection().versions
    const projector = Option.some({ path: "mmproj.gguf", sizeBytes: 670_000_000 })
    const inspection = makeInspection({
      projector,
      versions: [{ ...recommended!, sizeBytes: 3_010_000_000, downloadBytes: 3_680_000_000 }],
    })
    const [, drafter] = makeDrafterCandidates()
    expect(downloadSizeLine(inspection, inspection.versions[0]!, [{ ...drafter!, sizeBytes: 685_000_000 }]))
      .toBe("Download size: 4.37 GB (model 3.01 GB + projector 670 MB + drafter 685 MB)")
    expect(downloadSizeLine(inspection, { ...onDisk!, downloadBytes: 0 }, [{ ...drafter!, sizeBytes: 685_000_000 }]))
      .toBe("Download size: 685 MB (drafter)")
    expect(downloadSizeLine(inspection, { ...onDisk!, downloadBytes: 0 }, [{ ...drafter!, onThisComputer: true }]))
      .toBe("Download size: nothing to download, it's all on this computer.")
  })

  it("uses a selected version for the drafter list and hint", () => {
    const inspection = makeInspection({ selectedVersion: Option.some("Q6_K_P") })
    const output = renderInspection({ inspection, drafters: [], models: [] })
    expect(output).toContain("Fits (selected) · already on this computer")
    expect(output.trimEnd().split("\n").at(-1)).toBe(`Download: magnitude catalog pull ${TEST_REPOSITORY}:Q6_K_P`)
  })

  it("explains an unsupported architecture instead of listing versions", () => {
    const output = renderInspection({
      inspection: makeInspection({ versions: [], unsupportedArchitecture: Option.some([{ _tag: "Architecture", architecture: "deepseek4" }]) }),
      drafters: [],
      models: [],
    })
    expect(output).toContain("Magnitude can't run this model: No kernels for the deepseek4 architecture.")
    expect(output).not.toContain("VERSION")
    expect(output).not.toContain("Download:")
  })

  it("says when no version runs here and points local files at import", () => {
    const tooLarge = makeInspection({
      versions: makeInspection().versions.map((version) => ({ ...version, recommended: false, assessment: Option.some(doesNotFitAssessment()) })),
    })
    expect(renderInspection({ inspection: tooLarge, drafters: [], models: [] })).toContain("No version of this model runs on this computer. Every version is too large for this machine.")
    const nothingRuns = makeInspection({ versions: [makeInspection().versions[0]!, makeInspection().versions[3]!] })
    expect(renderInspection({ inspection: nothingRuns, drafters: [], models: [] })).toContain(
      "No version of this model runs on this computer. Some versions are too large for this machine, and Magnitude can't run the others: Uses Q2_K weights, which Magnitude doesn't support yet.",
    )
    const local = makeInspection({
      source: { _tag: "Local", path: "/models" },
      versions: [{ ...makeInspection().versions[2]!, label: "Q4_K_M", files: ["qwen.gguf"] }],
    })
    expect(renderInspection({ inspection: local, drafters: [], models: [] }).trimEnd().split("\n").at(-1))
      .toBe("Add: magnitude models import /models/qwen.gguf")
  })

  it("marks verdicts", () => {
    const [large, onDisk, recommended] = makeInspection().versions
    expect(versionVerdict(large!, false)).toBe("Too large")
    expect(versionVerdict(onDisk!, true)).toBe("Fits (selected) · already on this computer")
    expect(versionVerdict(recommended!, true)).toBe("Fits (recommended)")
    expect(versionVerdict({ ...recommended!, assessment: Option.none() }, false)).toBe("Not assessed (recommended)")
    const tensors = Array.from({ length: 60 }, (_, index) => ({ _tag: "UnrecognizedTensor" as const, name: `blk.${index}.attn_k` }))
    expect(versionVerdict({ ...large!, assessment: Option.some({
      _tag: "Unsupported", environmentId: "environment", profile: { contextLength: 32_768 }, reasons: tensors, detail: "",
    } as never) }, false)).toBe("Unsupported: Has parts Magnitude doesn't recognize (60 tensors)")
  })
})

describe("pull and model mutations", () => {
  const added = (outcome: "Started" | "AlreadyOnThisComputer", drafter = Option.some(makeDrafterCandidate())) => ({
    _tag: "Custom" as const,
    modelId: ADDED_MODEL_ID,
    version: "Q4_K_P",
    drafter,
    outcome,
    drafterDownloadStarted: outcome === "AlreadyOnThisComputer" && Option.isSome(drafter),
  })

  it("prints the chosen version, drafter, and model ID", () => {
    expect(renderPull({ _tag: "HuggingFace", repository: TEST_REPOSITORY, versionRequested: false, added: added("Started") })).toBe([
      `Downloading ${TEST_REPOSITORY}:Q4_K_P (the recommended version).`,
      "Drafter: Magnitude DFlash for Qwen Test",
      `Model ID: ${ADDED_MODEL_ID}`,
      `Check progress: magnitude models status ${ADDED_MODEL_ID}`,
      "",
    ].join("\n"))
  })

  it("downloads only the drafter for a version already on this computer", () => {
    const output = renderPull({ _tag: "HuggingFace", repository: TEST_REPOSITORY, versionRequested: true, added: added("AlreadyOnThisComputer") })
    expect(output).toContain(`${TEST_REPOSITORY}:Q4_K_P is already on this computer. Downloading only the drafter.`)
    expect(output).toContain("Check progress:")
    const nothing = renderPull({ _tag: "HuggingFace", repository: TEST_REPOSITORY, versionRequested: true, added: added("AlreadyOnThisComputer", Option.none()) })
    expect(nothing).toContain("Nothing to download.")
    expect(nothing).toContain("Drafter: none")
    expect(nothing).toContain(`Load it: magnitude models load ${ADDED_MODEL_ID}`)
  })

  it("says whether a pull downloads, resumes, or updates a model", () => {
    const modelId = CatalogFormModelIdSchema.make("qwen-test:gguf:q4")
    expect(renderPull({ _tag: "Catalog", modelId, outcome: "AlreadyCurrent" })).toBe(`${modelId} is already installed and up to date.\n`)
    expect(renderPull({ _tag: "Catalog", modelId, outcome: "Installing" }))
      .toBe(`Downloading ${modelId}.\nCheck progress: magnitude models status ${modelId}\n`)
    expect(renderPull({ _tag: "Catalog", modelId, outcome: "Updating" }))
      .toBe(`Updating ${modelId}.\nCheck progress: magnitude models status ${modelId}\n`)
    expect(renderPull({ _tag: "Custom", modelId: ADDED_MODEL_ID, outcome: "Installing" }))
      .toBe(`Downloading ${ADDED_MODEL_ID}.\nCheck progress: magnitude models status ${ADDED_MODEL_ID}\n`)
    expect(renderPull({ _tag: "Custom", modelId: ADDED_MODEL_ID, outcome: "DownloadingDrafter" }))
      .toContain(`Downloading the drafter for ${ADDED_MODEL_ID}.`)
    expect(renderPull({ _tag: "Custom", modelId: ADDED_MODEL_ID, outcome: "AlreadyCurrent" })).toBe(`${ADDED_MODEL_ID} is up to date.\n`)
  })

  it("renders import and drafter acknowledgements", () => {
    const modelId = CustomFormModelIdSchema.make(`local:${"b".repeat(64)}`)
    expect(renderImport({ path: "/models/q.gguf", modelId, drafter: Option.none(), outcome: "Started", drafterDownloadStarted: false })).toBe([
      "Added /models/q.gguf. Magnitude uses the file where it is and never moves or deletes it.",
      "Drafter: none",
      `Model ID: ${modelId}`,
      `Load it: magnitude models load ${modelId}`,
      "",
    ].join("\n"))
    expect(renderImport({ path: "/copies/q.gguf", modelId: FOUND_MODEL_ID, drafter: Option.none(), outcome: "AlreadyOnThisComputer", drafterDownloadStarted: false }))
      .toContain(`Already on this computer: /copies/q.gguf has the same bytes as ${FOUND_MODEL_ID}, so it is that model.`)
    expect(renderSetDrafter({ modelId, drafter: Option.none(), drafterDownloadStarted: false })).toBe(`Turned off the drafter for ${modelId}.\n`)
    expect(renderSetDrafter({ modelId, drafter: Option.none(), drafterDownloadStarted: false }, "Auto"))
      .toBe(`No drafter is known to work with ${modelId}, so it runs without one.\n`)
    expect(renderRemoval(modelId, Option.some("forget")))
      .toBe(`Forgot ${modelId}. Magnitude never deletes your file; it stays where it is.\n`)
    expect(renderRemoval(FOUND_MODEL_ID, Option.some("hide")))
      .toContain(`Hid ${FOUND_MODEL_ID} from Magnitude. Its files are untouched.`)
    expect(renderRemoval(ADDED_MODEL_ID, Option.some("delete")))
      .toContain("Magnitude deleted only its own copy")
    expect(renderSetDrafter({ modelId, drafter: Option.some(makeDrafterCandidate()), drafterDownloadStarted: true }))
      .toContain(`Downloading the drafter.\nCheck progress: magnitude models status ${modelId}`)
    const onDisk = renderSetDrafter({ modelId, drafter: Option.some(makeDrafterCandidate({ onThisComputer: false })), drafterDownloadStarted: false })
    expect(onDisk).toBe(`Drafter for ${modelId}: ${makeDrafterCandidate().name}.\n`)
    expect(onDisk).not.toContain("Downloading")
  })

  const pullReference = (input: string) => Effect.runSync(parsePullReference(input))
  const drafterArgument = (input: string) => Effect.runSync(parseDrafterArgument(input, "/work"))

  it("adds a Hugging Face model with the recommended version and auto drafter", async () => {
    const { client, models } = fakeClient({ addHuggingFaceModel: () => Effect.succeed(added("Started")) })
    const result = await run(pull(client, TEST_REPOSITORY, pullReference(TEST_REPOSITORY), drafterArgument("auto")))
    expect(models.addHuggingFaceModel).toHaveBeenCalledWith({
      repository: TEST_REPOSITORY,
      revision: Option.none(),
      version: Option.none(),
      drafter: { _tag: "Auto" },
      force: false,
    })
    expect(result).toMatchObject({ _tag: "Right", right: { _tag: "HuggingFace", versionRequested: false } })
  })

  it("passes --force and words a version the service refuses", async () => {
    const { client, models } = fakeClient({ addHuggingFaceModel: () => Effect.succeed(added("Started")) })
    await run(pull(client, TEST_REPOSITORY, pullReference(TEST_REPOSITORY), drafterArgument("auto"), true))
    expect(models.addHuggingFaceModel).toHaveBeenCalledWith(expect.objectContaining({ force: true }))

    const reference = `${TEST_REPOSITORY}:Q8_0`
    const tooLarge = fakeClient({
      addHuggingFaceModel: () => Effect.fail(new ModelVersionNotRunnable({
        reference, version: "Q8_0", verdict: { _tag: "DoesNotFit", deficitBytes: 6 * 1024 ** 3 },
      })),
    })
    const refused = await run(pull(tooLarge.client, reference, pullReference(reference), drafterArgument("auto")))
    expect(refused._tag === "Left" && (refused.left as { message: string }).message).toBe(
      `Q8_0 needs 6 GB more memory than this computer can provide. Download it anyway with: magnitude catalog pull ${reference} --force`,
    )
    const unsupported = fakeClient({
      addHuggingFaceModel: () => Effect.fail(new ModelVersionNotRunnable({
        reference, version: "Q2_K",
        verdict: { _tag: "Unsupported", reasons: [{ _tag: "WeightTypes", types: [{ ggmlType: "Q2_K", tensorCount: 2 }] }] },
      })),
    })
    const cannot = await run(pull(unsupported.client, reference, pullReference(reference), drafterArgument("auto"), true))
    expect(cannot._tag === "Left" && (cannot.left as { message: string }).message)
      .toBe("Magnitude can't run Q2_K: Uses Q2_K weights, which Magnitude doesn't support yet.")
  })

  it("keeps catalog pull on the catalog sync", async () => {
    const { client, models } = fakeClient({ syncLocalModel: () => Effect.succeed({ outcome: "Installing" }) })
    const result = await run(pull(client, "qwen-test:gguf:q4", pullReference("qwen-test:gguf:q4"), drafterArgument("auto")))
    expect(models.syncLocalModel).toHaveBeenCalledWith({ modelId: "qwen-test:gguf:q4" })
    expect(result).toMatchObject({ _tag: "Right", right: { _tag: "Catalog", outcome: "Installing" } })
  })

  it("resolves a drafter reference against the chosen version before adding", async () => {
    const other = makeDrafterCandidate({
      category: "Other",
      reference: { _tag: "Local", path: "/drafters/mtp.gguf" },
      label: "mtp.gguf",
    })
    const { client, models } = fakeClient({
      inspectModelCandidate: () => Effect.succeed(makeInspection({ selectedVersion: Option.some("Q6_K_P") })),
      listDrafterCandidates: () => Effect.succeed({ candidates: [makeDrafterCandidate(), other], entered: Option.none() }),
      addHuggingFaceModel: () => Effect.succeed(added("Started", Option.some(other))),
    })
    const input = `${TEST_REPOSITORY}:Q6_K_P`
    await run(pull(client, input, pullReference(input), drafterArgument("../drafters/mtp.gguf")))
    expect(models.inspectModelCandidate).toHaveBeenCalledWith({ reference: input })
    expect(models.listDrafterCandidates).toHaveBeenCalledWith({
      target: {
        _tag: "Candidate",
        source: { _tag: "HuggingFace", repository: TEST_REPOSITORY, commit: TEST_COMMIT },
        files: ["Qwen-Test-Q6_K_P.gguf"],
      },
      other: Option.some("/drafters/mtp.gguf"),
    })
    expect(models.addHuggingFaceModel).toHaveBeenCalledWith({
      repository: TEST_REPOSITORY,
      revision: Option.some(TEST_COMMIT),
      version: Option.some("Q6_K_P"),
      drafter: { _tag: "Reference", reference: { _tag: "Local", path: "/drafters/mtp.gguf" } },
      force: false,
    })
  })

  it("resolves the version a Hugging Face file link names", async () => {
    const { client, models } = fakeClient({
      inspectModelCandidate: () => Effect.succeed(makeInspection({ selectedVersion: Option.some("Q4_K_P") })),
      addHuggingFaceModel: () => Effect.succeed(added("Started")),
    })
    const link = `https://huggingface.co/${TEST_REPOSITORY}/blob/main/Qwen-Test-Q4_K_P.gguf`
    const result = await run(pull(client, link, pullReference(link), drafterArgument("none")))
    expect(models.addHuggingFaceModel).toHaveBeenCalledWith({
      repository: TEST_REPOSITORY,
      revision: Option.some(TEST_COMMIT),
      version: Option.some("Q4_K_P"),
      drafter: { _tag: "None" },
      force: false,
    })
    expect(result).toMatchObject({ _tag: "Right", right: { versionRequested: true } })
  })

  it("refuses a drafter the engine rejects", async () => {
    const rejected = makeDrafterCandidate({
      category: "Other",
      verdict: { _tag: "Rejected", reasons: [{ _tag: "TargetMismatch", detail: "x" }], detail: "x" },
    })
    const { client, models } = fakeClient({
      listDrafterCandidates: () => Effect.succeed({ candidates: [rejected], entered: Option.none() }),
      setModelDrafter: () => Effect.succeed({ drafter: Option.none() }),
    })
    const result = await run(setDrafter(client, FOUND_MODEL_ID, drafterArgument("z-lab/Other-DFlash/d.gguf")))
    expect(result).toMatchObject({
      _tag: "Left",
      left: { message: "z-lab/Other-DFlash/d.gguf can't be used as a drafter for this model: Its drafter wasn't built for this model." },
    })
    expect(models.setModelDrafter).not.toHaveBeenCalled()
  })

  // QA (mac-m1, round 3c): the catalog drafter typed by hand as a Hugging Face file was refused
  // ("couldn't find a drafter"), since its listed row is the catalog candidate, not an Other row.
  it("resolves a Hugging Face reference to the listed catalog drafter it is", async () => {
    const catalog = makeDrafterCandidate()
    const { client, models } = fakeClient({
      listDrafterCandidates: () => Effect.succeed({ candidates: [catalog], entered: Option.some(catalog) }),
      setModelDrafter: () => Effect.succeed({ drafter: Option.some(catalog), drafterDownloadStarted: false }),
    })
    const result = await run(setDrafter(client, FOUND_MODEL_ID, drafterArgument("magnitudedev/Qwen-Test-DFlash-GGUF/Qwen-Test-DFlash-Q8_0.gguf")))
    expect(result).toMatchObject({ _tag: "Right" })
    expect(models.setModelDrafter).toHaveBeenCalledWith({
      modelId: FOUND_MODEL_ID,
      drafter: { _tag: "Reference", reference: catalog.reference },
    })
  })

  it("sets a catalog drafter without a lookup", async () => {
    const { client, models } = fakeClient({ setModelDrafter: () => Effect.succeed({ drafter: Option.some(makeDrafterCandidate()) }) })
    await run(setDrafter(client, FOUND_MODEL_ID, drafterArgument("qwen-test:gguf:q4")))
    expect(models.setModelDrafter).toHaveBeenCalledWith({
      modelId: FOUND_MODEL_ID,
      drafter: { _tag: "Reference", reference: { _tag: "Catalog", modelId: "qwen-test:gguf:q4" } },
    })
  })

  it("imports a file with the chosen drafter", async () => {
    const modelId = CustomFormModelIdSchema.make(`local:${"c".repeat(64)}`)
    const { client, models } = fakeClient({ importModelFile: () => Effect.succeed({ modelId, drafter: Option.none(), outcome: "Started" }) })
    const result = await run(importFile(client, "/models/q.gguf", drafterArgument("built-in")))
    expect(models.importModelFile).toHaveBeenCalledWith({
      path: "/models/q.gguf",
      drafter: { _tag: "Reference", reference: { _tag: "Embedded" } },
    })
    expect(result).toMatchObject({ _tag: "Right", right: { path: "/models/q.gguf", modelId } })
  })
})

describe("model status", () => {
  it("keeps acquisition and residency in model status", () => {
    const downloading = makeAcquiringModel({
      _tag: "Installing",
      progress: { stage: "downloading", completedBytes: 50, totalBytes: 100, bytesPerSecond: Option.none() },
    })
    const output = renderModelsStatus(statusView({ models: [downloading, makeInstalledCatalogModel({ modelId: CatalogFormModelIdSchema.make("other:gguf:q8") })] }))
    expect(output).toContain("Downloading 50%")
    expect(output).toContain("Unloaded")
    expect(output).toContain(downloading.modelId)
  })

  it("lists every visible custom model with its source and a footer for what can't run", () => {
    const output = renderModelsStatus(statusView({
      models: [makeCatalogModel(), makeFoundModel(), makeAddedModel(), makeImportedModel()],
      unsupported: [makeUnsupportedLocalModel(), makeUnsupportedLocalModel({ displayName: "Other" })],
      hidden: [makeFoundModel({ modelId: CustomFormModelIdSchema.make("hf:a/b/c.gguf") })],
    }))
    expect(output).toMatch(/MODEL\s+SOURCE\s+VARIANT OF\s+DRAFTER\s+MEMORY\s+CONTEXT\s+STATUS\s+MODEL ID/)
    expect(output).not.toMatch(/qwen-test:gguf:q4$/m)
    expect(output).toContain("Qwen Test (qwen-test:gguf:q4)")
    expect(output).toContain("Hugging Face cache")
    expect(output).toContain("GGUF file")
    expect(output).toContain("Unavailable - The file was moved or deleted")
    expect(output).not.toContain("hf:a/b/c.gguf")
    expect(output.trimEnd().split("\n").slice(-3)).toEqual([
      "2 other models on this computer can't run in Magnitude.",
      "1 hidden model.",
      "List them: magnitude models status --all",
    ])
  })

  it("reports a projector a found catalog target lacks as image input, never as an update, with the command that downloads only it", () => {
    const lacking = makeInstalledCatalogModel({ missingComponents: [{ component: "Projector", downloadBytes: 1_034_945_421 }] })
    const command = `magnitude catalog pull ${lacking.modelId} --only projector`
    const list = renderModelsStatus(statusView({ models: [lacking] }))
    expect(list).not.toContain("Update available")
    expect(list).toContain(`Image input for Qwen Test (Q4) isn't downloaded (1.03 GB): ${command}`)
    const detail = renderModelDetail({ model: lacking, models: [lacking], drafters: { _tag: "NotApplicable" }, hidden: false })
    expect(detail).toMatch(new RegExp(`Image input\\s+Not downloaded \\(1\\.03 GB\\) - ${command}`))
    expect(detail).toMatch(/Installation\s+Installed/)
    expect(detail).not.toMatch(/Update\s+Available/)
    const downloading = makeInstalledCatalogModel({
      missingComponents: lacking.missingComponents,
      acquisitionState: { ...lacking.acquisitionState, _tag: "Optimizing", progress: { stage: "preparing", completed: 0, total: 0, device: Option.none() } } as never,
    })
    expect(renderModelDetail({ model: downloading, models: [], drafters: { _tag: "NotApplicable" }, hidden: false })).not.toContain("Image input")
  })

  it("lists unsupported models with reasons and hidden models under --all", () => {
    const hiddenId = CustomFormModelIdSchema.make("hf:a/b/c.gguf")
    const output = renderModelsStatus(statusView({
      models: [makeFoundModel()],
      unsupported: [makeUnsupportedLocalModel()],
      hidden: [makeFoundModel({ modelId: hiddenId })],
      all: true,
    }))
    expect(output).toMatch(new RegExp(`Hidden\\s+${hiddenId}`))
    expect(output).toMatch(/DeepSeek V4 Flash\s+Ollama\s+No kernels for the deepseek4 architecture/)
    expect(output).toContain("Unhide a model: magnitude models unhide <model-id>")
    expect(output).not.toContain("List them")
  })

  it("lists a hidden catalog model found in an external location under --all", () => {
    const hidden = makeInstalledCatalogModel()
    const output = renderModelsStatus(statusView({ hidden: [hidden], all: true }))
    expect(output).toMatch(new RegExp(`Hidden\\s+${hidden.modelId}`))
    expect(encode(modelsStatusJson, statusView({ hidden: [hidden] }))).toMatchObject({ hidden: [{ modelId: hidden.modelId }] })
  })

  it("renders explicit empty states", () => {
    expect(renderModelsStatus(statusView())).toBe("No local models are on this computer.\n")
    expect(renderModelsStatus(statusView({ unsupported: [makeUnsupportedLocalModel()] })))
      .toContain("1 other model on this computer can't run in Magnitude.")
  })

  it("shows source, variant, drafter, and drafter choices for a custom model", () => {
    const model = makeAddedModel()
    const output = renderModelDetail({
      model,
      models: [makeCatalogModel(), model],
      drafters: { _tag: "Listed", candidates: makeDrafterCandidates() }, hidden: false,
    })
    expect(output).toContain(`Source        Hugging Face · ${TEST_REPOSITORY}`)
    expect(output).toContain("Variant of    Qwen Test (qwen-test:gguf:q4)")
    expect(output).toMatch(/^ {2}Drafter {7}Magnitude DFlash for Qwen Test$/m)
    expect(output).toMatch(/^DRAFTER\s+SIZE\s+VERDICT\s+NOTE\s+USE$/m)
    expect(output).toMatch(/\* Magnitude DFlash for Qwen Test/)
    expect(output).toMatch(/ {2}None\s+—/)
    expect(output).toContain(`Change: magnitude models drafter ${model.modelId} <auto | none | drafter>`)
  })

  it("marks a drafter entered by hand, listed as Other with its verdict, as the one in use", () => {
    const hand = { _tag: "Local", path: "/drafters/hand-dspark.gguf" } as const
    const verdict = { _tag: "Bound", method: { _tag: "DSpark" }, memoryBytes: 1_073_741_824 } as const
    const model = makeAddedModel({
      drafterChoice: { _tag: "Reference", reference: hand },
      drafter: Option.some({
        reference: hand, name: "hand-dspark.gguf", label: "Untested", sizeBytes: 5_000_000, verdict,
        availability: { _tag: "Present" }, updateAvailable: false,
      }),
    })
    const candidates = [
      ...makeDrafterCandidates(),
      makeDrafterCandidate({
        category: "Other", reference: hand, name: "hand-dspark.gguf", label: "Untested", sizeBytes: 5_000_000,
        lineage: "None", verdict, onThisComputer: true, method: Option.some({ _tag: "DSpark" }),
      }),
    ]
    const output = renderModelDetail({ model, models: [model], drafters: { _tag: "Listed", candidates }, hidden: false })
    expect(output).toMatch(/^\* hand-dspark\.gguf\s+5 MB, no download\s+Works · DSpark\s+.*\/drafters\/hand-dspark\.gguf$/m)
    expect(output).toMatch(/^ {2}Magnitude DFlash for Qwen Test\s/m)
  })

  it("offers a recommended drafter and survives unavailable drafter choices", () => {
    const output = renderModelDetail({
      model: makeFoundModel(),
      models: [],
      drafters: { _tag: "Unavailable", message: "Hugging Face is unreachable" }, hidden: false,
    })
    expect(output).toContain("Drafter     None · Faster with a drafter · Add (900 MB)")
    expect(output).toContain("Drafter choices are unavailable: Hugging Face is unreachable")
    expect(output).toContain("Files       On this computer, managed outside Magnitude")
  })

  it("marks an added model that can't run", () => {
    const unsupported = makeAddedModel({
      servingState: {
        _tag: "Assessed",
        metadata: (makeAddedModel().servingState as Extract<CustomLocalModel["servingState"], { _tag: "Assessed" }>).metadata,
        capabilities: (makeAddedModel().servingState as Extract<CustomLocalModel["servingState"], { _tag: "Assessed" }>).capabilities,
        speculativeMethod: Option.none(),
        assessment: unsupportedAssessment("IQ1_M"),
      },
    })
    const output = renderModelDetail({ model: unsupported, models: [], drafters: { _tag: "NotApplicable" }, hidden: false })
    expect(output).toContain("Can't run     Uses IQ1_M weights, which Magnitude doesn't support yet")
    expect(renderModelsStatus(statusView({ models: [unsupported] }))).toContain("Can't run - Uses IQ1_M weights")
  })

  it("lists the drafter candidates for an addressed custom model", async () => {
    const model = makeFoundModel()
    const { client, models } = fakeClient({
      getCatalog: () => Effect.succeed(catalogState([model], [makeUnsupportedLocalModel()])),
      listDrafterCandidates: () => Effect.succeed({ candidates: makeDrafterCandidates(), entered: Option.none() }),
    })
    const detail = await run(readModelsStatus(client, Option.some(model.modelId), false))
    expect(models.listDrafterCandidates).toHaveBeenCalledWith({ target: { _tag: "Model", modelId: model.modelId }, other: Option.none() })
    expect(detail._tag === "Right" && renderModelsStatusResult(detail.right)).toContain("Built into this model")
    const list = await run(readModelsStatus(client, Option.none(), true))
    expect(list._tag === "Right" && renderModelsStatusResult(list.right)).toContain("No kernels for the deepseek4 architecture")
  })

  it("does not list drafter candidates for catalog models and reports unknown IDs", async () => {
    const catalog = makeInstalledCatalogModel()
    const { client, models } = fakeClient({ getCatalog: () => Effect.succeed(catalogState([catalog])) })
    const detail = await run(readModelsStatus(client, Option.some(catalog.modelId), false))
    expect(models.listDrafterCandidates).toBeUndefined()
    expect(detail._tag === "Right" && renderModelsStatusResult(detail.right)).toContain("Source        Catalog")
    expect(await run(readModelsStatus(client, Option.some(FOUND_MODEL_ID), false)))
      .toMatchObject({ _tag: "Left", left: { message: `Unknown model: ${FOUND_MODEL_ID}` } })
  })
})

describe("model locations", () => {
  it("parses automatic location arguments", () => {
    expect(parseAutomaticLocation("lm-studio")).toEqual(Option.some("LmStudio"))
    expect(parseAutomaticLocation(" Ollama ")).toEqual(Option.some("Ollama"))
    expect(parseAutomaticLocation("huggingface-cache")).toEqual(Option.some("HuggingFaceCache"))
    expect(parseAutomaticLocation("llama-cpp")).toEqual(Option.some("LlamaCpp"))
    expect(parseAutomaticLocation("folder")).toEqual(Option.none())
  })

  it("lists each automatic location's directories with their state, then your folders", () => {
    expect(renderModelLocations(makeLocations())).toBe([
      "Automatic locations",
      "",
      "LOCATION            STATUS     PATH",
      "Hugging Face cache  Found      /Volumes/hf/hub",
      "Hugging Face cache  Found      /Users/test/.cache/huggingface/hub",
      "LM Studio           Not found  /Users/test/.lmstudio/models",
      "LM Studio           Not found  /Users/test/.cache/lm-studio/models",
      "Ollama              Off        /Users/test/.ollama/models",
      "",
      "Your folders",
      "",
      "PATH                      STATUS",
      "/Volumes/External/models  Found",
      "",
    ].join("\n"))
    expect(renderModelLocations(makeLocations().slice(0, 1))).toContain("No folders added. Add one: magnitude models folders add <path>")
  })

  it("says which locations it can't read and how to grant access", () => {
    const rendered = renderModelLocations([
      { kind: "Ollama", paths: [
        { path: "/home/ada/.ollama/models", state: "Found" },
        { path: "/usr/share/ollama/.ollama/models", state: "Unreadable" },
        { path: "/var/lib/ollama", state: "NotFound" },
      ], enabled: true, automatic: true, state: "Found" },
      { kind: "Folder", paths: [{ path: "/srv/models", state: "Unreadable" }], enabled: true, automatic: false, state: "Unreadable" },
    ])
    expect(rendered).toContain("Ollama    Found       /home/ada/.ollama/models")
    expect(rendered).toContain("Ollama    Can't read  /usr/share/ollama/.ollama/models")
    expect(rendered).not.toContain("/var/lib/ollama")
    expect(rendered).toContain("Can't read /usr/share/ollama/.ollama/models. Magnitude doesn’t have permission to read this folder. A system Ollama service keeps its models in a folder that belongs to the ollama user: give your account read access to it, or set OLLAMA_MODELS to a folder you can read (Magnitude also scans the default folders).")
    expect(rendered).toContain("/srv/models  Can't read")
    expect(rendered).toContain("Can't read /srv/models. Magnitude doesn’t have permission to read this folder. Give your account read access to it.")
    expect(renderModelLocations([{ kind: "Ollama", paths: [{ path: "/o", state: "Unreadable" }], enabled: false, automatic: true, state: "Unreadable" }]))
      .not.toContain("permission")
  })
})

const encode = <A>(document: { readonly encode: (value: A) => Effect.Effect<unknown, unknown> }, value: A) =>
  JSON.parse(JSON.stringify(Effect.runSync(document.encode(value)))) as Record<string, unknown>

describe("--json output", () => {

  it("encodes Option fields as present or absent keys", () => {
    const search = encode(searchJson, { query: "q", all: false, response: makeSearchResponse(), models: [] })
    const results = search.results as Record<string, unknown>[]
    expect(results[0]).toMatchObject({ repository: TEST_REPOSITORY, variantOf: "qwen-test:gguf:q4", downloads: 48_213 })
    expect(results[1]).not.toHaveProperty("variantOf")
    expect(results[1]).not.toHaveProperty("unsupported")
    expect(JSON.stringify(search)).not.toContain("\"_id\":\"Option\"")
  })

  it("includes structured unsupported reasons and drafter verdicts", () => {
    const document = encode(inspectionJson, { inspection: makeInspection(), drafters: makeDrafterCandidates(), models: [] })
    const versions = (document.inspection as { versions: Record<string, unknown>[] }).versions
    expect(versions[3]!.assessment).toMatchObject({
      _tag: "Unsupported",
      reasons: [{ _tag: "WeightTypes", types: [{ ggmlType: "Q2_K", tensorCount: 2 }] }],
      detail: "source_element has no Q2_K case",
    })
    expect(versions[0]).not.toHaveProperty("onThisComputer")
    expect((document.drafters as Record<string, unknown>[])[0]!.verdict).toMatchObject({ _tag: "Rejected" })
  })

  it("renders and encodes an unreadable version, which has no serving profile", () => {
    const [, , recommended] = makeInspection().versions
    const inspection = makeInspection({
      versions: [recommended!, { ...recommended!, label: "Q3_K_M", recommended: false, assessment: Option.some(unreadableAssessment()) }],
    })
    const output = renderInspection({ inspection, drafters: [], models: [] })
    expect(output).toMatch(/Q3_K_M.*Unsupported: Magnitude can't read this file/)
    const document = encode(inspectionJson, { inspection, drafters: [], models: [] })
    const unreadable = (document.inspection as { versions: Record<string, unknown>[] }).versions[1]!.assessment
    expect(unreadable).toMatchObject({ _tag: "Unsupported", reasons: [{ _tag: "Unreadable" }] })
    expect(unreadable).not.toHaveProperty("profile")
  })

  it("encodes status, detail, pull, catalog, and hardware documents", () => {
    const status = encode(modelsStatusJson, statusView({ models: [makeFoundModel(), makeCatalogModel()], unsupported: [makeUnsupportedLocalModel()] }))
    expect(status.initializing).toBe(false)
    expect((status.models as Record<string, unknown>[]).map(({ modelId }) => modelId)).toEqual([FOUND_MODEL_ID])
    expect((status.unsupported as Record<string, unknown>[])[0]).toMatchObject({ reasons: [{ _tag: "Architecture", architecture: "deepseek4" }] })

    const detail = encode(modelDetailJson, { model: makeAddedModel(), models: [], drafters: { _tag: "Listed", candidates: makeDrafterCandidates() }, hidden: false })
    expect(detail.drafterCandidates).toHaveLength(3)
    expect(detail).not.toHaveProperty("drafterCandidatesUnavailable")

    const pull = encode(pullJson, {
      _tag: "HuggingFace",
      repository: TEST_REPOSITORY,
      versionRequested: false,
      added: { _tag: "Custom", modelId: ADDED_MODEL_ID, version: "Q4_K_P", drafter: Option.none(), outcome: "Started", drafterDownloadStarted: false },
    })
    expect(pull).toEqual({
      _tag: "HuggingFace",
      repository: TEST_REPOSITORY,
      versionRequested: false,
      added: { _tag: "Custom", modelId: ADDED_MODEL_ID, version: "Q4_K_P", outcome: "Started", drafterDownloadStarted: false },
    })

    expect(encode(catalogStatusJson, { _tag: "Initializing" })).toEqual({})
    const hardware = encode(hardwareJson, { hardware: makeHardware(), models: [makeFoundModel()] })
    expect(hardware).not.toHaveProperty("currentModel")
    expect((hardware.hardware as Record<string, unknown>).processor).toBe("Test CPU")
  })

  it("prints one JSON document in place of the text output", async () => {
    const { client } = fakeClient({
      searchHuggingFaceModels: () => Effect.succeed(makeSearchResponse()),
      getCatalog: () => Effect.succeed(catalogState()),
    })
    const effect = readSearch(client, "qwen", Option.some(5))
    const json = await capture(() => runCommand({ effect, render: renderSearch, json: asJson({ json: true }, searchJson) }))
    expect(json.stderr).toBe("")
    expect(JSON.parse(json.stdout).results).toHaveLength(2)
    expect(json.stdout.endsWith("}\n")).toBe(true)
    const text = await capture(() => runCommand({ effect, render: renderSearch, json: asJson({}, searchJson) }))
    expect(text.stdout).toContain("Hugging Face models matching \"qwen\"")
  })

  it("encodes models status results for every form", () => {
    expect(encode(modelsStatusResultJson, { _tag: "Initializing" })).toEqual({ initializing: true, models: [], unsupported: [], hidden: [] })
    expect(encode(modelsStatusResultJson, { _tag: "List", view: statusView({ models: [makeFoundModel()] }) }))
      .toMatchObject({ initializing: false, models: [{ _tag: "Custom", modelId: FOUND_MODEL_ID }] })
    expect(encode(modelsStatusResultJson, { _tag: "Detail", view: { model: makeFoundModel(), models: [], drafters: { _tag: "Unavailable", message: "offline" }, hidden: false } }))
      .toMatchObject({ model: { modelId: FOUND_MODEL_ID }, drafterCandidatesUnavailable: "offline" })
  })

  it("encodes model locations and acknowledgements", () => {
    const locations = encode(jsonDocument(ModelLocationsJsonSchema, (value: ReturnType<typeof makeLocations>) => ({ locations: value })), makeLocations())
    expect(locations.locations).toHaveLength(4)
    expect(encode(jsonDocument(ImportResultSchema, (value: ImportResult) => value), {
      path: "/m/q.gguf",
      modelId: CustomFormModelIdSchema.make(`local:${"d".repeat(64)}`),
      drafter: Option.none(),
      outcome: "Started",
      drafterDownloadStarted: false,
    })).toEqual({ path: "/m/q.gguf", modelId: `local:${"d".repeat(64)}`, outcome: "Started", drafterDownloadStarted: false })
  })

  it.each([
    [() => showModelsStatus("bad", { json: true }), "Invalid model ID: bad"],
    [() => removeModel("nonsense", { json: true }), "Invalid model ID: nonsense"],
    [() => loadInstance("local:not-a-digest", { json: true }), "Invalid model ID: local:not-a-digest"],
    [() => searchHuggingFace("  ", { json: true }), "Search query must not be empty"],
    [() => hideModel("nope", { json: true }), "Invalid model ID: nope"],
  ])("prints one JSON error document on stderr in JSON mode (%#)", async (command, message) => {
    const result = await capture(command)
    expect(result.stdout).toBe("")
    expect(JSON.parse(result.stderr)).toEqual({ error: { message, code: "model_command_error" } })
    expect(result.exitCode).toBe(1)
  })

  it("says plainly that a folder to add doesn't exist, in text and in JSON", async () => {
    const missing = new DirectoryNotFound({ path: "/home/qa/nonexistent" })
    const text = await capture(() => runCommand({ effect: Effect.fail(missing), render: () => "", json: asJson({}, searchJson) }))
    expect(text.stderr).toBe("/home/qa/nonexistent doesn't exist.\n")
    const json = await capture(() => runCommand({ effect: Effect.fail(missing), render: () => "", json: asJson({ json: true }, searchJson) }))
    expect(JSON.parse(json.stderr)).toEqual({ error: { message: "/home/qa/nonexistent doesn't exist.", code: "directory_not_found" } })
  })

  it("carries the service's own failure code in the JSON error document", async () => {
    const failure = { _tag: "LocalModelMutationFailed", code: "managed_model_not_hideable", message: "Remove it instead.", retryable: false }
    const result = await capture(() => runCommand({ effect: Effect.fail(failure), render: () => "", json: asJson({ json: true }, searchJson) }))
    expect(JSON.parse(result.stderr)).toEqual({ error: { message: "Remove it instead.", code: "managed_model_not_hideable" } })
    const text = await capture(() => runCommand({ effect: Effect.fail(failure), render: () => "", json: asJson({}, searchJson) }))
    expect(text.stderr).toBe("Remove it instead.\n")
  })
})

describe("upgraded model IDs", () => {
  it("finds the catalog model a request names by the ID its files had as a found model", async () => {
    const alias = CustomFormModelIdSchema.make("hf:unsloth/Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-UD-Q4_K_XL.gguf")
    const base = makeCatalogModel()
    const catalog = makeCatalogModel({ catalogData: { ...base.catalogData, aliases: [alias] } })
    const found = await run(findModel([catalog], alias))
    expect(found._tag === "Right" && found.right.modelId).toBe(catalog.modelId)
  })
})

describe("custom model license", () => {
  it("shows a custom model's license in its details when it is known", () => {
    const base = makeFoundModel()
    const licensed = makeFoundModel({ presentation: { ...base.presentation, license: Option.some("apache-2.0") } })
    expect(renderModelDetail({ model: licensed, models: [], drafters: { _tag: "NotApplicable" }, hidden: false })).toMatch(/License\s+apache-2\.0/)
    expect(renderModelDetail({ model: makeFoundModel({ presentation: { ...base.presentation, license: Option.none() } }), models: [], drafters: { _tag: "NotApplicable" }, hidden: false }))
      .not.toContain("License")
  })
})

describe("hidden models", () => {
  it("counts hidden models from the service in the footer and lists them marked hidden under --all", async () => {
    const hiddenId = CustomFormModelIdSchema.make("hf:a/b/c.gguf")
    const hidden = makeFoundModel({ modelId: hiddenId })
    const { client } = fakeClient({ getCatalog: () => Effect.succeed(catalogState([makeAddedModel()], [], [hidden])) })
    const list = await run(readModelsStatus(client, Option.none(), false))
    const listed = list._tag === "Right" ? renderModelsStatusResult(list.right) : ""
    expect(listed).toContain("1 hidden model.")
    expect(listed).not.toContain(hiddenId)
    const all = await run(readModelsStatus(client, Option.none(), true))
    expect(all._tag === "Right" && renderModelsStatusResult(all.right)).toMatch(new RegExp(`Hidden\\s+${hiddenId}`))
    expect(all._tag === "Right" && encode(modelsStatusResultJson, all.right)).toMatchObject({ hidden: [{ modelId: hiddenId }] })
  })
})

describe("drafter updates", () => {
  it("shows Update when the attached catalog drafter has a newer revision", () => {
    const base = makeAddedModel()
    const drafter = Option.getOrThrow(base.drafter)
    const found = makeFoundModel({ recommendedDrafter: Option.none(), drafter: Option.some({ ...drafter, updateAvailable: true }) })
    expect(renderModelsStatus(statusView({ models: [found] }))).toContain("Update available")
    expect(renderModelDetail({ model: found, models: [], drafters: { _tag: "NotApplicable" }, hidden: false }))
      .toContain(`Update      A newer drafter is available - magnitude catalog pull ${FOUND_MODEL_ID}`)
    const downloading = makeFoundModel({ drafter: Option.some({ ...drafter, updateAvailable: true, availability: { _tag: "Downloading", progress: { stage: "downloading", completedBytes: 1, totalBytes: 4, bytesPerSecond: Option.none() } } }) })
    expect(renderModelsStatus(statusView({ models: [downloading] }))).not.toContain("Update available")
    expect(renderModelsStatus(statusView({ models: [base] }))).not.toContain("Update available")
  })

  it("syncs a custom model when catalog pull names its ID", async () => {
    expect(Effect.runSync(parsePullReference(FOUND_MODEL_ID))).toEqual({ _tag: "Custom", modelId: FOUND_MODEL_ID })
    const { client, models } = fakeClient({ syncLocalModel: () => Effect.succeed({ outcome: "Updating" }) })
    const result = await run(pull(client, FOUND_MODEL_ID, Effect.runSync(parsePullReference(FOUND_MODEL_ID)), { _tag: "Auto" }))
    expect(models.syncLocalModel).toHaveBeenCalledWith({ modelId: FOUND_MODEL_ID })
    expect(result).toMatchObject({ _tag: "Right", right: { _tag: "Custom", modelId: FOUND_MODEL_ID, outcome: "Updating" } })
    expect(renderPull({ _tag: "Custom", modelId: FOUND_MODEL_ID, outcome: "Updating" }))
      .toBe(`Updating ${FOUND_MODEL_ID}.\nCheck progress: magnitude models status ${FOUND_MODEL_ID}\n`)
    expect(renderPull({ _tag: "Custom", modelId: FOUND_MODEL_ID, outcome: "AlreadyCurrent" })).toBe(`${FOUND_MODEL_ID} is up to date.\n`)
    expect(encode(pullJson, { _tag: "Custom", modelId: FOUND_MODEL_ID, outcome: "AlreadyCurrent" }))
      .toEqual({ _tag: "Custom", modelId: FOUND_MODEL_ID, outcome: "AlreadyCurrent" })
  })

  it("points --drafter on a custom model ID at models drafter", async () => {
    const result = await capture(() => pullModel(FOUND_MODEL_ID, { drafter: "none" }))
    expect(result.stderr).toBe(`Change this model's drafter with: magnitude models drafter ${FOUND_MODEL_ID} <auto | none | drafter>\n`)
    expect(result.exitCode).toBe(1)
  })
})

describe("--json internal identities", () => {
  const internal = /"(assessmentId|environmentId|rankingScores|fidelityRank|acceleratorId|memoryDomainId|deviceId|primaryPath|ownership|limitingResource)"/
  const loading = (model: CustomLocalModel): CustomLocalModel => model.acquisitionState._tag === "External"
    ? { ...model, acquisitionState: { ...model.acquisitionState, residencyState: { _tag: "Requested" } } }
    : model

  it("never includes internal identities, managed paths, ownership, or ranking inputs", () => {
    const assessed = makeAddedModel().servingState as Extract<CustomLocalModel["servingState"], { _tag: "Assessed" }>
    const unrunnable = makeImportedModel({ servingState: { ...assessed, assessment: unsupportedAssessment("IQ1_M") } })
    const catalog = catalogState([makeCatalogModel(), makeFoundModel(), makeAddedModel()], [makeUnsupportedLocalModel()], [unrunnable])
    const documents = [
      encode(catalogListJson, catalog),
      encode(catalogDetailJson, { model: makeCatalogModel(), models: [] }),
      encode(recommendationsJson, { catalog, models: [makeCatalogModel()], hardware: makeHardware(), preference: { label: "Balanced" }, ranked: [makeCatalogModel()] }),
      encode(modelsStatusJson, statusView({ models: [makeCatalogModel(), makeFoundModel(), makeAddedModel(), unrunnable], hidden: [makeFoundModel({ modelId: CustomFormModelIdSchema.make("hf:a/b/c.gguf") })] })),
      encode(modelDetailJson, { model: makeAddedModel(), models: [], drafters: { _tag: "Listed", candidates: makeDrafterCandidates() }, hidden: false }),
      encode(modelDetailJson, { model: makeInstalledCatalogModel(), models: [], drafters: { _tag: "NotApplicable" }, hidden: false }),
      encode(inspectionJson, { inspection: makeInspection(), drafters: makeDrafterCandidates(), models: [] }),
      encode(hardwareJson, { hardware: makeHardware(), models: [loading(makeFoundModel())] }),
    ]
    for (const document of documents) expect(JSON.stringify(document)).not.toMatch(internal)
    expect(JSON.stringify(makeFoundModel())).toMatch(internal)
  })

  it("keeps the product facts of each assessment", () => {
    const status = encode(modelsStatusJson, statusView({ models: [makeFoundModel()] }))
    expect((status.models as { servingState: { assessment: Record<string, unknown> } }[])[0]!.servingState.assessment)
      .toMatchObject({ _tag: "Fits", profile: { contextLength: expect.any(Number) }, memory: { totalRequiredBytes: expect.any(Number) } })
    const hardware = encode(hardwareJson, { hardware: makeHardware(), models: [loading(makeFoundModel())] })
    expect(hardware.currentModel).toMatchObject({ _tag: "Custom", modelId: FOUND_MODEL_ID, servingState: { _tag: "Assessed" } })
    const inspection = encode(inspectionJson, { inspection: makeInspection(), drafters: [], models: [] })
    expect((inspection.inspection as { versions: { assessment?: Record<string, unknown> }[] }).versions[3]!.assessment)
      .toMatchObject({ _tag: "Unsupported", reasons: [{ _tag: "WeightTypes" }] })
  })
})
