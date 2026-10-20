import { Effect, Option } from "effect"
import {
  AssessmentEnvironmentIdSchema,
  CatalogFormModelIdSchema,
  ModelAssessmentIdSchema,
  CustomFormModelIdSchema,
  LocalInferenceDeviceIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  type CatalogLocalModel,
  type CustomLocalModel,
  type ModelCatalogState,
} from "@magnitudedev/sdk"
import { afterEach, describe, expect, it, vi } from "vitest"
import {
  ADDED_MODEL_ID,
  CATALOG_DRAFTER_ID,
  catalogState,
  FOUND_MODEL_ID,
  IMPORTED_MODEL_ID,
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
  TEST_COMMIT,
  TEST_REPOSITORY,
} from "./fixtures/inference"
import { CliModelsConnection, withClient, type CliModelsClient } from "./inference-client"
import { parseDrafterArgument, parsePullReference } from "./model-references"
import { asJson, commandEffect } from "./output"
import {
  addModelFolder,
  cancelDownload,
  catalogDetailJson,
  catalogListJson,
  hardwareJson,
  hideModel,
  inspectionJson,
  listModelLocations,
  loadInstance,
  modelDetailJson,
  modelsStatusJson,
  pull,
  pullJson,
  readInspection,
  renderInspection,
  renderPull,
  recommendationsJson,
  removeModel,
  removeModelFolder,
  renderModelDetail,
  renderModelsStatus,
  searchJson,
  setModelDrafter,
  setModelLocationEnabled,
  showCatalogModel,
  showModelsStatus,
  stopInstance,
  unhideModel,
  type ModelsStatusView,
} from "./inference-runtime"

type Handler = (payload: never) => Effect.Effect<unknown, unknown>

const fakeService = (handlers: Record<string, Handler>) => {
  const models = Object.fromEntries(Object.entries(handlers).map(([name, handler]) => [name, vi.fn(handler)]))
  return { client: { models } as unknown as CliModelsClient, models }
}

/** Runs a command against a fake service, capturing what it prints. */
const capture = async (command: Effect.Effect<void>, client: CliModelsClient) => {
  let stdout = ""
  let stderr = ""
  process.exitCode = 0
  const out = vi.spyOn(process.stdout, "write").mockImplementation((chunk) => { stdout += String(chunk); return true })
  const err = vi.spyOn(process.stderr, "write").mockImplementation((chunk) => { stderr += String(chunk); return true })
  await Effect.runPromise(command.pipe(Effect.provideService(CliModelsConnection, { connect: Effect.succeed(client) })))
  out.mockRestore()
  err.mockRestore()
  const exitCode = process.exitCode
  process.exitCode = 0
  return { stdout, stderr, exitCode }
}

/** Runs a command in text and in `--json`, returning both outputs and every RPC payload it sent. */
const runBoth = async <O extends { readonly json?: boolean }>(
  handlers: Record<string, Handler>,
  command: (options: O) => Effect.Effect<void>,
  options = {} as O,
) => {
  const textService = fakeService(handlers)
  const text = await capture(command(options), textService.client)
  const textModels = textService.models
  const jsonService = fakeService(handlers)
  const json = await capture(command({ ...options, json: true }), jsonService.client)
  const jsonModels = jsonService.models
  expect(text.stderr).toBe("")
  expect(json.stderr).toBe("")
  expect(text.exitCode).toBe(0)
  expect(json.exitCode).toBe(0)
  return { text: text.stdout, json: JSON.parse(json.stdout) as Record<string, unknown>, models: textModels, jsonModels }
}

const succeed = (value: unknown = {}): Handler => () => Effect.succeed(value)

afterEach(() => {
  vi.restoreAllMocks()
  process.exitCode = 0
})

const MODEL_IDS = [
  ["hf:owner/repo/selector", ADDED_MODEL_ID],
  ["local:<digest>", IMPORTED_MODEL_ID],
] as const

describe("model mutations accept hf: and local: IDs", () => {
  it.each(MODEL_IDS)("catalog cancel with a %s ID", async (_, modelId) => {
    const { text, json, models } = await runBoth({ cancelLocalModelSync: succeed() }, (options) => cancelDownload(modelId, options))
    expect(models.cancelLocalModelSync).toHaveBeenCalledWith({ modelId })
    expect(text).toBe(`Cancelled work for ${modelId}.\n`)
    expect(json).toEqual({ modelId })
  })

  it.each([
    ["hf:owner/repo/selector", makeAddedModel(), `Removed ${ADDED_MODEL_ID}.\nMagnitude deleted only its own copy; files other apps manage are untouched.\n`],
    ["local:<digest>", makeImportedModel(), `Forgot ${IMPORTED_MODEL_ID}. Magnitude never deletes your file; it stays where it is.\n`],
  ] as const)("catalog remove with a %s ID", async (_, model, expected) => {
    const { text, json, models } = await runBoth({
      getCatalog: succeed(catalogState([model])),
      removeLocalModel: succeed({ keptComponents: [] }),
    }, (options) => removeModel(model.modelId, options))
    expect(models.removeLocalModel).toHaveBeenCalledWith({ modelId: model.modelId })
    expect(text).toBe(expected)
    expect(json).toEqual({ modelId: model.modelId })
  })

  it.each(MODEL_IDS)("models load with a %s ID", async (_, modelId) => {
    const { text, json, models } = await runBoth({ load: succeed() }, (options) => loadInstance(modelId, options))
    expect(models.load).toHaveBeenCalledWith({ modelId })
    expect(text).toBe(`Load requested for ${modelId}.\nCheck progress: magnitude models status ${modelId}\n`)
    expect(json).toEqual({ modelId })
  })

  it("models stop with no ID stops the active model", async () => {
    const { text, json, models } = await runBoth({ stop: succeed() }, (options) => stopInstance(undefined, options))
    expect(models.stop).toHaveBeenCalledWith({})
    expect(text).toBe("Stopped the active local model.\n")
    expect(json).toEqual({})
  })

  it.each([
    ["catalog", CATALOG_DRAFTER_ID],
    ["hf:", ADDED_MODEL_ID],
    ["local:", IMPORTED_MODEL_ID],
    ["alias", FOUND_MODEL_ID],
  ] as const)("models stop with a %s ID", async (_, modelId) => {
    const { text, json, models } = await runBoth({ stopModel: succeed({ stopped: true }), stop: succeed() }, (options) => stopInstance(` ${modelId} `, options))
    expect(models.stopModel).toHaveBeenCalledWith({ modelId })
    expect(models.stop).not.toHaveBeenCalled()
    expect(text).toBe(`Stopped ${modelId}.\n`)
    expect(json).toEqual({ modelId, stopped: true })
  })

  // QA linux-cpu rounds 3c and 4c: stopping a model that wasn't running said "Stopped".
  it("models stop of a model that isn't running says so and succeeds", async () => {
    const { text, json } = await runBoth({ stopModel: succeed({ stopped: false }) }, (options) => stopInstance(ADDED_MODEL_ID, options))
    expect(text).toBe(`${ADDED_MODEL_ID} wasn't running.\n`)
    expect(json).toEqual({ modelId: ADDED_MODEL_ID, stopped: false })
  })

  it("catalog remove of a found catalog model deletes only the parts Magnitude stores, and hides it without any", async () => {
    const weightsPath = "/home/ada/.cache/huggingface/hub/models--unsloth--Qwen3.5-4B-MTP-GGUF/snapshots/c/Qwen3.5-4B-UD-Q4_K_XL.gguf"
    const found = (storedComponents: readonly ("Projector" | "Drafter")[]) => makeInstalledCatalogModel({
      acquisitionState: {
        _tag: "Installed",
        installation: { _tag: "Resolved", installedBytes: 1, primaryPath: weightsPath, ownership: "External" },
        residencyState: { _tag: "Unloaded" },
      },
      storedComponents,
    })
    const withParts = found(["Projector", "Drafter"])
    const { text, json, models } = await runBoth({
      getCatalog: succeed(catalogState([withParts])),
      removeLocalModel: succeed({ keptComponents: [] }),
    }, (options) => removeModel(withParts.modelId, options))
    expect(models.removeLocalModel).toHaveBeenCalledWith({ modelId: withParts.modelId })
    expect(text).toBe([
      `Removed the image input and drafter Magnitude downloaded for ${withParts.modelId}.`,
      `It still runs from ${weightsPath}, which Magnitude never deletes.`,
      `To stop listing it: magnitude models hide ${withParts.modelId}`,
      "",
    ].join("\n"))
    expect(json).toEqual({ modelId: withParts.modelId })

    // With nothing of Magnitude's, Remove hides the found copy, as it hides any found model, and says so
    // (QA linux-arch round 3c: it refused here while a found Hugging Face cache model was hidden).
    const hidden = await runBoth({
      getCatalog: succeed(catalogState([found([])])),
      removeLocalModel: succeed({ keptComponents: [] }),
    }, (options) => removeModel(withParts.modelId, options))
    expect(hidden.text).toBe([
      `Hid ${withParts.modelId} from Magnitude. Its files are untouched.`,
      `Unhide it: magnitude models unhide ${withParts.modelId}`,
      "",
    ].join("\n"))

    // A drafter still queued is Magnitude's own part too: Remove cancels it rather than hiding the model
    // (QA mac round 4: it hid the model, which then looked like a catalog download).
    const fetching = makeInstalledCatalogModel({
      acquisitionState: {
        _tag: "Updating",
        installation: { _tag: "Resolved", installedBytes: 1, primaryPath: weightsPath, ownership: "External" },
        residencyState: { _tag: "Unloaded" },
        progress: { stage: "queued", completedBytes: 0, totalBytes: 685_000_000, bytesPerSecond: Option.none() },
      },
      storedComponents: [],
    })
    const cancelled = await runBoth({
      getCatalog: succeed(catalogState([fetching])),
      removeLocalModel: succeed({ keptComponents: [] }),
    }, (options) => removeModel(fetching.modelId, options))
    expect(cancelled.text).toBe([
      `Cancelled the download Magnitude had in progress for ${fetching.modelId}.`,
      `It still runs from ${weightsPath}, which Magnitude never deletes.`,
      `To stop listing it: magnitude models hide ${fetching.modelId}`,
      "",
    ].join("\n"))

    // A drafter another model serves with stays, and the removal says so rather than failing
    // (QA linux-cuda round 4e: "uses files another model on this computer also uses" and "Removal failed").
    const shared = await runBoth({
      getCatalog: succeed(catalogState([withParts])),
      removeLocalModel: succeed({ keptComponents: ["Drafter"] }),
    }, (options) => removeModel(withParts.modelId, options))
    expect(shared.text).toBe([
      `Removed the image input Magnitude downloaded for ${withParts.modelId}.`,
      `Kept the drafter Magnitude downloaded for ${withParts.modelId}: another model on this computer uses it.`,
      `It still runs from ${weightsPath}, which Magnitude never deletes.`,
      `To stop listing it: magnitude models hide ${withParts.modelId}`,
      "",
    ].join("\n"))
    expect(shared.json).toEqual({ modelId: withParts.modelId })
  })

  it("models hide fails with the service's error for an ID that isn't on this computer", async () => {
    const message = `${IMPORTED_MODEL_ID} is not a model on this computer.`
    const { client } = fakeService({
      hideModel: () => Effect.fail({ _tag: "LocalModelMutationFailed", code: "custom_model_not_found", message, retryable: false }),
    })
    const result = await capture(hideModel(IMPORTED_MODEL_ID), client)
    expect(result.stdout).toBe("")
    expect(result.stderr).toContain(message)
    expect(result.exitCode).toBe(1)
  })

  it.each(MODEL_IDS)("models hide with a %s ID", async (_, modelId) => {
    const { text, json, models } = await runBoth({ hideModel: succeed() }, (options) => hideModel(modelId, options))
    expect(models.hideModel).toHaveBeenCalledWith({ modelId })
    expect(text).toBe(`Hid ${modelId} from Magnitude. Its files are untouched.\nUnhide it: magnitude models unhide ${modelId}\n`)
    expect(json).toEqual({ modelId })
  })

  it.each(MODEL_IDS)("models unhide with a %s ID", async (_, modelId) => {
    const { text, json, models } = await runBoth({ unhideModel: succeed() }, (options) => unhideModel(modelId, options))
    expect(models.unhideModel).toHaveBeenCalledWith({ modelId })
    expect(text).toBe(`${modelId} is visible in Magnitude again.\n`)
    expect(json).toEqual({ modelId })
  })

  it.each(MODEL_IDS)("models drafter none with a %s ID", async (_, modelId) => {
    const { text, json, models } = await runBoth({
      setModelDrafter: succeed({ drafter: Option.none(), drafterDownloadStarted: false }),
    }, (options) => setModelDrafter(modelId, "none", options))
    expect(models.setModelDrafter).toHaveBeenCalledWith({ modelId, drafter: { _tag: "None" } })
    expect(text).toBe(`Turned off the drafter for ${modelId}.\n`)
    expect(json).toEqual({ modelId, drafterDownloadStarted: false })
  })

  it.each(MODEL_IDS)("models drafter auto with a %s ID downloads the preselected drafter", async (_, modelId) => {
    const drafter = makeDrafterCandidate()
    const { text, json, models } = await runBoth({
      setModelDrafter: succeed({ drafter: Option.some(drafter), drafterDownloadStarted: true }),
    }, (options) => setModelDrafter(modelId, "auto", options))
    expect(models.setModelDrafter).toHaveBeenCalledWith({ modelId, drafter: { _tag: "Auto" } })
    expect(text).toBe([
      `Drafter for ${modelId}: ${drafter.name}.`,
      "Downloading the drafter.",
      `Check progress: magnitude models status ${modelId}`,
      "",
    ].join("\n"))
    expect(json).toMatchObject({ modelId, drafter: { name: drafter.name, reference: { _tag: "Catalog" } }, drafterDownloadStarted: true })
  })
})

describe("model folders", () => {
  it("adds and removes a folder", async () => {
    const added = await runBoth({ addModelFolder: succeed() }, (options) => addModelFolder("/Volumes/External/models", options))
    expect(added.models.addModelFolder).toHaveBeenCalledWith({ path: "/Volumes/External/models" })
    expect(added.text).toBe("Added /Volumes/External/models. Magnitude watches it for GGUF files.\nSee them: magnitude models status\n")
    expect(added.json).toEqual({ path: "/Volumes/External/models" })

    const removed = await runBoth({ removeModelFolder: succeed() }, (options) => removeModelFolder("/Volumes/External/models", options))
    expect(removed.models.removeModelFolder).toHaveBeenCalledWith({ path: "/Volumes/External/models" })
    expect(removed.text).toBe("Removed /Volumes/External/models from Magnitude's folders. Its files are untouched.\n")
    expect(removed.json).toEqual({ path: "/Volumes/External/models" })
  })

  it("turns an automatic location off and on", async () => {
    const disabled = await runBoth({ setModelLocationEnabled: succeed() }, (options) => setModelLocationEnabled("lm-studio", false, options))
    expect(disabled.models.setModelLocationEnabled).toHaveBeenCalledWith({ kind: "LmStudio", enabled: false })
    expect(disabled.text).toBe("Magnitude no longer scans LM Studio. Its files are untouched.\n")
    expect(disabled.json).toEqual({ location: "LmStudio", enabled: false })

    const enabled = await runBoth({ setModelLocationEnabled: succeed() }, (options) => setModelLocationEnabled("ollama", true, options))
    expect(enabled.models.setModelLocationEnabled).toHaveBeenCalledWith({ kind: "Ollama", enabled: true })
    expect(enabled.text).toBe("Magnitude scans Ollama again.\n")
    expect(enabled.json).toEqual({ location: "Ollama", enabled: true })
  })

  it("refuses an unknown location without calling the service", async () => {
    const { client, models } = fakeService({ setModelLocationEnabled: succeed() })
    const result = await capture(setModelLocationEnabled("folder", true), client)
    expect(result.stderr).toBe("Unknown location \"folder\". Use huggingface-cache, lm-studio, llama-cpp, or ollama.\n")
    expect(models.setModelLocationEnabled).not.toHaveBeenCalled()
    expect(result.exitCode).toBe(1)
  })

  it("lists a system Ollama folder it can't read with what to do", async () => {
    const locations = [
      { kind: "Ollama" as const, paths: [
        { path: "/home/ada/.ollama/models", state: "Found" as const },
        { path: "/usr/share/ollama/.ollama/models", state: "Unreadable" as const },
      ] as const, enabled: true, automatic: true, state: "Found" as const },
    ]
    const { text, json, models } = await runBoth({ listModelLocations: succeed({ locations }) }, (options) => listModelLocations(options))
    expect(models.listModelLocations).toHaveBeenCalledWith({})
    expect(text).toMatch(/^Ollama\s+Found\s+\/home\/ada\/\.ollama\/models$/m)
    expect(text).toMatch(/^Ollama\s+Can't read\s+\/usr\/share\/ollama\/\.ollama\/models$/m)
    expect(text).toContain("Can't read /usr/share/ollama/.ollama/models. Magnitude doesn’t have permission to read this folder. A system Ollama service keeps its models in a folder that belongs to the ollama user: give your account read access to it, or set OLLAMA_MODELS to a folder you can read (Magnitude also scans the default folders).")
    expect(json).toEqual({ locations: [{ kind: "Ollama", paths: locations[0]!.paths, enabled: true, automatic: true, state: "Found" }] })
  })

  it("uses the same status words in text as the documentation", async () => {
    const { text } = await runBoth({ listModelLocations: succeed({ locations: makeLocations() }) }, (options) => listModelLocations(options))
    const statuses = text.split("\n").flatMap((line) => line.match(/\b(Off|Not found|Can't read|Found)\b/)?.slice(1) ?? [])
    expect(new Set(statuses)).toEqual(new Set(["Found", "Not found", "Off"]))
  })
})

describe("adding models with a drafter", () => {
  it("catalog pull --drafter none sends the choice and prints the model ID", async () => {
    const added = { _tag: "Custom", modelId: ADDED_MODEL_ID, version: "Q4_K_P", drafter: Option.none(), outcome: "Started", drafterDownloadStarted: false }
    const { text, json, models } = await runBoth({ addHuggingFaceModel: succeed(added) },
      (options) => commandEffect({
        effect: withClient((client) => Effect.flatMap(parseDrafterArgument("none"), (drafter) =>
          pull(client, `${TEST_REPOSITORY}:Q4_K_P`, Effect.runSync(parsePullReference(`${TEST_REPOSITORY}:Q4_K_P`)), drafter))),
        render: renderPull,
        json: asJson(options, pullJson),
      }))
    expect(models.addHuggingFaceModel).toHaveBeenCalledWith({
      repository: TEST_REPOSITORY,
      revision: Option.none(),
      version: Option.some("Q4_K_P"),
      drafter: { _tag: "None" },
      force: false,
    })
    expect(text).toContain("Drafter: none")
    expect(text).toContain(`Model ID: ${ADDED_MODEL_ID}`)
    expect(json).toEqual({
      _tag: "HuggingFace",
      repository: TEST_REPOSITORY,
      versionRequested: true,
      added: { _tag: "Custom", modelId: ADDED_MODEL_ID, version: "Q4_K_P", outcome: "Started", drafterDownloadStarted: false },
    })
  })

  it("catalog inspect passes a huggingface.co link through and prints the repository step", async () => {
    const link = `https://huggingface.co/${TEST_REPOSITORY}`
    const { text, json, models } = await runBoth({
      inspectModelCandidate: succeed(makeInspection()),
      getCatalog: succeed(catalogState([makeCatalogModel()])),
      listDrafterCandidates: succeed({ candidates: makeDrafterCandidates(), entered: Option.none() }),
    }, (options) => commandEffect({
      effect: withClient((client) => readInspection(client, link)),
      render: renderInspection,
      json: asJson(options, inspectionJson),
    }))
    expect(models.inspectModelCandidate).toHaveBeenCalledWith({ reference: link })
    expect(models.listDrafterCandidates).toHaveBeenCalledWith({
      target: { _tag: "Candidate", source: { _tag: "HuggingFace", repository: TEST_REPOSITORY, commit: TEST_COMMIT }, files: ["Qwen-Test-Q4_K_P.gguf"] },
      other: Option.none(),
    })
    expect(text.split("\n")[0]).toBe("Qwen Test Uncensored")
    expect(text.trimEnd().split("\n").at(-1)).toBe(`Download: magnitude catalog pull ${TEST_REPOSITORY}:Q4_K_P`)
    expect(json).toMatchObject({ inspection: { displayName: "Qwen Test Uncensored", versions: expect.any(Array) }, drafters: expect.any(Array) })
  })
})

describe("hidden models are addressable", () => {
  const hidden = makeFoundModel({ modelId: CustomFormModelIdSchema.make("hf:a/b/c.gguf") })

  it("shows a hidden model's status, saying it is hidden", async () => {
    const { text, json, models } = await runBoth({
      getCatalog: succeed(catalogState([makeAddedModel()], [], [hidden])),
      listDrafterCandidates: succeed({ candidates: makeDrafterCandidates(), entered: Option.none() }),
    }, (options) => showModelsStatus(hidden.modelId, options))
    expect(models.listDrafterCandidates).toHaveBeenCalledWith({ target: { _tag: "Model", modelId: hidden.modelId }, other: Option.none() })
    expect(text).toContain(`Model ID    ${hidden.modelId}`)
    expect(text).toContain(`Hidden      Yes - unhide it: magnitude models unhide ${hidden.modelId}`)
    expect(json).toMatchObject({ model: { modelId: hidden.modelId }, hidden: true })
  })

  it("reports a visible model as not hidden", async () => {
    const { json } = await runBoth({
      getCatalog: succeed(catalogState([makeAddedModel()], [], [hidden])),
      listDrafterCandidates: succeed({ candidates: [], entered: Option.none() }),
    }, (options) => showModelsStatus(ADDED_MODEL_ID, options))
    expect(json).toMatchObject({ model: { modelId: ADDED_MODEL_ID }, hidden: false })
  })

  it("removes a hidden model using what it knows about the model", async () => {
    const { text, models } = await runBoth({
      getCatalog: succeed(catalogState([], [], [hidden])),
      removeLocalModel: succeed({ keptComponents: [] }),
    }, (options) => removeModel(hidden.modelId, options))
    expect(models.removeLocalModel).toHaveBeenCalledWith({ modelId: hidden.modelId })
    expect(text).toContain(`Hid ${hidden.modelId} from Magnitude. Its files are untouched.`)
  })

  it("shows a hidden catalog model's catalog details", async () => {
    const catalog = makeInstalledCatalogModel()
    const { text, json } = await runBoth({ getCatalog: succeed(catalogState([], [], [catalog])) },
      (options) => showCatalogModel(catalog.modelId, options))
    expect(text).toContain(`Model ID      ${catalog.modelId}`)
    expect(json).toMatchObject({ model: { modelId: catalog.modelId } })
  })
})

const statusView = (overrides: Partial<ModelsStatusView> = {}): ModelsStatusView => ({
  models: [],
  unsupported: [],
  hidden: [],
  all: false,
  ...overrides,
})

const rowOf = (output: string, modelId: string): string => output.split("\n").find((line) => line.endsWith(modelId)) ?? ""

describe("models status variant and drafter columns", () => {
  const turnedOff = makeFoundModel({
    modelId: CustomFormModelIdSchema.make("hf:z/off/off.gguf"),
    drafterChoice: { _tag: "None" },
  })

  it("shows what each model is a variant of and its drafter", () => {
    const catalog = makeInstalledCatalogModel()
    const output = renderModelsStatus(statusView({
      models: [catalog, makeAddedModel(), makeFoundModel(), makeImportedModel(), turnedOff],
    }))
    expect(output).toMatch(/^MODEL\s+SOURCE\s+VARIANT OF\s+DRAFTER\s+MEMORY\s+CONTEXT\s+STATUS\s+MODEL ID$/m)
    expect(rowOf(output, ADDED_MODEL_ID)).toMatch(/Hugging Face\s+Qwen Test \(qwen-test:gguf:q4\)\s+Magnitude DFlash for Qwen Test\s/)
    expect(rowOf(output, FOUND_MODEL_ID)).toMatch(/Qwen Test \(qwen-test:gguf:q4\)\s+Faster with a drafter · Add \(900 MB\)\s/)
    expect(rowOf(output, IMPORTED_MODEL_ID)).toMatch(/GGUF file\s+—\s+—\s/)
    expect(rowOf(output, turnedOff.modelId)).toMatch(/Turned off\s/)
    expect(rowOf(output, turnedOff.modelId)).not.toContain("Faster with a drafter")
    expect(rowOf(output, catalog.modelId)).toMatch(/Catalog\s+—\s+—\s/)
  })

  it("shows an automatic drafter that isn't worked out yet as being found, never as none", () => {
    const pending = makeFoundModel({ drafterPending: true, recommendedDrafter: Option.none() })
    expect(rowOf(renderModelsStatus(statusView({ models: [pending] })), FOUND_MODEL_ID)).toMatch(/qwen-test:gguf:q4\s+Finding drafter…\s/)
    const output = renderModelDetail({ model: pending, models: [], drafters: { _tag: "Listed", candidates: [] }, hidden: false })
    expect(output).toMatch(/^ {2}Drafter\s+Finding drafter…$/m)
    expect(output).not.toMatch(/^ {2}Drafter\s+None/m)
  })

  it("names the variant by its ID when the catalog doesn't list it", () => {
    const output = renderModelsStatus(statusView({ models: [makeFoundModel()] }))
    expect(rowOf(output, FOUND_MODEL_ID)).toMatch(/Hugging Face cache\s+qwen-test:gguf:q4\s/)
  })

  it("shows a turned-off drafter in details and marks none as the current choice", () => {
    const output = renderModelDetail({ model: turnedOff, models: [], drafters: { _tag: "Listed", candidates: makeDrafterCandidates() }, hidden: false })
    expect(output).toMatch(/^ {2}Drafter\s+Turned off$/m)
    expect(output).not.toContain("Faster with a drafter")
    expect(output).toMatch(/^\* None\s+—/m)
    expect(output).not.toMatch(/^\* Magnitude DFlash/m)
  })

  it("marks the preselected drafter as current for an automatic choice with nothing attached yet", () => {
    const output = renderModelDetail({ model: makeFoundModel(), models: [], drafters: { _tag: "Listed", candidates: makeDrafterCandidates() }, hidden: false })
    expect(output).toMatch(/^\* Magnitude DFlash for Qwen Test/m)
    expect(output).toMatch(/^ {2}None\s+—/m)
  })

  it("shows no context while a model downloads, before it has an assessment", () => {
    const downloading = makeAddedModel({
      acquisitionState: { _tag: "Installing", progress: { stage: "downloading", completedBytes: 1, totalBytes: 4, bytesPerSecond: Option.none() } },
      servingState: { _tag: "Assessing", profile: { contextLength: 100_000 } },
    })
    const detail = renderModelDetail({ model: downloading, models: [], drafters: { _tag: "Listed", candidates: makeDrafterCandidates() }, hidden: false })
    expect(detail).not.toMatch(/^ {2}Context/m)
    expect(detail).not.toContain("100K")
    expect(rowOf(renderModelsStatus(statusView({ models: [downloading] })), ADDED_MODEL_ID)).not.toContain("100K")

    expect(renderModelDetail({ model: makeAddedModel(), models: [], drafters: { _tag: "Listed", candidates: makeDrafterCandidates() }, hidden: false }))
      .toMatch(/^ {2}Context\s+32K$/m)
  })

  it("hides Update for an attached drafter while it downloads", () => {
    const base = Option.getOrThrow(makeAddedModel().drafter)
    const downloading = makeAddedModel({
      drafter: Option.some({ ...base, updateAvailable: true, availability: { _tag: "Downloading", progress: { stage: "downloading", completedBytes: 1, totalBytes: 2, bytesPerSecond: Option.none() } } }),
    })
    expect(rowOf(renderModelsStatus(statusView({ models: [downloading] })), ADDED_MODEL_ID)).not.toContain("Update available")
  })
})

const INTERNAL_KEYS = [
  "assessmentId",
  "environmentId",
  "rankingScores",
  "fidelityRank",
  "acceleratorId",
  "memoryDomainId",
  "deviceId",
  "primaryPath",
  "ownership",
  "limitingResource",
] as const
const internalKey = new RegExp(`"(${INTERNAL_KEYS.join("|")})"`)

const encode = <A>(document: { readonly encode: (value: A) => Effect.Effect<unknown, unknown> }, value: A) =>
  JSON.parse(JSON.stringify(Effect.runSync(document.encode(value)))) as Record<string, unknown>

const domain = LocalInferenceMemoryDomainIdSchema.make("cuda0")
const memoryEvidence = { memoryDomainId: domain, capacityBytes: 100, requiredBytes: 10, compatibilityReserveBytes: 1, remainingBytes: 89 }
const allocation = { contextWindowTokens: 4096, memoryDomains: [{ memoryDomainId: domain, modelBytes: 8, contextBytes: 1, computeBytes: 1, auxiliaryBytes: 0 }] }
const plan = { contextWindowTokens: 4096, requiredMemoryBytes: 10, device: { deviceId: LocalInferenceDeviceIdSchema.make("cuda:0"), backend: "cuda" as const } }

const withEvidence = <M extends CatalogLocalModel | CustomLocalModel>(model: M): M => {
  const serving = model.servingState
  if (serving._tag !== "Assessed" || serving.assessment._tag !== "Fits") return model
  return { ...model, servingState: { ...serving, assessment: { ...serving.assessment, memory: { domains: [memoryEvidence], totalRequiredBytes: 10, requiredSystemMemoryBytes: 0 } } } }
}

const residencies = [
  { _tag: "Loading", stage: "loading_weights", fraction: 0.5, plannedAllocation: Option.some(plan) },
  { _tag: "Ready", allocation },
  { _tag: "Stopping", reason: "user_stop", allocation: { _tag: "Resident", allocation } },
  { _tag: "Stopping", reason: "user_stop", allocation: { _tag: "Planned", allocation: Option.some(plan) } },
] as const

const leakyModels = () => {
  const catalog = withEvidence(makeInstalledCatalogModel())
  const installed = catalog.acquisitionState as Extract<CatalogLocalModel["acquisitionState"], { _tag: "Installed" }>
  const catalogs = residencies.map((residencyState, index) => ({
    ...catalog,
    modelId: CatalogFormModelIdSchema.make(`leak-${index}:gguf:q4`),
    acquisitionState: { ...installed, residencyState },
  }) as CatalogLocalModel)
  const optimizing = {
    ...catalog,
    modelId: CatalogFormModelIdSchema.make("leak-optimizing:gguf:q4"),
    acquisitionState: { ...installed, _tag: "Optimizing", progress: { stage: "tuning", completed: 1, total: 2, device: Option.some(plan.device) } },
  } as CatalogLocalModel
  const found = withEvidence(makeFoundModel())
  const loadingFound = { ...found, acquisitionState: { ...found.acquisitionState, residencyState: residencies[0] } } as CustomLocalModel
  const assessed = makeAddedModel().servingState as Extract<CustomLocalModel["servingState"], { _tag: "Assessed" }>
  const tooLarge = makeImportedModel({
    modelId: CustomFormModelIdSchema.make(`local:${"f".repeat(64)}`),
    servingState: { ...assessed, assessment: { _tag: "DoesNotFit", assessmentId: ModelAssessmentIdSchema.make("assessment_large"), environmentId: AssessmentEnvironmentIdSchema.make("environment_test"), profile: { contextLength: 4096 }, memoryDomains: [memoryEvidence], totalRequiredBytes: 10, deficitBytes: 4, limitingResource: "cuda0" } },
  })
  return { catalog, catalogs, optimizing, found, loadingFound, tooLarge, added: withEvidence(makeAddedModel()) }
}

describe("--json never leaks internals", () => {
  it("strips identities, managed paths, ownership, and ranking inputs from every command's document", async () => {
    const { catalog, catalogs, optimizing, found, loadingFound, tooLarge, added } = leakyModels()
    const all = [catalog, ...catalogs, optimizing, found, tooLarge, added]
    const state: ModelCatalogState = catalogState(all, [], [withEvidence(makeFoundModel({ modelId: CustomFormModelIdSchema.make("hf:a/b/c.gguf") }))])
    for (const model of all) expect(JSON.stringify(model), model.modelId).toMatch(internalKey)

    const documents: Record<string, unknown>[] = [
      encode(catalogListJson, state),
      encode(catalogDetailJson, { model: catalog, models: all }),
      encode(recommendationsJson, { catalog: state, models: [catalog], hardware: makeHardware(), preference: { label: "Balanced" }, ranked: [catalog, ...catalogs] }),
      encode(modelsStatusJson, statusView({ models: all, hidden: [withEvidence(makeFoundModel({ modelId: CustomFormModelIdSchema.make("hf:a/b/c.gguf") }))] })),
      ...all.map((model) => encode(modelDetailJson, { model, models: all, drafters: { _tag: "Listed", candidates: makeDrafterCandidates() }, hidden: false })),
      encode(hardwareJson, { hardware: makeHardware(), models: [loadingFound] }),
      encode(hardwareJson, { hardware: makeHardware(), models: [catalogs[1]!] }),
      encode(searchJson, { query: "q", all: true, response: makeSearchResponse(), models: all }),
      encode(inspectionJson, { inspection: makeInspection(), drafters: makeDrafterCandidates(), models: all }),
    ]

    const status = await runBoth({ getCatalog: succeed(state) }, (options) => showModelsStatus(undefined, { ...options, all: true }))
    const locations = await runBoth({ listModelLocations: succeed({ locations: makeLocations() }) }, (options) => listModelLocations(options))
    const commandDocuments = [status.json, locations.json]
    for (const document of [...documents, ...commandDocuments]) expect(JSON.stringify(document)).not.toMatch(internalKey)
  })

  it("keeps catalog documents free of acquisition and residency state", () => {
    const { catalog, catalogs } = leakyModels()
    const state = catalogState([catalog, ...catalogs])
    const documents = [
      encode(catalogListJson, state),
      encode(catalogDetailJson, { model: catalog, models: [catalog] }),
      encode(recommendationsJson, { catalog: state, models: [catalog], hardware: makeHardware(), preference: { label: "Balanced" }, ranked: [catalog] }),
    ]
    for (const document of documents) expect(JSON.stringify(document)).not.toMatch(/"(acquisitionState|residencyState|installation)"/)
  })

  it("keeps the product facts behind the stripped fields", () => {
    const { catalogs, tooLarge, catalog } = leakyModels()
    const hardware = encode(hardwareJson, { hardware: makeHardware(), models: [catalogs[1]!] })
    expect(hardware.hardware).toMatchObject({
      accelerators: [{ name: "Test GPU", backend: "cuda" }],
      memoryDomains: [{ kind: "PhysicalDevice", totalBytes: 24 * 1024 ** 3, sharesSystemMemory: false }],
    })
    expect(hardware.currentModel).toMatchObject({
      acquisitionState: {
        _tag: "Installed",
        installation: { installedBytes: expect.any(Number) },
        residencyState: { _tag: "Ready", allocation: { contextWindowTokens: 4096, memoryDomains: [{ modelBytes: 8, contextBytes: 1 }] } },
      },
    })
    const loading = encode(modelDetailJson, { model: catalogs[0]!, models: [], drafters: { _tag: "NotApplicable" }, hidden: false })
    expect(loading.model).toMatchObject({ acquisitionState: { residencyState: { plannedAllocation: { device: { backend: "cuda" } } } } })
    const detail = encode(modelDetailJson, { model: tooLarge, models: [], drafters: { _tag: "NotApplicable" }, hidden: false })
    expect(detail.model).toMatchObject({ servingState: { assessment: { _tag: "DoesNotFit", totalRequiredBytes: 10, deficitBytes: 4 } } })
    const shown = encode(catalogDetailJson, { model: catalog, models: [] })
    expect(shown.model).toMatchObject({
      catalogData: { intelligence: 75, quantizationAware: false },
      servingState: { assessment: { _tag: "Fits", memory: { totalRequiredBytes: 10 } } },
    })
  })
})
