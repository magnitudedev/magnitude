import {
  AutomaticModelLocationKindSchema,
  CustomFormModelIdSchema,
  ModelIdSchema,
  ModelLocationSchema,
  Models,
  type AutomaticModelLocationKind,
  type CustomFormModelId,
  type DrafterCandidate,
  type ModelId,
  type ModelLocation,
  type ModelLocationKind,
  type ModelLocationState,
} from "@magnitudedev/sdk"
import { shownModelLocationPaths, unreadableModelLocationMessage } from "@magnitudedev/client-common"
import { Effect, Option, Schema } from "effect"
import { resolveDrafterChoice } from "./hugging-face-runtime"
import { withClient, type CliModelsClient } from "./inference-client"
import {
  absolutePath,
  decodeCustomModelId,
  decodeModelId,
  ModelCommandError,
  parseDrafterArgument,
  type DrafterArgument,
} from "./model-references"
import { ensureTrailingNewline, jsonDocument, renderTable, runCommand, asJson, type JsonOption } from "./output"

const requirePath = (input: string, what: string): Effect.Effect<string, ModelCommandError> => {
  const trimmed = input.trim()
  return trimmed.length === 0
    ? Effect.fail(new ModelCommandError({ message: `Expected ${what}` }))
    : Effect.succeed(absolutePath(trimmed))
}

const drafterLine = (drafter: Option.Option<DrafterCandidate>) =>
  `Drafter: ${Option.match(drafter, { onNone: () => "none", onSome: ({ name }) => name })}`

export const ImportResultSchema = Schema.extend(
  Schema.Struct({ path: Schema.String }),
  Models.importModelFile.successSchema,
)
export type ImportResult = typeof ImportResultSchema.Type

export const renderImport = (result: ImportResult): string => ensureTrailingNewline([
  result.outcome === "AlreadyOnThisComputer"
    ? `Already on this computer: ${result.path} has the same bytes as ${result.modelId}, so it is that model. Magnitude never moves or deletes it.`
    : `Added ${result.path}. Magnitude uses the file where it is and never moves or deletes it.`,
  drafterLine(result.drafter),
  `Model ID: ${result.modelId}`,
  result.drafterDownloadStarted
    ? `Check progress: magnitude models status ${result.modelId}`
    : `Load it: magnitude models load ${result.modelId}`,
].join("\n"))

export const importFile = (client: CliModelsClient, path: string, argument: DrafterArgument) => Effect.gen(function* () {
  const drafter = yield* resolveDrafterChoice(client, {
    _tag: "Candidate",
    source: { _tag: "Local", path },
    files: [path],
  }, argument)
  const imported = yield* client.models.importModelFile({ path, drafter })
  return { path, ...imported } satisfies ImportResult
})

export const importModel = (pathInput: string, options: JsonOption & { readonly drafter?: string } = {}) => runCommand({
  effect: Effect.gen(function* () {
    const path = yield* requirePath(pathInput, "a path to a .gguf file")
    if (!path.toLowerCase().endsWith(".gguf")) {
      return yield* new ModelCommandError({ message: `Expected a .gguf file: ${path}` })
    }
    const drafterArgument = yield* parseDrafterArgument(options.drafter ?? "auto")
    return yield* withClient((client) => importFile(client, path, drafterArgument))
  }),
  render: renderImport,
  json: asJson(options, jsonDocument(ImportResultSchema, (result: ImportResult) => result)),
})

export const catalogDrafterRefusal = (modelId: string): string =>
  `${modelId} is a catalog model. Catalog models keep the drafter Magnitude ships with them; it changes only when you update with \`magnitude catalog pull ${modelId}\`.`

export const SetDrafterResultSchema = Schema.extend(
  Schema.Struct({ modelId: CustomFormModelIdSchema }),
  Models.setModelDrafter.successSchema,
)
export type SetDrafterResult = typeof SetDrafterResultSchema.Type

export const renderSetDrafter = (result: SetDrafterResult, requested: DrafterArgument["_tag"] = "None"): string => Option.match(result.drafter, {
  onNone: () => requested === "None"
    ? `Turned off the drafter for ${result.modelId}.\n`
    : `No drafter is known to work with ${result.modelId}, so it runs without one.\n`,
  onSome: (drafter) => ensureTrailingNewline([
    `Drafter for ${result.modelId}: ${drafter.name}.`,
    ...(result.drafterDownloadStarted
      ? ["Downloading the drafter.", `Check progress: magnitude models status ${result.modelId}`]
      : []),
  ].join("\n")),
})

export const setDrafter = (client: CliModelsClient, modelId: CustomFormModelId, argument: DrafterArgument) =>
  Effect.gen(function* () {
    const drafter = yield* resolveDrafterChoice(client, { _tag: "Model", modelId }, argument)
    const result = yield* client.models.setModelDrafter({ modelId, drafter })
    return { modelId, ...result } satisfies SetDrafterResult
  })

export const setModelDrafter = (modelInput: string, drafterInput: string, options: JsonOption = {}) => runCommand({
  effect: Effect.gen(function* () {
    const modelId = yield* decodeCustomModelId(modelInput.trim(), catalogDrafterRefusal)
    const argument = yield* parseDrafterArgument(drafterInput)
    const result = yield* withClient((client) => setDrafter(client, modelId, argument))
    return { result, requested: argument._tag }
  }),
  render: ({ result, requested }) => renderSetDrafter(result, requested),
  json: asJson(options, jsonDocument(SetDrafterResultSchema, ({ result }: { readonly result: SetDrafterResult }) => result)),
})

const ModelIdJsonSchema = Schema.Struct({ modelId: ModelIdSchema })
const modelIdJson = jsonDocument(ModelIdJsonSchema, (modelId: ModelId) => ({ modelId }))

/** Found models, including a catalog model whose files are only in an external location; the service refuses others. */
export const hideModel = (modelInput: string, options: JsonOption = {}) => runCommand({
  effect: decodeModelId(modelInput.trim()).pipe(Effect.flatMap((modelId) =>
    withClient((client) => client.models.hideModel({ modelId }).pipe(Effect.as(modelId))))),
  render: (modelId) => `Hid ${modelId} from Magnitude. Its files are untouched.\nUnhide it: magnitude models unhide ${modelId}\n`,
  json: asJson(options, modelIdJson),
})

export const unhideModel = (modelInput: string, options: JsonOption = {}) => runCommand({
  effect: decodeModelId(modelInput.trim()).pipe(Effect.flatMap((modelId) =>
    withClient((client) => client.models.unhideModel({ modelId }).pipe(Effect.as(modelId))))),
  render: (modelId) => `${modelId} is visible in Magnitude again.\n`,
  json: asJson(options, modelIdJson),
})

const LOCATION_NAMES: Record<ModelLocationKind, string> = {
  HuggingFaceCache: "Hugging Face cache",
  LmStudio: "LM Studio",
  LlamaCpp: "llama.cpp",
  Ollama: "Ollama",
  Folder: "Folder",
}

const LOCATION_STATES: Record<ModelLocationState, string> = {
  Found: "Found",
  NotFound: "Not found",
  Unreadable: "Can't read",
}

/** One table row per directory a location shows, so each directory carries its own status. */
interface LocationRow {
  readonly location: ModelLocation
  readonly path: string
  readonly state: ModelLocationState
}

const locationRows = (locations: readonly ModelLocation[]): readonly LocationRow[] =>
  locations.flatMap((location) => shownModelLocationPaths(location).map(({ path, state }): LocationRow => ({ location, path, state })))

const rowStatus = ({ location, state }: LocationRow): string => location.enabled ? LOCATION_STATES[state] : "Off"

const unreadableNotes = (locations: readonly ModelLocation[]): readonly string[] =>
  locations.filter(({ enabled }) => enabled).flatMap(({ kind, paths }) => paths
    .filter(({ state }) => state === "Unreadable")
    .flatMap(({ path }) => ["", `Can't read ${path}. ${unreadableModelLocationMessage(kind)}`]))

const folderPath = (folder: ModelLocation): string => folder.paths[0].path

export const renderModelLocations = (locations: readonly ModelLocation[]): string => {
  const automatic = locations.filter(({ automatic }) => automatic)
  const folders = locations.filter(({ automatic }) => !automatic)
    .sort((left, right) => folderPath(left).localeCompare(folderPath(right)))
  return ensureTrailingNewline([
    "Automatic locations",
    "",
    automatic.length === 0 ? "No automatic locations were found on this computer." : renderTable(locationRows(automatic), [
      { heading: "LOCATION", value: ({ location }) => LOCATION_NAMES[location.kind] },
      { heading: "STATUS", value: rowStatus },
      { heading: "PATH", value: ({ path }) => path },
    ]).trimEnd(),
    ...unreadableNotes(automatic),
    "",
    "Your folders",
    "",
    folders.length === 0 ? "No folders added. Add one: magnitude models folders add <path>" : renderTable(locationRows(folders), [
      { heading: "PATH", value: ({ path }) => path },
      { heading: "STATUS", value: rowStatus },
    ]).trimEnd(),
    ...unreadableNotes(folders),
  ].join("\n"))
}

export const ModelLocationsJsonSchema = Schema.Struct({ locations: Schema.Array(ModelLocationSchema) })

export const listModelLocations = (options: JsonOption = {}) => runCommand({
  effect: withClient((client) => client.models.listModelLocations({})).pipe(Effect.map(({ locations }) => locations)),
  render: renderModelLocations,
  json: asJson(options, jsonDocument(ModelLocationsJsonSchema, (locations: readonly ModelLocation[]) => ({ locations }))),
})

const FolderJsonSchema = Schema.Struct({ path: Schema.String })
const folderJson = jsonDocument(FolderJsonSchema, (path: string) => ({ path }))

export const addModelFolder = (pathInput: string, options: JsonOption = {}) => runCommand({
  effect: requirePath(pathInput, "a folder path").pipe(Effect.flatMap((path) =>
    withClient((client) => client.models.addModelFolder({ path }).pipe(Effect.as(path))))),
  render: (path) => `Added ${path}. Magnitude watches it for GGUF files.\nSee them: magnitude models status\n`,
  json: asJson(options, folderJson),
})

export const removeModelFolder = (pathInput: string, options: JsonOption = {}) => runCommand({
  effect: requirePath(pathInput, "a folder path").pipe(Effect.flatMap((path) =>
    withClient((client) => client.models.removeModelFolder({ path }).pipe(Effect.as(path))))),
  render: (path) => `Removed ${path} from Magnitude's folders. Its files are untouched.\n`,
  json: asJson(options, folderJson),
})

const AUTOMATIC_LOCATION_ARGUMENTS: Record<string, AutomaticModelLocationKind> = {
  "huggingface-cache": "HuggingFaceCache",
  "lm-studio": "LmStudio",
  "llama-cpp": "LlamaCpp",
  ollama: "Ollama",
}

export const parseAutomaticLocation = (input: string): Option.Option<AutomaticModelLocationKind> =>
  Option.fromNullable(AUTOMATIC_LOCATION_ARGUMENTS[input.trim().toLowerCase()])

const LocationEnabledJsonSchema = Schema.Struct({ location: AutomaticModelLocationKindSchema, enabled: Schema.Boolean })

export const setModelLocationEnabled = (locationInput: string, enabled: boolean, options: JsonOption = {}) => runCommand({
  effect: Option.match(parseAutomaticLocation(locationInput), {
    onNone: () => Effect.fail(new ModelCommandError({ message: `Unknown location "${locationInput}". Use huggingface-cache, lm-studio, llama-cpp, or ollama.` })),
    onSome: (kind) => withClient((client) => client.models.setModelLocationEnabled({ kind, enabled }).pipe(Effect.as(kind))),
  }),
  render: (kind) => enabled
    ? `Magnitude scans ${LOCATION_NAMES[kind]} again.\n`
    : `Magnitude no longer scans ${LOCATION_NAMES[kind]}. Its files are untouched.\n`,
  json: asJson(options, jsonDocument(LocationEnabledJsonSchema, (kind: AutomaticModelLocationKind) => ({ location: kind, enabled }))),
})
