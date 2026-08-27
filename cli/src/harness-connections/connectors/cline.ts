import { applyEdits, modify } from "jsonc-parser"
import { Effect, Option } from "effect"
import { isDeepStrictEqual } from "node:util"
import type { HarnessConnectionSpec } from "../contract"
import type { HarnessConnectionPaths } from "../paths"
import {
  LOCAL_TOKEN,
  OPENAI_BASE_URL,
  defineConnector,
  jsonObject,
  launchPlan,
  readOr,
  updateJsonc,
  valueAt,
  writeIfChanged,
} from "../shared"

export const clineProviderSettings = (current: Option.Option<string>) => ({
  provider: "openai-compatible",
  apiKey: LOCAL_TOKEN,
  baseUrl: OPENAI_BASE_URL,
  ...Option.match(current, { onNone: () => ({}), onSome: (model) => ({ model }) }),
})

export const clineModelCatalog = (models: HarnessConnectionSpec["models"]) => Object.fromEntries(
  models.map(({ id, name }) => [id, { id, name }]),
)

const isManagedSettings = (value: unknown, current: Option.Option<string>): boolean => {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false
  const settings = value as Record<string, unknown>
  const expected = clineProviderSettings(current)
  return Object.keys(settings).length === Object.keys(expected).length
    && Object.entries(expected).every(([key, expectedValue]) => settings[key] === expectedValue)
}

export const makeClineConnector = (paths: HarnessConnectionPaths) => defineConnector({
  id: "cline",
  name: "Cline",
  executable: "cline",
  skillInstallationTarget: "cline-user",
  configurationFiles: [paths.clineProviders, paths.clineModels],
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.clineProviders, "{}\n")
    const value = jsonObject(source)
    const existing = valueAt(value, ["providers", "openai-compatible"])
    if (existing !== undefined && (
      !isManagedSettings(valueAt(existing, ["settings"]), spec.setCurrent)
      || valueAt(existing, ["tokenSource"]) !== "manual"
    )) throw new Error("Cline already contains a conflicting openai-compatible provider")
    const version = valueAt(value, ["version"])
    if (version !== undefined && version !== 1) throw new Error("Cline providers configuration has an unsupported version")
    yield* writeIfChanged(paths.clineProviders, source, updateJsonc(source, [
      [["version"], 1],
      [["modes"], valueAt(value, ["modes"]) ?? {}],
      [["providers", "openai-compatible"], {
        settings: clineProviderSettings(spec.setCurrent),
        updatedAt: new Date().toISOString(),
        tokenSource: "manual",
      }],
    ]))
    const modelsSource = yield* readOr(paths.clineModels, "{}\n")
    const modelsValue = jsonObject(modelsSource)
    const modelsVersion = valueAt(modelsValue, ["version"])
    if (modelsVersion !== undefined && modelsVersion !== 1) {
      throw new Error("Cline models configuration has an unsupported version")
    }
    const existingModels = valueAt(modelsValue, ["providers", "openai-compatible", "models"])
    if (existingModels !== undefined && !isDeepStrictEqual(existingModels, clineModelCatalog(spec.models))) {
      throw new Error("Cline already contains a conflicting openai-compatible model catalog")
    }
    yield* writeIfChanged(paths.clineModels, modelsSource, updateJsonc(modelsSource, [
      [["version"], 1],
      [["providers", "openai-compatible", "models"], clineModelCatalog(spec.models)],
    ]))
  }),
  disconnect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.clineProviders, "{}\n")
    const value = jsonObject(source)
    const entry = valueAt(value, ["providers", "openai-compatible"])
    const settings = valueAt(entry, ["settings"])
    const managed = isManagedSettings(settings, spec.setCurrent)
      || (Option.isSome(spec.setCurrent) && isManagedSettings(settings, Option.none()))
    if (entry !== undefined && managed && valueAt(entry, ["tokenSource"]) === "manual") {
      yield* writeIfChanged(paths.clineProviders, source, applyEdits(source, modify(
        source,
        ["providers", "openai-compatible"],
        undefined,
        { formattingOptions: { insertSpaces: true, tabSize: 2, eol: "\n" } },
      )))
    }
    const modelsSource = yield* readOr(paths.clineModels, "{}\n")
    const modelsValue = jsonObject(modelsSource)
    const provider = valueAt(modelsValue, ["providers", "openai-compatible"])
    const catalog = valueAt(provider, ["models"])
    if (!isDeepStrictEqual(catalog, clineModelCatalog(spec.models))) return
    const providerPath = provider !== null
      && typeof provider === "object"
      && !Array.isArray(provider)
      && Object.keys(provider).length === 1
      ? ["providers", "openai-compatible"]
      : ["providers", "openai-compatible", "models"]
    yield* writeIfChanged(paths.clineModels, modelsSource, applyEdits(modelsSource, modify(
      modelsSource,
      providerPath,
      undefined,
      { formattingOptions: { insertSpaces: true, tabSize: 2, eol: "\n" } },
    )))
  }),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--tui", "--provider", "openai-compatible", "--model", modelId])
  },
})
