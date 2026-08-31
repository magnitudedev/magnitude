import { Effect, Option } from "effect"
import type { HarnessConnectionSpec } from "../contract"
import type { HarnessConnectionPaths } from "../paths"
import {
  LOCAL_TOKEN,
  OPENAI_BASE_URL,
  defineConnector,
  jsonObject,
  launchPlan,
  readOr,
  removeJsoncPaths,
  updateJsonc,
  valueAt,
  writeIfChanged,
} from "../shared"
import { modelMaxTokens } from "../model-fields"
import { enabledReasoningEfforts, supportsReasoningEffort } from "../reasoning"

const CLINE_EFFORTS = new Set(["minimal", "low", "medium", "high", "xhigh", "max"])

const clineReasoningOptions = (model: HarnessConnectionSpec["models"][number]): ReadonlyArray<Record<string, unknown>> => {
  if (!model.capabilities.reasoning.supported) return []
  const efforts = enabledReasoningEfforts(model).filter((effort) => CLINE_EFFORTS.has(effort))
  return [
    ...(supportsReasoningEffort(model, "none") ? [{ type: "toggle" }] : []),
    ...(efforts.length === 0 ? [] : [{ type: "effort", values: ["default", ...efforts] }]),
  ]
}

export const clineProviderSettings = (current: Option.Option<string>) => ({
  provider: "openai-compatible",
  apiKey: LOCAL_TOKEN,
  baseUrl: OPENAI_BASE_URL,
  protocol: "openai-chat",
  client: "openai-compatible",
  ...Option.match(current, { onNone: () => ({}), onSome: (model) => ({ model }) }),
})

export const clineModelCatalog = (models: HarnessConnectionSpec["models"]) => Object.fromEntries(
  models.map((model) => [model.id, {
    id: model.id,
    name: model.name,
    contextWindow: model.contextWindow,
    maxTokens: modelMaxTokens(model),
    supportsVision: model.capabilities.vision,
    supportsReasoning: model.capabilities.reasoning.supported,
    capabilities: [
      ...(model.capabilities.tools ? ["tools"] : []),
      ...(model.capabilities.vision ? ["images"] : []),
      ...(model.capabilities.reasoning.supported ? ["reasoning"] : []),
      ...(enabledReasoningEfforts(model).some((effort) => CLINE_EFFORTS.has(effort))
        ? ["reasoning-effort"]
        : []),
    ],
    ...(model.capabilities.reasoning.supported
      ? { reasoningOptions: clineReasoningOptions(model) }
      : {}),
  }]),
)

export const clineModelRegistryEntry = (models: HarnessConnectionSpec["models"]) => ({
  models: clineModelCatalog(models),
})

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
    const updatedAt = valueAt(existing, ["updatedAt"])
    yield* writeIfChanged(paths.clineProviders, source, updateJsonc(source, [
      [["version"], 1],
      [["modes"], valueAt(value, ["modes"]) ?? {}],
      [["providers", "openai-compatible"], {
        settings: clineProviderSettings(spec.setCurrent),
        updatedAt: typeof updatedAt === "string" ? updatedAt : new Date().toISOString(),
        tokenSource: "manual",
      }],
    ]))
    const modelsSource = yield* readOr(paths.clineModels, "{}\n")
    yield* writeIfChanged(paths.clineModels, modelsSource, updateJsonc(modelsSource, [
      [["version"], 1],
      [["providers", "openai-compatible"], clineModelRegistryEntry(spec.models)],
    ]))
  }),
  disconnect: () => Effect.gen(function* () {
    const source = yield* readOr(paths.clineProviders, "{}\n")
    yield* writeIfChanged(paths.clineProviders, source, removeJsoncPaths(source, [["providers", "openai-compatible"]]))
    const modelsSource = yield* readOr(paths.clineModels, "{}\n")
    yield* writeIfChanged(paths.clineModels, modelsSource, removeJsoncPaths(modelsSource, [["providers", "openai-compatible"]]))
  }),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, [
      "--tui",
      "--data-dir",
      paths.clineDataDir,
      "--provider",
      "openai-compatible",
      "--model",
      modelId,
    ])
  },
})
