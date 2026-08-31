import { Effect } from "effect"
import type { HarnessConnectionPaths } from "../paths"
import {
  CHAT_COMPLETIONS_API,
  LOCAL_TOKEN,
  OPENAI_BASE_URL,
  defineConnector,
  launchPlan,
  readOr,
  removeJsoncPaths,
  updateJsonc,
  writeIfChanged,
} from "../shared"
import type { HarnessConnectionSpec } from "../contract"
import { modelInput, modelMaxTokens, zeroCost } from "../model-fields"
import { hasReasoning, projectReasoningControls } from "../reasoning"

const PI_THINKING_SURFACE = {
  controls: ["off", "minimal", "low", "medium", "high", "xhigh", "max"],
  off: "off",
  soleEnabled: "high",
  aliases: { medium: "adaptive" },
} as const

export const piModels = (models: HarnessConnectionSpec["models"]) => models.map((model) => {
  const reasoning = hasReasoning(model)
  return {
    id: model.id,
    name: model.name,
    reasoning,
    thinkingLevelMap: projectReasoningControls(model, PI_THINKING_SURFACE).map,
    input: modelInput(model),
    cost: zeroCost(),
    contextWindow: model.contextWindow,
    maxTokens: modelMaxTokens(model),
    compat: { supportsReasoningEffort: reasoning },
  }
})

export const piProviderConfig = (models: HarnessConnectionSpec["models"]) => ({
  baseUrl: OPENAI_BASE_URL,
  api: CHAT_COMPLETIONS_API,
  apiKey: LOCAL_TOKEN,
  models: piModels(models),
})

export const makePiConnector = (paths: HarnessConnectionPaths) => defineConnector({
  id: "pi",
  name: "Pi",
  executable: "pi",
  skillInstallationTarget: "shared-agents",
  configurationFiles: [paths.piModels],
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.piModels, "{}\n")
    yield* writeIfChanged(paths.piModels, source, updateJsonc(source, [[
      ["providers", "magnitude"], piProviderConfig(spec.models),
    ]]))
  }),
  disconnect: () => Effect.gen(function* () {
    const source = yield* readOr(paths.piModels, "{}\n")
    yield* writeIfChanged(paths.piModels, source, removeJsoncPaths(source, [["providers", "magnitude"]]))
  }),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--model", `magnitude/${modelId}`])
  },
})
