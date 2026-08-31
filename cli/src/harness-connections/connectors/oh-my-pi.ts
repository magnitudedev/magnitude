import { Effect } from "effect"
import type { HarnessConnectionPaths } from "../paths"
import {
  CHAT_COMPLETIONS_API,
  OPENAI_BASE_URL,
  defineConnector,
  launchPlan,
  readOr,
  removeYamlPaths,
  updateYaml,
  writeIfChanged,
} from "../shared"
import type { HarnessConnectionSpec } from "../contract"
import { modelInput, modelMaxTokens, zeroCost } from "../model-fields"
import { hasReasoning, projectReasoningControls, supportsReasoningEffort } from "../reasoning"

const OMP_EFFORT_SURFACE = {
  controls: ["minimal", "low", "medium", "high", "xhigh", "max"],
  soleEnabled: "high",
  aliases: { medium: "adaptive" },
} as const

export const ohMyPiModels = (models: HarnessConnectionSpec["models"]) => models.map((model) => {
  const projection = projectReasoningControls(model, OMP_EFFORT_SURFACE)
  const effortMap = Object.fromEntries(Object.entries(projection.map).filter((entry): entry is [string, string] =>
    entry[1] !== null))
  const efforts = Object.keys(effortMap)
  return {
    id: model.id,
    name: model.name,
    reasoning: hasReasoning(model),
    ...(efforts.length === 0 ? {} : {
      thinking: {
        mode: "effort",
        efforts,
        ...(projection.defaultControl === undefined ? {} : { defaultLevel: projection.defaultControl }),
        effortMap,
        requiresEffort: !supportsReasoningEffort(model, "none"),
      },
    }),
    input: modelInput(model),
    cost: zeroCost(),
    contextWindow: model.contextWindow,
    maxTokens: modelMaxTokens(model),
  }
})

export const ohMyPiProviderConfig = (models: HarnessConnectionSpec["models"]) => ({
  baseUrl: OPENAI_BASE_URL,
  auth: "none",
  api: CHAT_COMPLETIONS_API,
  models: ohMyPiModels(models),
})

export const makeOhMyPiConnector = (paths: HarnessConnectionPaths) => defineConnector({
  id: "oh-my-pi",
  name: "Oh My Pi",
  executable: "omp",
  skillInstallationTarget: "shared-agents",
  configurationFiles: [paths.ompModels],
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.ompModels, "{}\n")
    yield* writeIfChanged(paths.ompModels, source, updateYaml(source, [[
      ["providers", "magnitude"], ohMyPiProviderConfig(spec.models),
    ]]))
  }),
  disconnect: () => Effect.gen(function* () {
    const source = yield* readOr(paths.ompModels, "{}\n")
    yield* writeIfChanged(paths.ompModels, source, removeYamlPaths(source, [["providers", "magnitude"]]))
  }),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--model", `magnitude/${modelId}`])
  },
})
