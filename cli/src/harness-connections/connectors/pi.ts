import { Effect, Option } from "effect"
import type { HarnessConnectionPaths } from "../paths"
import {
  CHAT_COMPLETIONS_API,
  LOCAL_TOKEN,
  OPENAI_BASE_URL,
  defineConnector,
  jsonObject,
  launchPlan,
  readOr,
  qualifiedModelSelection,
  removeJsoncPaths,
  splitQualifiedModelSelection,
  updateJsonc,
  valueAt,
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
  configurationFiles: [paths.piModels, paths.piSettings],
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.piModels, "{}\n")
    yield* writeIfChanged(paths.piModels, source, updateJsonc(source, [[
      ["providers", "magnitude"], piProviderConfig(spec.models),
    ]]))
    if (Option.isNone(spec.model)) return Option.none()
    const settingsSource = yield* readOr(paths.piSettings, "{}\n")
    const settings = jsonObject(settingsSource)
    const restore = {
      model: qualifiedModelSelection(
        valueAt(settings, ["defaultProvider"]),
        valueAt(settings, ["defaultModel"]),
      ),
    }
    yield* writeIfChanged(paths.piSettings, settingsSource, updateJsonc(settingsSource, [
      [["defaultProvider"], "magnitude"],
      [["defaultModel"], spec.model.value],
    ]))
    return Option.some(restore)
  }),
  disconnect: (spec) => Effect.gen(function* () {
    const settingsSource = yield* readOr(paths.piSettings, "{}\n")
    const settings = jsonObject(settingsSource)
    if (valueAt(settings, ["defaultProvider"]) === "magnitude" && Option.isSome(spec.restore)) {
      const previous = Option.flatMap(spec.restore.value.model, (selection) =>
        Option.fromNullable(splitQualifiedModelSelection(selection)))
      yield* writeIfChanged(paths.piSettings, settingsSource, updateJsonc(settingsSource, [
        [["defaultProvider"], Option.match(previous, { onNone: () => undefined, onSome: ({ provider }) => provider })],
        [["defaultModel"], Option.match(previous, { onNone: () => undefined, onSome: ({ model }) => model })],
      ]))
    }
    const source = yield* readOr(paths.piModels, "{}\n")
    yield* writeIfChanged(paths.piModels, source, removeJsoncPaths(source, [["providers", "magnitude"]]))
  }),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--model", `magnitude/${modelId}`])
  },
})
