import { json5Object, updateJson5 } from "../json5"
import { Effect, Option } from "effect"
import type { HarnessConnectionPaths } from "../paths"
import {
  CHAT_COMPLETIONS_API,
  LOCAL_TOKEN,
  OPENAI_BASE_URL,
  defineConnector,
  readOr,
  valueAt,
  writeIfChanged,
} from "../shared"
import type { HarnessConnectionSpec } from "../contract"
import { modelInput, modelMaxTokens, zeroCost } from "../model-fields"
import { hasReasoning, projectReasoningControls } from "../reasoning"

const OPENCLAW_THINKING_SURFACE = {
  controls: ["off", "minimal", "low", "medium", "high", "xhigh", "max"],
  off: "off",
  soleEnabled: "high",
  aliases: { medium: "adaptive" },
} as const

const openClawReasoning = (model: HarnessConnectionSpec["models"][number]) =>
  projectReasoningControls(model, OPENCLAW_THINKING_SURFACE)

export const openClawModels = (models: HarnessConnectionSpec["models"]) => models.map((model) => {
  const reasoning = hasReasoning(model)
  return {
    id: model.id,
    name: model.name,
    reasoning,
    input: modelInput(model),
    cost: zeroCost(),
    contextWindow: model.contextWindow,
    maxTokens: modelMaxTokens(model),
    thinkingLevelMap: openClawReasoning(model).map,
    compat: {
      supportsReasoningEffort: reasoning,
      supportsTools: model.capabilities.tools,
      supportedReasoningEfforts: model.capabilities.reasoning.efforts,
    },
  }
})

export const openClawProviderConfig = (models: HarnessConnectionSpec["models"], baseUrl = OPENAI_BASE_URL) => ({
  baseUrl,
  apiKey: LOCAL_TOKEN,
  api: CHAT_COMPLETIONS_API,
  models: openClawModels(models),
})

export const openClawAgentConfig = (model: HarnessConnectionSpec["models"][number]) => {
  const { defaultControl } = openClawReasoning(model)
  return {
    model: `magnitude/${model.id}`,
    ...(defaultControl === undefined ? {} : { thinkingDefault: defaultControl }),
  }
}

export const makeOpenClawConnector = (paths: HarnessConnectionPaths, baseUrl = OPENAI_BASE_URL) => defineConnector({
  id: "openclaw",
  name: "OpenClaw",
  executable: "openclaw",
  skillInstallationTarget: "shared-agents",
  configurationFiles: [paths.openclaw],
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.openclaw, "{}\n")
    const value = json5Object(source)
    const entries = valueAt(value, ["agents", "entries"])
    const hasDefault = entries !== null && typeof entries === "object"
      && Object.values(entries).some(entry => valueAt(entry, ["default"]) === true)
    const explicitOwnership = valueAt(value, ["agents", "ownership"]) === "explicit"
    const changes: Array<readonly [ReadonlyArray<string>, unknown]> = [[
      ["models", "providers", "magnitude"], openClawProviderConfig(spec.models, baseUrl),
    ]]
    const previous = valueAt(value, ["agents", "defaults", "model", "primary"])
    if (Option.isSome(spec.model)) {
      const selectedModelId = spec.model.value
      const selected = spec.models.find((model) => model.id === selectedModelId)
      if (selected !== undefined) changes.push(
        [["agents", "defaults", "model", "primary"], `magnitude/${selectedModelId}`],
        [["agents", "entries", "magnitude"], {
          ...openClawAgentConfig(selected),
          ...(!explicitOwnership && entries !== null && typeof entries === "object" && Object.keys(entries).some(key => key !== "magnitude")
            && (!hasDefault || valueAt(entries, ["magnitude", "default"]) === true) ? { default: true } : {}),
        }],
      )
    }
    yield* writeIfChanged(paths.openclaw, source, updateJson5(source, changes))
    return Option.map(spec.model, () => ({
      model: typeof previous === "string" ? Option.some(previous) : Option.none(),
    }))
  }),
  disconnect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.openclaw, "{}\n")
    const value = json5Object(source)
    const current = valueAt(value, ["agents", "defaults", "model", "primary"])
    const withoutProvider = updateJson5(source, [[["models", "providers", "magnitude"], undefined]])
    const withoutAgent = updateJson5(withoutProvider, [[["agents", "entries", "magnitude"], undefined]])
    let next = typeof current === "string" && current.startsWith("magnitude/") && Option.isSome(spec.restore)
      ? updateJson5(withoutAgent, [[
          ["agents", "defaults", "model", "primary"], Option.getOrUndefined(spec.restore.value.model),
        ]])
      : withoutAgent
    for (const parent of [["agents", "entries"], ["agents", "defaults", "model"], ["agents", "defaults"], ["agents"]]) {
      const remaining = valueAt(json5Object(next), parent)
      if (remaining !== null && typeof remaining === "object" && Object.keys(remaining).length === 0) {
        next = updateJson5(next, [[parent, undefined]])
      }
    }
    yield* writeIfChanged(paths.openclaw, source, next)
  }),
})
