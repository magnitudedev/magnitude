import { Effect, Option } from "effect"
import { randomUUID } from "node:crypto"
import type { HarnessConnectionPaths } from "../paths"
import {
  CHAT_COMPLETIONS_API,
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

export const openClawProviderConfig = (models: HarnessConnectionSpec["models"]) => ({
  baseUrl: OPENAI_BASE_URL,
  apiKey: LOCAL_TOKEN,
  api: CHAT_COMPLETIONS_API,
  models: openClawModels(models),
})

export const openClawAgentConfig = (model: HarnessConnectionSpec["models"][number]) => {
  const { defaultControl } = openClawReasoning(model)
  return {
    id: "magnitude",
    model: `magnitude/${model.id}`,
    ...(defaultControl === undefined ? {} : { thinkingDefault: defaultControl }),
  }
}

export const makeOpenClawConnector = (paths: HarnessConnectionPaths) => defineConnector({
  id: "openclaw",
  name: "OpenClaw",
  executable: "openclaw",
  skillInstallationTarget: "shared-agents",
  configurationFiles: [paths.openclaw],
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.openclaw, "{}\n")
    const value = jsonObject(source)
    const existingAgents = valueAt(value, ["agents", "list"])
    const agents: ReadonlyArray<unknown> = Array.isArray(existingAgents) ? existingAgents : []
    const changes: Array<readonly [ReadonlyArray<string>, unknown]> = [[
      ["models", "providers", "magnitude"], openClawProviderConfig(spec.models),
    ]]
    const previous = valueAt(value, ["agents", "defaults", "model", "primary"])
    if (Option.isSome(spec.model)) {
      const selectedModelId = spec.model.value
      const selected = spec.models.find((model) => model.id === selectedModelId)
      if (selected !== undefined) changes.push(
        [["agents", "defaults", "model", "primary"], `magnitude/${selectedModelId}`],
        [["agents", "list"], [
          ...agents.filter((entry) => valueAt(entry, ["id"]) !== "magnitude"),
          openClawAgentConfig(selected),
        ]],
      )
    }
    yield* writeIfChanged(paths.openclaw, source, updateJsonc(source, changes))
    return Option.map(spec.model, () => ({
      model: typeof previous === "string" ? Option.some(previous) : Option.none(),
    }))
  }),
  disconnect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.openclaw, "{}\n")
    const value = jsonObject(source)
    const agents = valueAt(value, ["agents", "list"])
    const current = valueAt(value, ["agents", "defaults", "model", "primary"])
    const withoutProvider = removeJsoncPaths(source, [["models", "providers", "magnitude"]])
    const withoutAgent = !Array.isArray(agents)
      ? withoutProvider
      : updateJsonc(withoutProvider, [[
          ["agents", "list"], agents.filter((entry) => valueAt(entry, ["id"]) !== "magnitude"),
        ]])
    const next = typeof current === "string" && current.startsWith("magnitude/") && Option.isSome(spec.restore)
      ? updateJsonc(withoutAgent, [[
          ["agents", "defaults", "model", "primary"], Option.getOrUndefined(spec.restore.value.model),
        ]])
      : withoutAgent
    yield* writeIfChanged(paths.openclaw, source, next)
  }),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, [
      "tui",
      "--local",
      "--session",
      `agent:magnitude:${randomUUID()}`,
    ])
  },
})
