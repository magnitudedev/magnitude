import { Effect, Option } from "effect"
import type { HarnessConnectionPaths } from "../paths"
import {
  CHAT_COMPLETIONS_API,
  LOCAL_TOKEN,
  OPENAI_BASE_URL,
  defineConnector,
  jsonObject,
  launchPlan,
  modelEntries,
  readOr,
  removeJsoncPaths,
  updateJsonc,
  valueAt,
  writeIfChanged,
} from "../shared"

export const openClawProviderConfig = (models: Parameters<typeof modelEntries>[0]) => ({
  baseUrl: OPENAI_BASE_URL,
  apiKey: LOCAL_TOKEN,
  api: CHAT_COMPLETIONS_API,
  models: modelEntries(models),
})

export const openClawAgentConfig = (modelId: string) => ({
  id: "magnitude",
  model: `magnitude/${modelId}`,
})

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
    if (Option.isSome(spec.setCurrent)) changes.push([
      ["agents", "list"],
      [
        ...agents.filter((entry) => valueAt(entry, ["id"]) !== "magnitude"),
        openClawAgentConfig(spec.setCurrent.value),
      ],
    ])
    yield* writeIfChanged(paths.openclaw, source, updateJsonc(source, changes))
  }),
  disconnect: () => Effect.gen(function* () {
    const source = yield* readOr(paths.openclaw, "{}\n")
    const value = jsonObject(source)
    const agents = valueAt(value, ["agents", "list"])
    const withoutProvider = removeJsoncPaths(source, [["models", "providers", "magnitude"]])
    const next = !Array.isArray(agents)
      ? withoutProvider
      : updateJsonc(withoutProvider, [[
          ["agents", "list"], agents.filter((entry) => valueAt(entry, ["id"]) !== "magnitude"),
        ]])
    yield* writeIfChanged(paths.openclaw, source, next)
  }),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["tui", "--local", "--session", "agent:magnitude:main"])
  },
})
