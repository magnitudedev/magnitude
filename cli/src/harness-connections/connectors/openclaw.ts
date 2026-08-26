import { Effect, Option } from "effect"
import { isDeepStrictEqual } from "node:util"
import type { HarnessConnectionPaths } from "../paths"
import {
  CHAT_COMPLETIONS_API,
  LOCAL_TOKEN,
  OPENAI_BASE_URL,
  defineConnector,
  jsonObject,
  launchPlan,
  modelEntries,
  ownedVariant,
  readOr,
  removeOwnedJsonc,
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
  skillFile: paths.skills.openclaw!,
  configurationFiles: [paths.openclaw],
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.openclaw, "{}\n")
    const value = jsonObject(source)
    const existingProvider = valueAt(value, ["models", "providers", "magnitude"])
    if (existingProvider !== undefined && (
      valueAt(existingProvider, ["baseUrl"]) !== OPENAI_BASE_URL
      || valueAt(existingProvider, ["api"]) !== CHAT_COMPLETIONS_API
    )) throw new Error("OpenClaw already contains a conflicting Magnitude provider")
    const existingAgents = valueAt(value, ["agents", "list"])
    if (existingAgents !== undefined && !Array.isArray(existingAgents)) {
      throw new Error("OpenClaw agents.list is not an array")
    }
    const agents = (existingAgents ?? []) as ReadonlyArray<unknown>
    const existingAgent = agents.find((entry) => valueAt(entry, ["id"]) === "magnitude")
    const desiredAgent = Option.map(spec.setCurrent, openClawAgentConfig)
    if (existingAgent !== undefined && (
      Option.isNone(desiredAgent) || !isDeepStrictEqual(existingAgent, desiredAgent.value)
    )) throw new Error("OpenClaw already contains a conflicting magnitude agent")
    const changes: Array<readonly [ReadonlyArray<string>, unknown]> = [[
      ["models", "providers", "magnitude"], openClawProviderConfig(spec.models),
    ]]
    if (Option.isSome(desiredAgent)) changes.push([
      ["agents", "list"],
      [...agents.filter((entry) => valueAt(entry, ["id"]) !== "magnitude"), desiredAgent.value],
    ])
    yield* writeIfChanged(paths.openclaw, source, updateJsonc(source, changes))
  }),
  disconnect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.openclaw, "{}\n")
    const value = jsonObject(source)
    const provider = valueAt(value, ["models", "providers", "magnitude"])
    const withoutProvider = removeOwnedJsonc(source, [
      ...ownedVariant(
        ["models", "providers", "magnitude"],
        provider,
        [openClawProviderConfig(spec.models)],
      ),
      ...Option.match(spec.setCurrent, {
        onNone: () => [],
        onSome: (modelId) => [[
          ["agents", "defaults", "model", "primary"], `magnitude/${modelId}`,
        ] as const],
      }),
    ])
    const agents = valueAt(value, ["agents", "list"])
    if (!Array.isArray(agents) || Option.isNone(spec.setCurrent)) {
      yield* writeIfChanged(paths.openclaw, source, withoutProvider)
      return
    }
    const desiredAgent = openClawAgentConfig(spec.setCurrent.value)
    const nextAgents = agents.filter((entry) => !isDeepStrictEqual(entry, desiredAgent))
    const next = nextAgents.length === agents.length
      ? withoutProvider
      : updateJsonc(withoutProvider, [[["agents", "list"], nextAgents]])
    yield* writeIfChanged(paths.openclaw, source, next)
  }),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["tui", "--local", "--session", "agent:magnitude:main"])
  },
})
