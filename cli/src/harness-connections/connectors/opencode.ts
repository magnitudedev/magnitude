import { Effect, Option } from "effect"
import type { HarnessConnectionSpec } from "../contract"
import type { HarnessConnectionPaths } from "../paths"
import { modelInput, modelMaxTokens } from "../model-fields"
import {
  LOCAL_TOKEN,
  OPENAI_BASE_URL,
  OPENAI_COMPATIBLE_PACKAGE,
  defineConnector,
  jsonObject,
  launchPlan,
  readOr,
  removeJsoncPaths,
  updateJsonc,
  valueAt,
  writeIfChanged,
} from "../shared"

const reasoningVariants = (model: HarnessConnectionSpec["models"][number]) => Object.fromEntries(
  model.capabilities.reasoning.efforts.map((effort) => [
    effort === "none" ? "off" : effort,
    { reasoningEffort: effort },
  ]),
)

export const openCodeProviderConfig = (models: HarnessConnectionSpec["models"]) => ({
  npm: OPENAI_COMPATIBLE_PACKAGE,
  name: "Magnitude",
  options: { baseURL: OPENAI_BASE_URL, apiKey: LOCAL_TOKEN },
  models: Object.fromEntries(models.map((model) => [model.id, {
    name: model.name,
    limit: { context: model.contextWindow, output: modelMaxTokens(model) },
    modalities: {
      input: modelInput(model),
      output: ["text"],
    },
    variants: reasoningVariants(model),
  }])),
})

export const makeOpenCodeConnector = (paths: HarnessConnectionPaths) => defineConnector({
  id: "opencode",
  name: "OpenCode",
  executable: "opencode",
  skillInstallationTarget: "shared-agents",
  configurationFiles: [paths.opencode],
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.opencode, "{}\n")
    const previous = valueAt(jsonObject(source), ["model"])
    const changes: Array<readonly [ReadonlyArray<string>, unknown]> = [[
      ["provider", "magnitude"], openCodeProviderConfig(spec.models),
    ]]
    if (Option.isSome(spec.model)) changes.push([["model"], `magnitude/${spec.model.value}`])
    yield* writeIfChanged(paths.opencode, source, updateJsonc(source, changes))
    return Option.map(spec.model, () => ({
      model: typeof previous === "string" ? Option.some(previous) : Option.none(),
    }))
  }),
  disconnect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.opencode, "{}\n")
    const current = valueAt(jsonObject(source), ["model"])
    const withoutProvider = removeJsoncPaths(source, [["provider", "magnitude"]])
    const next = typeof current === "string" && current.startsWith("magnitude/") && Option.isSome(spec.restore)
      ? updateJsonc(withoutProvider, [[
          ["model"], Option.getOrUndefined(spec.restore.value.model),
        ]])
      : withoutProvider
    yield* writeIfChanged(paths.opencode, source, next)
  }),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--model", `magnitude/${modelId}`])
  },
})
