import { Effect, Option } from "effect"
import { parseDocument } from "yaml"
import type { HarnessConnectionPaths } from "../paths"
import {
  CHAT_COMPLETIONS_API,
  OPENAI_BASE_URL,
  defineConnector,
  launchPlan,
  readOr,
  removeYamlPaths,
  updateYaml,
  valueAt,
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
  configurationFiles: [paths.ompModels, paths.ompSettings],
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.ompModels, "{}\n")
    yield* writeIfChanged(paths.ompModels, source, updateYaml(source, [[
      ["providers", "magnitude"], ohMyPiProviderConfig(spec.models),
    ]]))
    if (Option.isNone(spec.model)) return Option.none()
    const settingsSource = yield* readOr(paths.ompSettings, "{}\n")
    const document = parseDocument(settingsSource.trim() === "" ? "{}\n" : settingsSource)
    if (document.errors.length > 0) throw new Error("configuration is not valid YAML")
    const previous = valueAt(document.toJS(), ["modelRoles", "default"])
    yield* writeIfChanged(paths.ompSettings, settingsSource, updateYaml(settingsSource, [[
      ["modelRoles", "default"], `magnitude/${spec.model.value}`,
    ]]))
    return Option.some({
      model: typeof previous === "string" ? Option.some(previous) : Option.none(),
    })
  }),
  disconnect: (spec) => Effect.gen(function* () {
    const settingsSource = yield* readOr(paths.ompSettings, "{}\n")
    const document = parseDocument(settingsSource.trim() === "" ? "{}\n" : settingsSource)
    if (document.errors.length > 0) throw new Error("configuration is not valid YAML")
    const current = valueAt(document.toJS(), ["modelRoles", "default"])
    if (typeof current === "string" && current.startsWith("magnitude/") && Option.isSome(spec.restore)) {
      const next = Option.match(spec.restore.value.model, {
        onNone: () => removeYamlPaths(settingsSource, [["modelRoles", "default"]]),
        onSome: (model) => updateYaml(settingsSource, [[["modelRoles", "default"], model]]),
      })
      yield* writeIfChanged(paths.ompSettings, settingsSource, next)
    }
    const source = yield* readOr(paths.ompModels, "{}\n")
    yield* writeIfChanged(paths.ompModels, source, removeYamlPaths(source, [["providers", "magnitude"]]))
  }),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--model", `magnitude/${modelId}`])
  },
})
