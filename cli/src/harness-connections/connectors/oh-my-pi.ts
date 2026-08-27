import { Effect, Option } from "effect"
import { parseDocument } from "yaml"
import type { HarnessConnectionPaths } from "../paths"
import {
  CHAT_COMPLETIONS_API,
  OPENAI_BASE_URL,
  defineConnector,
  launchPlan,
  modelEntries,
  ownedVariant,
  readOr,
  removeOwnedJsonc,
  removeOwnedYaml,
  updateJsonc,
  updateYaml,
  valueAt,
  writeIfChanged,
} from "../shared"

export const ohMyPiProviderConfig = (models: Parameters<typeof modelEntries>[0]) => ({
  baseUrl: OPENAI_BASE_URL,
  auth: "none",
  api: CHAT_COMPLETIONS_API,
  models: modelEntries(models),
})

export const makeOhMyPiConnector = (paths: HarnessConnectionPaths) => defineConnector({
  id: "oh-my-pi",
  name: "Oh My Pi",
  executable: "omp",
  skillInstallationTarget: "shared-agents",
  configurationFiles: [paths.ompModels, paths.ompSettings],
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.ompModels, "{}\n")
    const document = parseDocument(source)
    if (document.errors.length > 0) throw new Error("Oh My Pi models configuration is not valid YAML")
    const existing = document.getIn(["providers", "magnitude"])
    if (existing !== undefined && (
      document.getIn(["providers", "magnitude", "baseUrl"]) !== OPENAI_BASE_URL
      || document.getIn(["providers", "magnitude", "api"]) !== CHAT_COMPLETIONS_API
    )) throw new Error("Oh My Pi already contains a conflicting Magnitude provider")
    yield* writeIfChanged(paths.ompModels, source, updateYaml(source, [[
      ["providers", "magnitude"], ohMyPiProviderConfig(spec.models),
    ]]))
    if (Option.isSome(spec.setCurrent)) {
      const settings = yield* readOr(paths.ompSettings, "{}\n")
      yield* writeIfChanged(paths.ompSettings, settings, updateJsonc(settings, [[
        ["model"], `magnitude/${spec.setCurrent.value}`,
      ]]))
    }
  }),
  disconnect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.ompModels, "{}\n")
    const provider = valueAt(parseDocument(source).toJS(), ["providers", "magnitude"])
    yield* writeIfChanged(paths.ompModels, source, removeOwnedYaml(source, ownedVariant(
      ["providers", "magnitude"],
      provider,
      [ohMyPiProviderConfig(spec.models)],
    )))
    const settings = yield* readOr(paths.ompSettings, "{}\n")
    yield* writeIfChanged(paths.ompSettings, settings, removeOwnedJsonc(settings, Option.match(spec.setCurrent, {
      onNone: () => [],
      onSome: (modelId) => [[["model"], `magnitude/${modelId}`] as const],
    })))
  }),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--model", `magnitude/${modelId}`])
  },
})
