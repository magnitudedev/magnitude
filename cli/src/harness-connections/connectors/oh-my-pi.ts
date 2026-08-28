import { Effect } from "effect"
import type { HarnessConnectionPaths } from "../paths"
import {
  CHAT_COMPLETIONS_API,
  OPENAI_BASE_URL,
  defineConnector,
  launchPlan,
  modelEntries,
  readOr,
  removeYamlPaths,
  updateYaml,
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
