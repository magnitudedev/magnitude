import { Effect } from "effect"
import type { HarnessConnectionPaths } from "../paths"
import {
  CHAT_COMPLETIONS_API,
  LOCAL_TOKEN,
  OPENAI_BASE_URL,
  defineConnector,
  launchPlan,
  modelEntries,
  readOr,
  removeJsoncPaths,
  updateJsonc,
  writeIfChanged,
} from "../shared"

export const piProviderConfig = (models: Parameters<typeof modelEntries>[0]) => ({
  baseUrl: OPENAI_BASE_URL,
  api: CHAT_COMPLETIONS_API,
  apiKey: LOCAL_TOKEN,
  models: modelEntries(models),
})

export const makePiConnector = (paths: HarnessConnectionPaths) => defineConnector({
  id: "pi",
  name: "Pi",
  executable: "pi",
  skillInstallationTarget: "shared-agents",
  configurationFiles: [paths.piModels],
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.piModels, "{}\n")
    yield* writeIfChanged(paths.piModels, source, updateJsonc(source, [[
      ["providers", "magnitude"], piProviderConfig(spec.models),
    ]]))
  }),
  disconnect: () => Effect.gen(function* () {
    const source = yield* readOr(paths.piModels, "{}\n")
    yield* writeIfChanged(paths.piModels, source, removeJsoncPaths(source, [["providers", "magnitude"]]))
  }),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--model", `magnitude/${modelId}`])
  },
})
