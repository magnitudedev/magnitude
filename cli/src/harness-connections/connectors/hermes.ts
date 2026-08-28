import { Effect } from "effect"
import type { HarnessConnectionPaths } from "../paths"
import {
  LOCAL_TOKEN,
  OPENAI_BASE_URL,
  defineConnector,
  launchPlan,
  readOr,
  removeYamlPaths,
  updateYaml,
  writeIfChanged,
} from "../shared"

export const hermesProviderConfig = () => ({
  name: "Magnitude",
  base_url: OPENAI_BASE_URL,
  api_key: LOCAL_TOKEN,
  transport: "chat_completions",
})

export const makeHermesConnector = (paths: HarnessConnectionPaths) => defineConnector({
  id: "hermes",
  name: "Hermes",
  executable: "hermes",
  skillInstallationTarget: "hermes-user",
  configurationFiles: [paths.hermes],
  connect: () => Effect.gen(function* () {
    const source = yield* readOr(paths.hermes, "{}\n")
    yield* writeIfChanged(paths.hermes, source, updateYaml(source, [[
      ["providers", "magnitude"], hermesProviderConfig(),
    ]]))
  }),
  disconnect: () => readOr(paths.hermes, "{}\n").pipe(Effect.flatMap((source) =>
    writeIfChanged(paths.hermes, source, removeYamlPaths(source, [["providers", "magnitude"]])))),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--provider", "custom:magnitude", "--model", modelId])
  },
})
