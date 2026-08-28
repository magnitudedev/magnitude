import { Effect } from "effect"
import type { HarnessConnectionSpec } from "../contract"
import type { HarnessConnectionPaths } from "../paths"
import {
  LOCAL_TOKEN,
  OPENAI_BASE_URL,
  OPENAI_COMPATIBLE_PACKAGE,
  defineConnector,
  launchPlan,
  readOr,
  removeJsoncPaths,
  updateJsonc,
  writeIfChanged,
} from "../shared"

export const openCodeProviderConfig = (models: HarnessConnectionSpec["models"]) => ({
  npm: OPENAI_COMPATIBLE_PACKAGE,
  name: "Magnitude",
  options: { baseURL: OPENAI_BASE_URL, apiKey: LOCAL_TOKEN },
  models: Object.fromEntries(models.map(({ id, name }) => [id, { name }])),
})

export const makeOpenCodeConnector = (paths: HarnessConnectionPaths) => defineConnector({
  id: "opencode",
  name: "OpenCode",
  executable: "opencode",
  skillInstallationTarget: "shared-agents",
  configurationFiles: [paths.opencode],
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.opencode, "{}\n")
    yield* writeIfChanged(paths.opencode, source, updateJsonc(source, [[
      ["provider", "magnitude"], openCodeProviderConfig(spec.models),
    ]]))
  }),
  disconnect: () => readOr(paths.opencode, "{}\n").pipe(Effect.flatMap((source) =>
    writeIfChanged(paths.opencode, source, removeJsoncPaths(source, [["provider", "magnitude"]])))),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--model", `magnitude/${modelId}`])
  },
})
