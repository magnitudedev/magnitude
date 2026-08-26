import * as FileSystem from "@effect/platform/FileSystem"
import { Effect, Option } from "effect"
import type { HarnessConnectionSpec } from "../contract"
import type { HarnessConnectionPaths } from "../paths"
import { OPENAI_BASE_URL, defineConnector, launchPlan, readOr } from "../shared"
import { writeFileAtomic } from "../../utils/atomic-file"

export const codexConfig = (spec: HarnessConnectionSpec): string => `model_provider = "magnitude"
${Option.match(spec.setCurrent, {
  onNone: () => "\n",
  onSome: (modelId) => `model = ${JSON.stringify(modelId)}\n\n`,
})}[model_providers.magnitude]
name = "Magnitude"
base_url = ${JSON.stringify(OPENAI_BASE_URL)}
wire_api = "responses"
requires_openai_auth = false
`

export const makeCodexConnector = (paths: HarnessConnectionPaths) => defineConnector({
  id: "codex",
  name: "Codex",
  executable: "codex",
  skillFile: paths.skills.codex!,
  configurationFiles: [paths.codex],
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.codex, "")
    if (source.trim() !== "" && !source.includes("[model_providers.magnitude]")) {
      throw new Error(`Codex profile file is not Magnitude-owned: ${paths.codex}`)
    }
    yield* writeFileAtomic(paths.codex, codexConfig(spec))
  }),
  disconnect: (spec) => FileSystem.FileSystem.pipe(
    Effect.flatMap((fs) => fs.readFileString(paths.codex).pipe(
      Effect.flatMap((source) => source === codexConfig(spec) ? fs.remove(paths.codex) : Effect.void),
      Effect.catchTag("SystemError", (error) => error.reason === "NotFound" ? Effect.void : Effect.fail(error)),
    )),
  ),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--profile", "magnitude"])
  },
})
