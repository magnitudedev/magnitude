import { Effect } from "effect"
import type { HarnessConnectionPaths } from "../paths"
import {
  LOCAL_TOKEN,
  OPENAI_BASE_URL,
  defineConnector,
  launchPlan,
  readOr,
  removeOwnedYaml,
  removeYamlPaths,
  updateYaml,
  valueAt,
  writeIfChanged,
} from "../shared"
import { parseDocument } from "yaml"
import type { HarnessConnectionSpec } from "../contract"

export const hermesProviderConfig = () => ({
  name: "Magnitude",
  base_url: OPENAI_BASE_URL,
  api_key: LOCAL_TOKEN,
  transport: "chat_completions",
})

export const hermesReasoningOverrides = (models: HarnessConnectionSpec["models"]) => Object.fromEntries(
  models.map((model) => [
    model.id,
    model.capabilities.reasoning.supported ? model.capabilities.reasoning.defaultEffort : "none",
  ]),
)

export const makeHermesConnector = (paths: HarnessConnectionPaths) => defineConnector({
  id: "hermes",
  name: "Hermes",
  executable: "hermes",
  skillInstallationTarget: "hermes-user",
  configurationFiles: [paths.hermes],
  connect: (spec) => Effect.gen(function* () {
    const original = yield* readOr(paths.hermes, "{}\n")
    const source = removeOwnedYaml(original, Object.entries(hermesReasoningOverrides(spec.previousModels ?? [])).map(
      ([modelId, effort]) => [["agent", "reasoning_overrides", modelId], effort] as const,
    ))
    const existing = parseDocument(source.trim() === "" ? "{}\n" : source).toJS()
    const changes: Array<readonly [ReadonlyArray<string>, unknown]> = [[
      ["providers", "magnitude"], hermesProviderConfig(),
    ]]
    for (const [modelId, effort] of Object.entries(hermesReasoningOverrides(spec.models))) {
      const current = valueAt(existing, ["agent", "reasoning_overrides", modelId])
      if (current === undefined) {
        changes.push([["agent", "reasoning_overrides", modelId], effort])
      }
    }
    yield* writeIfChanged(paths.hermes, original, updateYaml(source, changes))
  }),
  disconnect: (spec) => readOr(paths.hermes, "{}\n").pipe(Effect.flatMap((source) => {
    const withoutOverrides = removeOwnedYaml(source, Object.entries(hermesReasoningOverrides(spec.models)).map(
      ([modelId, effort]) => [["agent", "reasoning_overrides", modelId], effort] as const,
    ))
    return writeIfChanged(
      paths.hermes,
      source,
      removeYamlPaths(withoutOverrides, [["providers", "magnitude"]]),
    )
  })),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--provider", "custom:magnitude", "--model", modelId])
  },
})
