import { Effect, Option } from "effect"
import type { HarnessConnectionPaths } from "../paths"
import {
  LOCAL_TOKEN,
  OPENAI_BASE_URL,
  defineConnector,
  launchPlan,
  readOr,
  qualifiedModelSelection,
  removeOwnedYaml,
  removeYamlPaths,
  splitQualifiedModelSelection,
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
    const restore = Option.map(spec.model, () => ({
      model: qualifiedModelSelection(
        valueAt(existing, ["model", "provider"]),
        valueAt(existing, ["model", "default"]),
      ),
    }))
    if (Option.isSome(spec.model)) {
      changes.push(
        [["model", "provider"], "custom:magnitude"],
        [["model", "default"], spec.model.value],
      )
    }
    for (const [modelId, effort] of Object.entries(hermesReasoningOverrides(spec.models))) {
      const current = valueAt(existing, ["agent", "reasoning_overrides", modelId])
      if (current === undefined) {
        changes.push([["agent", "reasoning_overrides", modelId], effort])
      }
    }
    yield* writeIfChanged(paths.hermes, original, updateYaml(source, changes))
    return restore
  }),
  disconnect: (spec) => readOr(paths.hermes, "{}\n").pipe(Effect.flatMap((source) => {
    const withoutOverrides = removeOwnedYaml(source, Object.entries(hermesReasoningOverrides(spec.models)).map(
      ([modelId, effort]) => [["agent", "reasoning_overrides", modelId], effort] as const,
    ))
    const withoutProvider = removeYamlPaths(withoutOverrides, [["providers", "magnitude"]])
    const current = parseDocument(source.trim() === "" ? "{}\n" : source).toJS()
    if (valueAt(current, ["model", "provider"]) !== "custom:magnitude" || Option.isNone(spec.restore)) {
      return writeIfChanged(paths.hermes, source, withoutProvider)
    }
    const previous = Option.flatMap(spec.restore.value.model, (selection) =>
      Option.fromNullable(splitQualifiedModelSelection(selection)))
    const restored = Option.match(previous, {
      onNone: () => removeYamlPaths(withoutProvider, [["model", "provider"], ["model", "default"]]),
      onSome: ({ provider, model }) => updateYaml(withoutProvider, [
        [["model", "provider"], provider],
        [["model", "default"], model],
      ]),
    })
    return writeIfChanged(
      paths.hermes,
      source,
      restored,
    )
  })),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--provider", "custom:magnitude", "--model", modelId])
  },
})
