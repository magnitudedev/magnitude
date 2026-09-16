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

export const openCodeProviderConfig = (models: HarnessConnectionSpec["models"], baseUrl = OPENAI_BASE_URL) => ({
  npm: OPENAI_COMPATIBLE_PACKAGE,
  name: "Magnitude",
  options: { baseURL: baseUrl, apiKey: LOCAL_TOKEN },
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

const isObject = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)

// OpenCode recursively merges objects; higher-priority scalar/array values replace lower ones.
const mergeConfiguration = (left: Record<string, unknown>, right: Record<string, unknown>): Record<string, unknown> =>
  Object.fromEntries([...new Set([...Object.keys(left), ...Object.keys(right)])].map(key => {
    const previous = left[key]
    const next = right[key]
    return [key, Object.hasOwn(right, key)
      ? isObject(previous) && isObject(next) ? mergeConfiguration(previous, next) : next
      : previous]
  }))

export const openCodeConfigurationFiles = (paths: HarnessConnectionPaths) => paths.opencodeFiles ?? [paths.opencode]

export const readOpenCodeConfiguration = (paths: HarnessConnectionPaths) => Effect.gen(function* () {
  let merged: Record<string, unknown> = {}
  for (const file of openCodeConfigurationFiles(paths)) {
    const source = yield* readOr(file, "{}\n")
    const document = yield* Effect.try(() => jsonObject(source))
    merged = mergeConfiguration(merged, document)
  }
  return merged
})

export const makeOpenCodeConnector = (paths: HarnessConnectionPaths, baseUrl = OPENAI_BASE_URL) => defineConnector({
  id: "opencode",
  name: "OpenCode",
  executable: "opencode",
  skillInstallationTarget: "shared-agents",
  configurationFiles: openCodeConfigurationFiles(paths),
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.opencode, "{}\n")
    const previous = valueAt(yield* readOpenCodeConfiguration(paths), ["model"])
    const changes: Array<readonly [ReadonlyArray<string>, unknown]> = [[
      ["provider", "magnitude"], openCodeProviderConfig(spec.models, baseUrl),
    ]]
    if (Option.isSome(spec.model)) changes.push([["model"], `magnitude/${spec.model.value}`])
    yield* writeIfChanged(paths.opencode, source, updateJsonc(source, changes))
    return Option.map(spec.model, () => ({
      model: typeof previous === "string" ? Option.some(previous) : Option.none(),
    }))
  }),
  disconnect: (spec) => Effect.gen(function* () {
    const current = valueAt(yield* readOpenCodeConfiguration(paths), ["model"])
    const restore = typeof current === "string" && current.startsWith("magnitude/") && Option.isSome(spec.restore)
      ? spec.restore : Option.none()
    for (const file of openCodeConfigurationFiles(paths)) {
      const source = yield* readOr(file, "{}\n")
      const document = jsonObject(source)
      const model = valueAt(document, ["model"])
      let next = valueAt(document, ["provider", "magnitude"]) === undefined
        ? source : removeJsoncPaths(source, [["provider", "magnitude"]])
      if (typeof model === "string" && model.startsWith("magnitude/") && Option.isSome(spec.restore)) {
        next = removeJsoncPaths(next, [["model"]])
      }
      if (file === paths.opencode && Option.isSome(restore) && Option.isSome(restore.value.model)) {
        next = updateJsonc(next, [[["model"], restore.value.model.value]])
      }
      yield* writeIfChanged(file, source, next)
    }
  }),
})
