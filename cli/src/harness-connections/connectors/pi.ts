import { Effect, Option } from "effect"
import type { HarnessConnectionPaths } from "../paths"
import {
  CHAT_COMPLETIONS_API,
  LOCAL_TOKEN,
  OPENAI_BASE_URL,
  defineConnector,
  jsonObject,
  launchPlan,
  modelEntries,
  ownedVariant,
  readOr,
  removeOwnedJsonc,
  updateJsonc,
  valueAt,
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
  skillFile: paths.skills.pi!,
  configurationFiles: [paths.piModels, paths.piSettings],
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.piModels, "{}\n")
    const existing = valueAt(jsonObject(source), ["providers", "magnitude"])
    if (existing !== undefined && (
      typeof existing !== "object"
      || valueAt(existing, ["baseUrl"]) !== OPENAI_BASE_URL
      || valueAt(existing, ["api"]) !== CHAT_COMPLETIONS_API
    )) throw new Error("Pi already contains a conflicting providers.magnitude")
    yield* writeIfChanged(paths.piModels, source, updateJsonc(source, [[
      ["providers", "magnitude"], piProviderConfig(spec.models),
    ]]))
    if (Option.isSome(spec.setCurrent)) {
      const settings = yield* readOr(paths.piSettings, "{}\n")
      yield* writeIfChanged(paths.piSettings, settings, updateJsonc(settings, [
        [["defaultProvider"], "magnitude"],
        [["defaultModel"], spec.setCurrent.value],
      ]))
    }
  }),
  disconnect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.piModels, "{}\n")
    const provider = valueAt(jsonObject(source), ["providers", "magnitude"])
    yield* writeIfChanged(paths.piModels, source, removeOwnedJsonc(source, ownedVariant(
      ["providers", "magnitude"],
      provider,
      [piProviderConfig(spec.models)],
    )))
    const settings = yield* readOr(paths.piSettings, "{}\n")
    const owned = Option.match(spec.setCurrent, {
      onNone: () => [],
      onSome: (modelId) => [
        [["defaultProvider"], "magnitude"] as const,
        [["defaultModel"], modelId] as const,
      ],
    })
    yield* writeIfChanged(paths.piSettings, settings, removeOwnedJsonc(settings, owned))
  }),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--model", `magnitude/${modelId}`])
  },
})
