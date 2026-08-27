import { Effect, Option } from "effect"
import type { HarnessConnectionSpec } from "../contract"
import type { HarnessConnectionPaths } from "../paths"
import {
  LOCAL_TOKEN,
  OPENAI_BASE_URL,
  OPENAI_COMPATIBLE_PACKAGE,
  defineConnector,
  jsonObject,
  launchPlan,
  ownedVariant,
  readOr,
  removeOwnedJsonc,
  updateJsonc,
  valueAt,
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
    const existing = valueAt(jsonObject(source), ["provider", "magnitude"])
    if (existing !== undefined && (
      typeof existing !== "object"
      || valueAt(existing, ["options", "baseURL"]) !== OPENAI_BASE_URL
      || valueAt(existing, ["npm"]) !== OPENAI_COMPATIBLE_PACKAGE
    )) throw new Error("OpenCode already contains a conflicting provider.magnitude")
    const changes: Array<readonly [ReadonlyArray<string>, unknown]> = [[
      ["provider", "magnitude"], openCodeProviderConfig(spec.models),
    ]]
    if (Option.isSome(spec.setCurrent)) changes.push([["model"], `magnitude/${spec.setCurrent.value}`])
    yield* writeIfChanged(paths.opencode, source, updateJsonc(source, changes))
  }),
  disconnect: (spec) => readOr(paths.opencode, "{}\n").pipe(
    Effect.flatMap((source) => {
      const provider = valueAt(jsonObject(source), ["provider", "magnitude"])
      return writeIfChanged(paths.opencode, source, removeOwnedJsonc(source, [
        ...ownedVariant(["provider", "magnitude"], provider, [openCodeProviderConfig(spec.models)]),
        ...Option.match(spec.setCurrent, {
          onNone: () => [],
          onSome: (modelId) => [[["model"], `magnitude/${modelId}`] as const],
        }),
      ]))
    }),
  ),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--model", `magnitude/${modelId}`])
  },
})
