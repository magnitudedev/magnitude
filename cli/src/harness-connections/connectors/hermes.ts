import { Effect, Option } from "effect"
import { parseDocument } from "yaml"
import type { HarnessConnectionPaths } from "../paths"
import {
  LOCAL_TOKEN,
  OPENAI_BASE_URL,
  defineConnector,
  launchPlan,
  ownedWhen,
  readOr,
  removeOwnedYaml,
  updateYaml,
  valueAt,
  writeIfChanged,
} from "../shared"

export const hermesProviderConfig = () => ({
  name: "Magnitude",
  base_url: OPENAI_BASE_URL,
  api_key: LOCAL_TOKEN,
  transport: "chat_completions",
})

const isManagedProvider = (value: unknown, currentModel: Option.Option<string>): boolean => {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false
  const provider = value as Record<string, unknown>
  const expected = hermesProviderConfig()
  const allowedKeys = new Set([...Object.keys(expected), "default_model"])
  return Object.keys(provider).every((key) => allowedKeys.has(key))
    && Object.entries(expected).every(([key, expectedValue]) => provider[key] === expectedValue)
    && (provider.default_model === undefined || Option.contains(currentModel, provider.default_model))
}

export const makeHermesConnector = (paths: HarnessConnectionPaths) => defineConnector({
  id: "hermes",
  name: "Hermes",
  executable: "hermes",
  skillInstallationTarget: "hermes-user",
  configurationFiles: [paths.hermes],
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.hermes, "{}\n")
    const document = parseDocument(source)
    if (document.errors.length > 0) throw new Error("Hermes configuration is not valid YAML")
    const existing = valueAt(document.toJS(), ["providers", "magnitude"])
    if (existing !== undefined && !isManagedProvider(existing, spec.setCurrent)) {
      throw new Error("Hermes already contains a conflicting providers.magnitude")
    }
    const changes: Array<readonly [ReadonlyArray<string>, unknown]> = [[
      ["providers", "magnitude"], hermesProviderConfig(),
    ]]
    if (Option.isSome(spec.setCurrent)) changes.push(
      [["model", "default"], spec.setCurrent.value],
      [["model", "provider"], "custom:magnitude"],
      [["model", "base_url"], OPENAI_BASE_URL],
      [["model", "api_mode"], "chat_completions"],
    )
    yield* writeIfChanged(paths.hermes, source, updateYaml(source, changes))
  }),
  disconnect: (spec) => readOr(paths.hermes, "{}\n").pipe(
    Effect.flatMap((source) => {
      const provider = valueAt(parseDocument(source).toJS(), ["providers", "magnitude"])
      return writeIfChanged(paths.hermes, source, removeOwnedYaml(source, [
        ...ownedWhen(["providers", "magnitude"], provider, isManagedProvider(provider, spec.setCurrent)),
        ...Option.match(spec.setCurrent, {
          onNone: () => [],
          onSome: (modelId) => [
            [["model", "default"], modelId] as const,
            [["model", "provider"], "custom:magnitude"] as const,
            [["model", "base_url"], OPENAI_BASE_URL] as const,
            [["model", "api_mode"], "chat_completions"] as const,
          ],
        }),
      ]))
    }),
  ),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--provider", "custom:magnitude", "--model", modelId])
  },
})
