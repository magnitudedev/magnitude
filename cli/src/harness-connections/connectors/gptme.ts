import { Data, Effect, Option } from "effect"
import type { HarnessConnectionPaths } from "../paths"
import {
  LOCAL_TOKEN,
  OPENAI_BASE_URL,
  defineConnector,
  launchPlan,
  readOr,
  removeTomlArrayBlock,
  replaceOrAppendTomlArrayBlock,
  tomlTableScalarValue,
  updateTomlTableScalar,
  writeIfChanged,
} from "../shared"

const PROVIDER_TABLE = "providers"
const PROVIDER_NAME = "magnitude"

class GptmeConnectorError extends Data.TaggedError("GptmeConnectorError")<{
  readonly message: string
}> {}

const gptmeProviderBlock = (modelId?: string): string => [
  "[[providers]]",
  `name = "${PROVIDER_NAME}"`,
  `base_url = "${OPENAI_BASE_URL}"`,
  `api_key = "${LOCAL_TOKEN}"`,
  ...(modelId !== undefined ? [`default_model = "${modelId}"`] : []),
].join("\n") + "\n"

/** Parse the [[providers]] array from a TOML source, returning undefined on any error. */
const parseProviders = (source: string): Array<Record<string, unknown>> | undefined => {
  try {
    const parsed = Bun.TOML.parse(source)
    const providers = (parsed as Record<string, unknown>)?.providers
    return Array.isArray(providers) ? (providers as Array<Record<string, unknown>>) : undefined
  } catch {
    return undefined
  }
}

export const makeGptmeConnector = (paths: HarnessConnectionPaths) => defineConnector({
  id: "gptme",
  name: "gptme",
  executable: "gptme",
  skillInstallationTarget: "shared-agents",
  configurationFiles: [paths.gptme],
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.gptme, "")
    // Guard: fail if a user-owned [[providers]] name = "magnitude" block exists
    // (i.e. a block we don't own — different base_url or api_key)
    const providers = parseProviders(source)
    if (providers !== undefined) {
      const existing = providers.find((p) => p["name"] === PROVIDER_NAME)
      if (
        existing !== undefined &&
        (existing["base_url"] !== OPENAI_BASE_URL || existing["api_key"] !== LOCAL_TOKEN)
      ) {
        return yield* new GptmeConnectorError({
          message: `A "[[providers]] name = \\"${PROVIDER_NAME}\\"" block already exists in the gptme config with a different base_url or api_key. Remove it manually before connecting Magnitude.`,
        })
      }
    }
    const modelId = Option.isSome(spec.model) ? spec.model.value : undefined
    const block = gptmeProviderBlock(modelId)
    let next = replaceOrAppendTomlArrayBlock(source, PROVIDER_TABLE, PROVIDER_NAME, block)
    const restore = Option.map(spec.model, () => ({
      model: (() => {
        const previous = tomlTableScalarValue(source, "models", "default")
        return typeof previous === "string" && previous.length > 0
          ? Option.some(previous)
          : Option.none<string>()
      })(),
    }))
    if (Option.isSome(spec.model)) {
      next = updateTomlTableScalar(next, "models", [["default", `magnitude/${spec.model.value}`]])
    }
    yield* writeIfChanged(paths.gptme, source, next)
    return restore
  }),
  disconnect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.gptme, "")
    const currentDefault = tomlTableScalarValue(source, "models", "default")
    let next = removeTomlArrayBlock(source, PROVIDER_TABLE, PROVIDER_NAME)
    if (
      typeof currentDefault === "string" &&
      currentDefault.startsWith("magnitude/") &&
      Option.isSome(spec.restore)
    ) {
      const previous = Option.flatMap(spec.restore.value.model, Option.some)
      next = updateTomlTableScalar(next, "models", [
        ["default", Option.isSome(previous) ? previous.value : undefined],
      ])
    } else if (
      typeof currentDefault === "string" &&
      currentDefault.startsWith("magnitude/") &&
      Option.isNone(spec.restore)
    ) {
      next = updateTomlTableScalar(next, "models", [["default", undefined]])
    }
    yield* writeIfChanged(paths.gptme, source, next)
  }),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--model", `magnitude/${modelId}`])
  },
})
